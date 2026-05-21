// This example takes step1 and adds the end-to-end RAG query path: a
// hardcoded book-specific question is embedded with the same model used to
// build the corpus, the top-5 most similar chunks are pulled out of pgvector
// via rag.SearchDocuments, the prompt template is populated with the result
// of rag.BuildContext, and the chat model streams the answer back. No
// interactive loop yet — that arrives in step3.
//
// # Running the example
//
//	$ make example10-step2
//
// # Optional environment overrides
//
//	LLM_SERVER    chat completions endpoint (default: http://localhost:11435/v1/chat/completions)
//	LLM_MODEL     chat model name           (default: Qwen3-8B-Q8_0)
//	EMBED_SERVER  embeddings endpoint       (default: http://localhost:11435/v1/embeddings)
//	EMBED_MODEL   embedding model name      (default: embeddinggemma-300m-qat-Q8_0)
//
// # Prerequisites
//
//	$ make compose-up
//	$ make kronk-up
//
// # What this step adds over step1
//
//	The complete embed → search → inject → streamed-answer pipeline, run once
//	against a fixed question so the prompt assembly is easy to trace.

// Example 10 — Step 2 — End-to-End RAG Query
package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/rag"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/jmoiron/sqlx"
)

const (
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5432"
	dbName     = "postgres"
)

var (
	llmURL     = "http://localhost:11435/v1/chat/completions"
	llmModel   = "Qwen3-8B-Q8_0"
	embedURL   = "http://localhost:11435/v1/embeddings"
	embedModel = "embeddinggemma-300m-qat-Q8_0"
)

func init() {
	if v := os.Getenv("LLM_SERVER"); v != "" {
		llmURL = v
	}

	if v := os.Getenv("LLM_MODEL"); v != "" {
		llmModel = v
	}

	if v := os.Getenv("EMBED_SERVER"); v != "" {
		embedURL = v
	}

	if v := os.Getenv("EMBED_MODEL"); v != "" {
		embedModel = v
	}
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	fmt.Printf("\nChat Server:\n%s\n", llmURL)
	fmt.Printf("\nChat Model:\n%s\n", llmModel)
	fmt.Printf("\nEmbedding Server:\n%s\n", embedURL)
	fmt.Printf("\nEmbedding Model:\n%s\n", embedModel)

	chatClient := client.NewLLM(llmURL, llmModel)
	embedClient := client.NewLLM(embedURL, embedModel)

	// -------------------------------------------------------------------------
	// Connect to PostgreSQL.
	// region Connect to PostgreSQL.

	db, err := sqldb.Open(sqldb.Config{
		User:         dbUser,
		Password:     dbPassword,
		Host:         dbHost,
		Name:         dbName,
		Schema:       "public",
		MaxIdleConns: 2,
		MaxOpenConns: 5,
		DisableTLS:   true,
	})
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer db.Close()

	if err := sqldb.StatusCheck(ctx, db); err != nil {
		return fmt.Errorf("status check: %w", err)
	}

	fmt.Print("\nPostgreSQL connected.\n")

	// endregion

	// -------------------------------------------------------------------------
	// 1) Parse book chunks.
	// region 1) Parse book chunks.

	fmt.Print("\n============================================================\n")
	fmt.Print("1) Parse Book Chunks\n")
	fmt.Print("============================================================\n")

	data, err := os.ReadFile("zarf/data/book.chunks")
	if err != nil {
		return fmt.Errorf("read chunks file: %w", err)
	}

	re := regexp.MustCompile(`<CHUNK>([\s\S]*?)</CHUNK>`)
	matches := re.FindAllStringSubmatch(string(data), -1)

	chunks := make([]string, len(matches))
	for i, m := range matches {
		chunks[i] = strings.TrimSpace(m[1])
	}

	fmt.Printf("\nTotal chunks found: %d\n", len(chunks))
	fmt.Printf("\nSample chunk (first 200 chars):\n\n%.200s...\n", chunks[0])

	// endregion

	// -------------------------------------------------------------------------
	// 2) Embed each chunk and store in pgvector (warm-start aware).
	//
	// If the documents table already has len(chunks) rows, skip ingestion
	// so step2 can be re-run cheaply after step1 has already populated it.
	// region 2) Embed each chunk and store in pgvector (warm-start aware).

	fmt.Print("\n============================================================\n")
	fmt.Print("2) Embed Chunks & Store in pgvector\n")
	fmt.Print("============================================================\n")

	if err := sqldb.ExecContext(ctx, db, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return fmt.Errorf("create extension vector: %w", err)
	}

	existing, err := existingDocumentsCount(ctx, db)
	if err != nil {
		return fmt.Errorf("check existing documents: %w", err)
	}

	switch {
	case existing == len(chunks):
		fmt.Printf("\nReusing %d existing documents (warm start, skipping ingestion).\n", existing)

	default:
		if existing > 0 {
			fmt.Printf("\nFound %d existing documents but corpus has %d chunks — re-ingesting.\n", existing, len(chunks))
		}

		firstEmbedding, err := embedClient.EmbedText(ctx, chunks[0])
		if err != nil {
			return fmt.Errorf("embed first chunk: %w", err)
		}

		dimensions := len(firstEmbedding)
		fmt.Printf("\nEmbedding dimensions: %d\n", dimensions)

		if err := initDB(ctx, db, dimensions); err != nil {
			return err
		}

		documents := make([]rag.Document, 0, len(chunks))
		documents = append(documents, rag.Document{
			ID:        0,
			Name:      fmt.Sprintf("Chunk %d", 0),
			Text:      chunks[0],
			Embedding: firstEmbedding,
		})

		fmt.Print("\nEmbedding chunks:\n\n")

		for i := 1; i < len(chunks); i++ {
			fmt.Printf("\r  Processing: %d of %d   ", i+1, len(chunks))

			embedding, err := embedClient.EmbedText(ctx, chunks[i])
			if err != nil {
				return fmt.Errorf("embed chunk %d: %w", i, err)
			}

			documents = append(documents, rag.Document{
				ID:        i,
				Name:      fmt.Sprintf("Chunk %d", i),
				Text:      chunks[i],
				Embedding: embedding,
			})
		}

		fmt.Print("\n")

		if err := insertDocuments(ctx, db, documents); err != nil {
			return err
		}

		fmt.Printf("\n%d documents stored in pgvector.\n", len(documents))
	}

	// endregion

	// -------------------------------------------------------------------------
	// 3) End-to-end RAG query: embed the question, retrieve top-5 chunks,
	// build the context block, inject it into the prompt template, and
	// stream the answer back from the chat model.
	// region 3) End-to-end RAG query: embed the question, retrieve top-5 chunks,

	fmt.Print("\n============================================================\n")
	fmt.Print("3) End-to-End RAG Query\n")
	fmt.Print("============================================================\n")

	const question = `According to the Ultimate Go Notebook, what is Bill Kennedy's "never, ever, never" rule about switching between data semantics in a call chain, and what does he say is never safe?`

	results, err := rag.SearchDocuments(ctx, db, embedClient, question, 5)
	if err != nil {
		return fmt.Errorf("search documents: %w", err)
	}

	printResults(results)

	contextText := rag.BuildContext(results)

	const prompt = `/no_think

Use the following pieces of information to answer the user's question.
If you don't know the answer, say that you don't know.

Context: %s

Question: %s

Answer the question and provide additional helpful information, but be concise.

Responses should be properly formatted to be easily read.
`

	finalPrompt := fmt.Sprintf(prompt, contextText, question)

	ch, err := chatClient.ChatCompletionsSSE(ctx, finalPrompt)
	if err != nil {
		return fmt.Errorf("chat completions sse: %w", err)
	}

	fmt.Print("\n==============================\n")
	fmt.Print("-----\n")
	fmt.Printf("> Prompt: %s\n", question)
	fmt.Print("-----\n")
	fmt.Print("> Response: ")

	for resp := range ch {
		if len(resp.Choices) == 0 {
			continue
		}

		fmt.Print(resp.Choices[0].Delta.Content)
	}

	fmt.Print("\n-----\n")

	// endregion

	return nil
}

// =============================================================================

func initDB(ctx context.Context, db *sqlx.DB, dimensions int) error {
	if err := sqldb.ExecContext(ctx, db, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return fmt.Errorf("create extension vector: %w", err)
	}

	if err := sqldb.ExecContext(ctx, db, `DROP TABLE IF EXISTS documents`); err != nil {
		return fmt.Errorf("drop table: %w", err)
	}

	query := fmt.Sprintf(`
CREATE TABLE documents (
	id        BIGINT PRIMARY KEY,
	name      TEXT NOT NULL,
	text      TEXT NOT NULL,
	embedding VECTOR(%d) NOT NULL
)`, dimensions)

	if err := sqldb.ExecContext(ctx, db, query); err != nil {
		return fmt.Errorf("create table: %w", err)
	}

	fmt.Print("\nTable 'documents' created with pgvector column.\n")

	return nil
}

// existingDocumentsCount returns the number of rows in the documents table,
// or 0 if the table does not exist yet. Used by the warm-start check.
func existingDocumentsCount(ctx context.Context, db *sqlx.DB) (int, error) {
	const stmt = `
SELECT COALESCE((
	SELECT COUNT(*)::int FROM documents
), 0)
WHERE EXISTS (
	SELECT 1 FROM information_schema.tables WHERE table_name = 'documents'
)`

	var count int
	if err := db.GetContext(ctx, &count, stmt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return 0, nil
		}
		return 0, err
	}

	return count, nil
}

func insertDocuments(ctx context.Context, db *sqlx.DB, documents []rag.Document) error {
	const query = `
INSERT INTO documents (id, name, text, embedding)
VALUES ($1, $2, $3, $4::vector)
`

	for _, doc := range documents {
		if _, err := db.ExecContext(ctx, query, doc.ID, doc.Name, doc.Text, vector.FormatPGVector(doc.Embedding)); err != nil {
			return fmt.Errorf("insert document %d: %w", doc.ID, err)
		}
	}

	return nil
}

func printResults(results []rag.SearchResult) {
	fmt.Print("\nTop matching chunks:\n\n")

	for i, r := range results {
		preview := r.Text
		if len(preview) > 120 {
			preview = preview[:120] + "..."
		}

		fmt.Printf("%d. %s (similarity: %.2f%%) %s\n\n", i+1, r.Name, r.Similarity*100, preview)
	}
}
