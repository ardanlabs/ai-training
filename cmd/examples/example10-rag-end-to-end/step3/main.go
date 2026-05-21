// This example takes step2 and:
//
//   - skips ingestion when the documents table is already populated for
//     this corpus (warm start, so the REPL is usable instantly on rerun);
//   - runs a deliberately out-of-corpus question so students see what RAG
//     looks like when retrieval has no good match (low similarity, model
//     should refuse rather than hallucinate);
//   - runs a small recall@5 evaluation against 5 hardcoded questions with
//     known expected chunk IDs, so retrieval quality is measured instead
//     of asserted;
//   - then drops into an interactive REPL where each question runs the
//     same embed → search → inject → streamed-answer flow.
//
// Type `quit`, `/exit`, or `/bye` to leave the REPL.
//
// # Running the example
//
//	$ make example10
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
// # What this step adds over step2
//
//   - Warm-start ingestion (skip if table is already populated).
//   - Out-of-corpus / "no good match" demonstration.
//   - Tiny recall@5 evaluation with hardcoded expected chunk IDs.
//   - Interactive REPL reusing the same machinery.

// Example 10 — Step 3 — Interactive RAG REPL with Eval
package main

import (
	"bufio"
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

// evalQuestion is a question with its expected chunk IDs for recall@k. The
// expected IDs were picked by reading the corpus by hand — chunk 40 is the
// "never, ever, never" rule, chunk 57 is interface pollution, etc. If
// retrieval doesn't return them in the top-K, recall@K drops. We report
// honestly; we do not adjust the expected set to make the number look good.
type evalQuestion struct {
	Question         string
	ExpectedChunkIDs []int
}

var evalSet = []evalQuestion{
	{
		Question:         `What is Bill Kennedy's "never, ever, never" rule about switching between data semantics in a call chain?`,
		ExpectedChunkIDs: []int{40},
	},
	{
		Question:         "What is interface pollution in Go and how can it be identified?",
		ExpectedChunkIDs: []int{57},
	},
	{
		Question:         "How does a mutex provide synchronization between goroutines?",
		ExpectedChunkIDs: []int{71},
	},
	{
		Question:         "What does the basic syntax for Go generics look like and what is the `any` constraint?",
		ExpectedChunkIDs: []int{96},
	},
	{
		Question:         "Why would you use a worker pool of goroutines instead of spawning one goroutine per task?",
		ExpectedChunkIDs: []int{134},
	},
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

		if err := ingestChunks(ctx, db, embedClient, chunks); err != nil {
			return fmt.Errorf("ingest chunks: %w", err)
		}
	}

	// endregion

	// -------------------------------------------------------------------------
	// 3) End-to-end RAG query (the canonical motivating question from
	//    example 08, now answered against the full corpus).
	// region 3) End-to-end RAG query (the canonical motivating question from

	fmt.Print("\n============================================================\n")
	fmt.Print("3) End-to-End RAG Query\n")
	fmt.Print("============================================================\n")

	const question = `According to the Ultimate Go Notebook, what is Bill Kennedy's "never, ever, never" rule about switching between data semantics in a call chain, and what does he say is never safe?`

	if err := answerWithRAG(ctx, db, embedClient, chatClient, question); err != nil {
		return err
	}

	// endregion

	// -------------------------------------------------------------------------
	// 4) Out-of-corpus / "no good match" question.
	//
	// The book has nothing about Kubernetes. We run the same retrieval +
	// answer flow so students see (a) low similarity scores in the top
	// hits, and (b) what the model does when the context is irrelevant.
	// The system prompt tells the model to say "I don't know" — whether it
	// actually does is informative.
	// region 4) Out-of-corpus / "no good match" question.

	fmt.Print("\n============================================================\n")
	fmt.Print("4) Out-of-Corpus Question (no good match expected)\n")
	fmt.Print("============================================================\n")

	const oocQuestion = "How does the Kubernetes scheduler decide which pod to place on which node?"

	if err := answerWithRAG(ctx, db, embedClient, chatClient, oocQuestion); err != nil {
		return err
	}

	// endregion

	// -------------------------------------------------------------------------
	// 5) Retrieval eval — recall@5 and top-1 hit rate.
	//
	// For each eval question, embed → search top-5 → check whether the
	// expected chunk IDs appear in the result set. This is the simplest
	// possible retrieval metric and it's intentionally honest: a low
	// recall here means retrieval is bad on these questions, not that the
	// metric is wrong.
	// region 5) Retrieval eval — recall@5 and top-1 hit rate.

	fmt.Print("\n============================================================\n")
	fmt.Print("5) Retrieval Eval (recall@5, top-1 hit rate)\n")
	fmt.Print("============================================================\n")

	if err := runRetrievalEval(ctx, db, embedClient); err != nil {
		return fmt.Errorf("eval: %w", err)
	}

	// endregion

	// -------------------------------------------------------------------------
	// 6) Interactive question loop.
	// region 6) Interactive question loop.

	fmt.Print("\n============================================================\n")
	fmt.Print("6) Interactive Question Loop\n")
	fmt.Print("============================================================\n")

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n==============================\n")
		fmt.Print("-----\n")
		fmt.Print("> Prompt: ")

		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		switch strings.ToLower(input) {
		case "quit", "/quit", "/exit", "/bye":
			return nil
		}

		fmt.Print("-----\n")

		if err := answerWithRAG(ctx, db, embedClient, chatClient, input); err != nil {
			return err
		}
	}

	// endregion

	return nil
}

// =============================================================================

// answerWithRAG runs one full embed → search → inject → streamed-answer
// cycle. Used by section 3, section 4, and the REPL.
func answerWithRAG(ctx context.Context, db *sqlx.DB, embedClient, chatClient *client.LLM, question string) error {
	const prompt = `/no_think

Use the following pieces of information to answer the user's question.
If you don't know the answer, say that you don't know.

Context: %s

Question: %s

Answer the question and provide additional helpful information, but be concise.

Responses should be properly formatted to be easily read.
`

	results, err := rag.SearchDocuments(ctx, db, embedClient, question, 5)
	if err != nil {
		return fmt.Errorf("search documents: %w", err)
	}

	printResults(results)

	contextText := rag.BuildContext(results)
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

	return nil
}

// runRetrievalEval computes recall@5 and top-1 hit rate across the eval
// set. Prints per-question detail and an aggregate at the end. No fudging:
// if the expected chunk doesn't show up, that's a miss.
func runRetrievalEval(ctx context.Context, db *sqlx.DB, embedClient *client.LLM) error {
	const k = 5

	var (
		totalRecall float64
		top1Hits    int
	)

	for i, q := range evalSet {
		results, err := rag.SearchDocuments(ctx, db, embedClient, q.Question, k)
		if err != nil {
			return fmt.Errorf("eval search %d: %w", i, err)
		}

		retrieved := make(map[int]struct{}, len(results))
		for _, r := range results {
			retrieved[r.ID] = struct{}{}
		}

		expected := make(map[int]struct{}, len(q.ExpectedChunkIDs))
		for _, id := range q.ExpectedChunkIDs {
			expected[id] = struct{}{}
		}

		hits := 0
		for id := range expected {
			if _, ok := retrieved[id]; ok {
				hits++
			}
		}

		recall := float64(hits) / float64(len(expected))
		totalRecall += recall

		top1Hit := false
		if len(results) > 0 {
			if _, ok := expected[results[0].ID]; ok {
				top1Hits++
				top1Hit = true
			}
		}

		topIDs := make([]int, 0, len(results))
		for _, r := range results {
			topIDs = append(topIDs, r.ID)
		}

		fmt.Printf("\nQ%d: %s\n", i+1, q.Question)
		fmt.Printf("    expected=%v top5=%v recall@5=%.2f top1_hit=%t\n",
			q.ExpectedChunkIDs, topIDs, recall, top1Hit)
	}

	n := float64(len(evalSet))
	fmt.Printf("\nAggregate: mean recall@%d = %.2f, top-1 hit rate = %.2f (%d/%d)\n",
		k, totalRecall/n, float64(top1Hits)/n, top1Hits, len(evalSet))

	return nil
}

// ingestChunks embeds every chunk sequentially and inserts the results into
// the documents table. The dimension is discovered from the first
// embedding.
func ingestChunks(ctx context.Context, db *sqlx.DB, embedClient *client.LLM, chunks []string) error {
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
