// This example is a retrieval-debugging exercise. The point is to look at
// what a vector search actually returns BEFORE any LLM gets involved. The
// program re-ingests the Ultimate Go Notebook chunks into pgvector (using
// the on-disk embedding cache from example07 so re-runs are fast), then
// runs a series of canned queries and prints the ranked results with
// similarity scores. Some queries are well-covered by the book and should
// return obviously relevant chunks. One query is intentionally off-topic
// and should return weak matches — that contrast is the lesson.
//
// Most "the LLM is bad" complaints in production RAG systems are actually
// retrieval problems. Debug retrieval in isolation first.
//
// # Running the example
//
//	$ make example09-step1
//
// # Optional environment overrides
//
//	EMBED_SERVER  embeddings endpoint  (default: http://localhost:11435/v1/embeddings)
//	EMBED_MODEL   embedding model name (default: embeddinggemma-300m-qat-Q8_0)
//
// # Prerequisites
//
//	$ make compose-up
//	$ make kronk-up

// Example 09 — Step 1 — Debug Retrieval Against The Book Corpus
package main

import (
	"bufio"
	"context"
	"encoding/json"
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

	chunksPath     = "zarf/data/book.chunks"
	embeddingsPath = "zarf/data/book.embeddings"
)

var (
	embedURL   = "http://localhost:11435/v1/embeddings"
	embedModel = "embeddinggemma-300m-qat-Q8_0"
)

func init() {
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

	embedClient := client.NewLLM(embedURL, embedModel)

	// -------------------------------------------------------------------------
	// Connect to PostgreSQL.
	// region Connect to PostgreSQL.

	fmt.Println("\nConnecting to PostgreSQL")

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

	// endregion

	// -------------------------------------------------------------------------
	// Re-ingest the book chunks into pgvector. The example is self-contained:
	// it rebuilds the documents table every run. The on-disk embedding cache
	// at zarf/data/book.embeddings (produced by example07) makes the
	// subsequent runs fast.
	// region Re-ingest the book chunks into pgvector. The example is self-contained:

	if err := ingestBook(ctx, db, embedClient); err != nil {
		return fmt.Errorf("ingestBook: %w", err)
	}

	// endregion

	// -------------------------------------------------------------------------
	// Run a set of canned queries and print the ranked retrieval results.
	//
	// Three queries are well-covered by the Ultimate Go Notebook and should
	// surface obviously relevant chunks at the top. The last query is
	// deliberately off-topic — the book says nothing about Rust parsing —
	// so the "top" results should look weak. Reading the similarity scores
	// side-by-side is the entire point of this exercise.
	// region Run a set of canned queries and print the ranked retrieval results.

	queries := []string{
		"Why should you never switch from pointer semantics back to value semantics in a call chain?",
		"When should you use a pointer receiver versus a value receiver on a method?",
		"What is the difference between a goroutine and an operating system thread?",
		"How do I write a recursive descent parser in Rust?", // off-topic
	}

	const topN = 5

	for _, q := range queries {
		fmt.Print("\n============================================================\n")
		fmt.Printf("Query: %s\n", q)
		fmt.Print("============================================================\n")

		results, err := rag.SearchDocuments(ctx, db, embedClient, q, topN)
		if err != nil {
			return fmt.Errorf("search documents: %w", err)
		}

		printResults(results)
	}

	// endregion

	return nil
}

// =============================================================================
// Ingestion (self-contained — adapted from example07-ingestion/step4).

func ingestBook(ctx context.Context, db *sqlx.DB, embedClient *client.LLM) error {
	fmt.Println("\nLoading Chunks")

	chunks, err := loadChunks(chunksPath)
	if err != nil {
		return fmt.Errorf("loadChunks: %w", err)
	}

	fmt.Printf("Loaded %d chunks from %s\n", len(chunks), chunksPath)

	fmt.Println("\nGenerating Embeddings (or loading from cache)")

	docs, err := createBookEmbeddings(ctx, embedClient, chunks)
	if err != nil {
		return fmt.Errorf("createBookEmbeddings: %w", err)
	}

	if len(docs) == 0 {
		return fmt.Errorf("no documents produced")
	}

	dimensions := len(docs[0].Embedding)
	fmt.Printf("\nEmbedding dimensions: %d\n", dimensions)

	fmt.Println("\nInitializing 'documents' Table")

	if err := initDB(ctx, db, dimensions); err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	if err := insertDocuments(ctx, db, docs); err != nil {
		return fmt.Errorf("insertDocuments: %w", err)
	}

	return nil
}

func loadChunks(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	r := regexp.MustCompile(`<CHUNK>[\w\W]*?<\/CHUNK>`)
	raw := r.FindAllString(string(data), -1)

	chunks := make([]string, len(raw))
	for i, c := range raw {
		c = strings.TrimPrefix(c, "<CHUNK>")
		c = strings.TrimSuffix(c, "</CHUNK>")
		chunks[i] = c
	}

	return chunks, nil
}

func createBookEmbeddings(ctx context.Context, llm *client.LLM, chunks []string) ([]rag.Document, error) {
	if docs, err := loadCachedDocuments(embeddingsPath); err == nil {
		fmt.Printf("Cache hit — loaded %d embeddings from %s\n", len(docs), embeddingsPath)
		return docs, nil
	}

	output, err := os.Create(embeddingsPath)
	if err != nil {
		return nil, fmt.Errorf("create file: %w", err)
	}
	defer output.Close()

	docs := make([]rag.Document, len(chunks))

	fmt.Print("\n")

	for i, chunk := range chunks {
		fmt.Printf("\rVectorizing Data: %d of %d   ", i+1, len(chunks))

		embedding, err := llm.EmbedText(ctx, chunk)
		if err != nil {
			return nil, fmt.Errorf("embed chunk %d: %w", i, err)
		}

		docs[i] = rag.Document{
			ID:        i,
			Name:      fmt.Sprintf("Chunk %d", i),
			Text:      chunk,
			Embedding: embedding,
		}

		raw, err := json.Marshal(docs[i])
		if err != nil {
			return nil, fmt.Errorf("marshal: %w", err)
		}

		if _, err := output.Write(raw); err != nil {
			return nil, fmt.Errorf("write: %w", err)
		}

		if _, err := output.Write([]byte{'\n'}); err != nil {
			return nil, fmt.Errorf("write newline: %w", err)
		}
	}

	fmt.Print("\n")

	return docs, nil
}

func loadCachedDocuments(path string) ([]rag.Document, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)

	var docs []rag.Document

	for scanner.Scan() {
		var d rag.Document
		if err := json.Unmarshal(scanner.Bytes(), &d); err != nil {
			return nil, fmt.Errorf("unmarshal: %w", err)
		}

		docs = append(docs, d)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan: %w", err)
	}

	return docs, nil
}

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

	return nil
}

func insertDocuments(ctx context.Context, db *sqlx.DB, docs []rag.Document) error {
	const query = `
INSERT INTO documents (id, name, text, embedding)
VALUES ($1, $2, $3, $4::vector)
`

	fmt.Print("\n")

	for i, d := range docs {
		fmt.Printf("\rInserting Data: %d of %d   ", i+1, len(docs))

		if _, err := db.ExecContext(ctx, query, d.ID, d.Name, d.Text, vector.FormatPGVector(d.Embedding)); err != nil {
			return fmt.Errorf("insert document %d: %w", d.ID, err)
		}
	}

	fmt.Print("\n")

	return nil
}

// =============================================================================

func printResults(results []rag.SearchResult) {
	if len(results) == 0 {
		fmt.Print("\n(no results)\n")
		return
	}

	fmt.Print("\n")

	for i, r := range results {
		preview := strings.ReplaceAll(r.Text, "\n", " ")
		if len(preview) > 180 {
			preview = preview[:180] + "..."
		}

		fmt.Printf("%d. %s  similarity=%.2f%%  distance=%.4f\n", i+1, r.Name, r.Similarity*100, r.Distance)
		fmt.Printf("   %s\n\n", preview)
	}
}
