// This example takes step1 and adds knob-tuning to the retrieval debugging
// exercise. Same self-contained book ingestion, same canned queries — but
// now the program sweeps the two knobs every RAG system has to set:
//
//	K          — how many top results to return.
//	Threshold  — a minimum similarity below which a result is discarded.
//
// For each query, the program fetches the top-10 once, then shows what
// survives at K = {1, 3, 5, 10} and at similarity thresholds = {0.30,
// 0.50, 0.70}. The "good" queries should keep useful chunks even at the
// high threshold. The off-topic query should be cut entirely once the
// threshold rises. That contrast is the lesson: a threshold protects the
// LLM from being handed irrelevant context dressed up with confident
// similarity numbers. Still no LLM is invoked — keep the retrieval
// concerns isolated from generation.
//
// # Running the example
//
//	$ make example09-step2
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
//
// # What this step adds over step1
//
//	K and similarity-threshold sweeps over the same queries, so students
//	can see directly what gets included or cut at each setting.

// Example 09 — Step 2 — Threshold And K Sweeps
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
	// Re-ingest the book chunks. Same self-contained ingestion as step1.
	// region Re-ingest the book chunks. Same self-contained ingestion as step1.

	if err := ingestBook(ctx, db, embedClient); err != nil {
		return fmt.Errorf("ingestBook: %w", err)
	}

	// endregion

	// -------------------------------------------------------------------------
	// Sweep K and similarity threshold for each canned query.
	// region Sweep K and similarity threshold for each canned query.

	queries := []string{
		"Why should you never switch from pointer semantics back to value semantics in a call chain?",
		"When should you use a pointer receiver versus a value receiver on a method?",
		"What is the difference between a goroutine and an operating system thread?",
		"How do I write a recursive descent parser in Rust?", // off-topic
	}

	const fetchN = 10
	kValues := []int{1, 3, 5, 10}
	thresholds := []float64{0.30, 0.50, 0.70}

	for _, q := range queries {
		fmt.Print("\n============================================================\n")
		fmt.Printf("Query: %s\n", q)
		fmt.Print("============================================================\n")

		results, err := rag.SearchDocuments(ctx, db, embedClient, q, fetchN)
		if err != nil {
			return fmt.Errorf("search documents: %w", err)
		}

		printTop(results)
		printKSweep(results, kValues)
		printThresholdSweep(results, thresholds)
	}

	// endregion

	return nil
}

// =============================================================================
// Sweeps.

func printTop(results []rag.SearchResult) {
	fmt.Print("\nTop 10 (unfiltered):\n\n")

	if len(results) == 0 {
		fmt.Print("  (no results)\n")
		return
	}

	for i, r := range results {
		fmt.Printf("  %2d. %-12s  similarity=%.2f%%\n", i+1, r.Name, r.Similarity*100)
	}
}

func printKSweep(results []rag.SearchResult, kValues []int) {
	fmt.Print("\nK sweep — how many top results we keep:\n\n")

	for _, k := range kValues {
		kept := results
		if len(kept) > k {
			kept = kept[:k]
		}

		fmt.Printf("  K=%-2d  kept=%d   ", k, len(kept))
		printNames(kept)
	}
}

func printThresholdSweep(results []rag.SearchResult, thresholds []float64) {
	fmt.Print("\nThreshold sweep — only chunks above similarity threshold survive:\n\n")

	for _, t := range thresholds {
		var kept []rag.SearchResult
		for _, r := range results {
			if r.Similarity >= t {
				kept = append(kept, r)
			}
		}

		fmt.Printf("  threshold=%.2f  kept=%d/%d   ", t, len(kept), len(results))
		printNames(kept)
	}
}

func printNames(results []rag.SearchResult) {
	if len(results) == 0 {
		fmt.Print("(nothing — query rejected)\n")
		return
	}

	names := make([]string, len(results))
	for i, r := range results {
		names[i] = fmt.Sprintf("%s(%.0f%%)", r.Name, r.Similarity*100)
	}

	fmt.Println(strings.Join(names, ", "))
}

// =============================================================================
// Ingestion (self-contained — duplicated from step1 on purpose).

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
