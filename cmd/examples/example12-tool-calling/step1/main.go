// This step is the ingestion bootstrap for example12 — NOT a tool-calling
// step. It parses the Ultimate Go Notebook chunks, embeds them in parallel
// via errgroup, and stores them in pgvector. The model is not contacted
// with any tool definitions yet.
//
// Why it lives here: step2 and step3 (the actual tool-calling lesson) need
// a populated `documents` table so `tool_search_book` has something to
// search. Run this once; steps 2 and 3 reuse the resulting table.
//
// The code below is carried over from example11-rag-perf (parallel
// embedding via errgroup) with no new ingestion concept. The tool-calling
// material starts in step2.
//
// # Running the example
//
//	$ make example12-step1
//
// # Optional environment overrides
//
//	EMBED_SERVER   embeddings endpoint  (default: http://localhost:11435/v1/embeddings)
//	EMBED_MODEL    embedding model name (default: embeddinggemma-300m-qat-Q8_0)
//
// # Prerequisites
//
//   - make compose-up
//   - make kronk-up

// Example 12 — Step 1 — Ingestion Bootstrap (prerequisite for steps 2-3)
package main

import (
	"context"
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
	"golang.org/x/sync/errgroup"
)

const (
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5432"
	dbName     = "postgres"
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
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

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

	fmt.Printf("\nTotal chunks: %d\n", len(chunks))

	firstEmbedding, err := embedClient.EmbedText(ctx, chunks[0])
	if err != nil {
		return fmt.Errorf("embed first chunk: %w", err)
	}

	dimensions := len(firstEmbedding)
	fmt.Printf("\nEmbedding dimensions: %d\n", dimensions)

	// endregion

	// -------------------------------------------------------------------------
	// 2) Parallel embedding with errgroup (carried over from example11).
	// region 2) Parallel embedding with errgroup (carried over from example11).

	fmt.Print("\n============================================================\n")
	fmt.Print("2) Parallel Embedding (errgroup, concurrency=5)\n")
	fmt.Print("============================================================\n")

	const concurrency = 5

	parStart := time.Now()

	parDocs := make([]rag.Document, len(chunks))
	parDocs[0] = rag.Document{ID: 0, Name: fmt.Sprintf("Chunk %d", 0), Text: chunks[0], Embedding: firstEmbedding}

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(concurrency)

	for i := 1; i < len(chunks); i++ {
		g.Go(func() error {
			embedding, err := embedClient.EmbedText(gCtx, chunks[i])
			if err != nil {
				return fmt.Errorf("parallel embed chunk %d: %w", i, err)
			}
			parDocs[i] = rag.Document{ID: i, Name: fmt.Sprintf("Chunk %d", i), Text: chunks[i], Embedding: embedding}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	parDuration := time.Since(parStart)
	fmt.Printf("\nParallel (%d workers): %d chunks in %s\n", concurrency, len(chunks), parDuration)

	// endregion

	// -------------------------------------------------------------------------
	// 3) Store the parallel results in pgvector.
	// region 3) Store the parallel results in pgvector.

	fmt.Print("\n============================================================\n")
	fmt.Print("3) Store in pgvector\n")
	fmt.Print("============================================================\n")

	if err := initDB(ctx, db, dimensions); err != nil {
		return err
	}

	if err := insertDocuments(ctx, db, parDocs); err != nil {
		return err
	}

	fmt.Printf("\n%d documents stored.\n", len(parDocs))

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

	fmt.Print("\nTable 'documents' created.\n")

	return nil
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
