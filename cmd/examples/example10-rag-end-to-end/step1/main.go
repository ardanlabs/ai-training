// This example parses the pre-chunked Ultimate Go Notebook, generates one
// embedding per chunk via the LLM server, and stores the documents into a
// pgvector-backed `documents` table. Search and the streamed LLM answer
// arrive in step2 and the interactive REPL arrives in step3.
//
// # Running the example
//
//	$ make example10-step1
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

// Example 10 — Step 1 — Ingest Book Into pgvector
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

	fmt.Printf("\nEmbedding Server:\n%s\n", embedURL)
	fmt.Printf("\nEmbedding Model:\n%s\n", embedModel)

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
	// 2) Embed each chunk and store in pgvector.
	//
	// Warm-start: if the documents table already exists and holds exactly
	// len(chunks) rows, skip embedding entirely. On the first run this
	// section embeds the whole book; on later runs it returns in
	// milliseconds. Subsequent examples (step2, step3, example 11) rely on
	// this so the REPL is usable without paying ingestion cost every time.
	// region 2) Embed each chunk and store in pgvector.

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

	if existing == len(chunks) {
		fmt.Printf("\nReusing %d existing documents (warm start, skipping ingestion).\n", existing)
		return nil
	}

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
		// No rows returned means the EXISTS clause was false: table not there.
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
