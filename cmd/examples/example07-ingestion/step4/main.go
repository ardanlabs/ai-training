// This example takes step3 and inserts the cached embeddings into a pgvector
// table. It reuses the on-disk cache built in step3, then connects to
// PostgreSQL, (re)creates the table sized to the embedding dimension, and
// streams each document into it.
//
// # Running the example
//
//	$ make example07
//
// # Prerequisites
//
//	$ make compose-up
//	$ make kronk-up
//
// # What this step adds over step3
//
//	pgvector ingestion of the documents produced (or cached) by step3.

// Example 07 — Step 4 — Ingestion
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
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/jmoiron/sqlx"
)

const (
	// Step 04
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5432"
	dbName     = "postgres"
	tableName  = "step05"

	chunksPath = "zarf/data/book.chunks"

	// Step 03
	embeddingsPath = "zarf/data/book.embeddings"
)

// Step 02
var (
	embedURL   = "http://localhost:11435/v1/embeddings"
	embedModel = "embeddinggemma-300m-qat-Q8_0"
)

// Step 02
func init() {
	if v := os.Getenv("EMBED_SERVER"); v != "" {
		embedURL = v
	}

	if v := os.Getenv("EMBED_MODEL"); v != "" {
		embedModel = v
	}
}

// =============================================================================

// Step 02
type document struct {
	ID        int       `json:"id"`
	Name      string    `json:"name"`
	Text      string    `json:"text"`
	Embedding []float64 `json:"embedding"`
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	// Step 02
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	// -------------------------------------------------------------------------
	// Step 04
	// Connect to PostgreSQL.
	// region Step 04

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
	// Load the pre-chunked book from disk.
	// region Load the pre-chunked book from disk.

	fmt.Println("\nLoading Chunks")

	chunks, err := loadChunks(chunksPath)
	if err != nil {
		return fmt.Errorf("loadChunks: %w", err)
	}

	fmt.Printf("Loaded %d chunks from %s\n", len(chunks), chunksPath)

	// endregion

	// -------------------------------------------------------------------------
	// Preview the first chunk so we can see what we're working with.
	// region Preview the first chunk so we can see what we're working with.

	if len(chunks) > 0 {
		preview := chunks[0]
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}

		fmt.Println("\nFirst chunk preview:")
		fmt.Println(preview)
	}

	// endregion

	// -------------------------------------------------------------------------
	// Step 02
	// Generate one embedding per chunk, sequentially, via the LLM server.
	// region Step 02

	fmt.Println("\nGenerating Embeddings")

	embedClient := client.NewLLM(embedURL, embedModel)

	docs, err := createBookEmbeddings(ctx, embedClient, chunks)
	if err != nil {
		return fmt.Errorf("createBookEmbeddings: %w", err)
	}

	// endregion

	// -------------------------------------------------------------------------
	// Step 02
	// Show the dimensionality of what the model returned.
	// region Step 02

	if len(docs) > 0 {
		fmt.Printf("\nFirst embedding has %d dimensions\n", len(docs[0].Embedding))
	}

	// endregion

	// -------------------------------------------------------------------------
	// Step 04
	// Initialize the table sized to the embedding dimension, then insert
	// every document.
	// region Step 04

	fmt.Println("\nInitializing Table")

	if err := initDB(ctx, db, len(docs[0].Embedding)); err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	if err := insertDocuments(ctx, db, docs); err != nil {
		return fmt.Errorf("insertDocuments: %w", err)
	}

	// endregion

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

// Step 02
func createBookEmbeddings(ctx context.Context, llm *client.LLM, chunks []string) ([]document, error) {
	// Step 03
	if docs, err := loadCachedDocuments(embeddingsPath); err == nil {
		fmt.Printf("Cache hit — loaded %d embeddings from %s\n", len(docs), embeddingsPath)
		return docs, nil
	}

	// Step 03
	output, err := os.Create(embeddingsPath)
	if err != nil {
		return nil, fmt.Errorf("create file: %w", err)
	}
	defer output.Close()

	docs := make([]document, len(chunks))

	fmt.Print("\n")

	for i, chunk := range chunks {
		fmt.Printf("\rVectorizing Data: %d of %d   ", i+1, len(chunks))

		embedding, err := llm.EmbedText(ctx, chunk)
		if err != nil {
			return nil, fmt.Errorf("embed chunk %d: %w", i, err)
		}

		docs[i] = document{
			ID:        i,
			Name:      fmt.Sprintf("Chunk %d", i),
			Text:      chunk,
			Embedding: embedding,
		}

		// Step 03
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

// Step 03
func loadCachedDocuments(path string) ([]document, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)

	var docs []document

	for scanner.Scan() {
		var d document
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

// Step 04
func initDB(ctx context.Context, db *sqlx.DB, dimensions int) error {
	if err := sqldb.ExecContext(ctx, db, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return fmt.Errorf("create extension vector: %w", err)
	}

	if err := sqldb.ExecContext(ctx, db, fmt.Sprintf(`DROP TABLE IF EXISTS %s`, tableName)); err != nil {
		return fmt.Errorf("drop table: %w", err)
	}

	query := fmt.Sprintf(`
CREATE TABLE %s (
	id        BIGINT PRIMARY KEY,
	name      TEXT NOT NULL,
	text      TEXT NOT NULL,
	embedding VECTOR(%d) NOT NULL
)`, tableName, dimensions)

	if err := sqldb.ExecContext(ctx, db, query); err != nil {
		return fmt.Errorf("create table: %w", err)
	}

	return nil
}

// Step 04
func insertDocuments(ctx context.Context, db *sqlx.DB, docs []document) error {
	query := fmt.Sprintf(`
INSERT INTO %s (id, name, text, embedding)
VALUES ($1, $2, $3, $4::vector)
`, tableName)

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
