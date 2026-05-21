// This example ingests the pre-chunked Ultimate Go Notebook into a pgvector
// table. The program loads `zarf/data/book.chunks`, generates one embedding
// per chunk (caching them to disk), creates a `step06` table sized to match
// the embedding model, and streams the cached embeddings in. No question is
// asked yet — search arrives in step2 and the LLM answer in step3.
//
// # Running the example
//
//	$ make example08-step1
//
// # Prerequisites
//
//	$ make compose-up
//	$ make kronk-up

// Example 08 — Step 1 — Ingest Book Into pgvector
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
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5432"
	dbName     = "postgres"
	tableName  = "step06"

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
	// Generate embeddings (cached to disk) for every book chunk.
	// region Generate embeddings (cached to disk) for every book chunk.

	fmt.Println("\nGenerating Embeddings")

	if err := createBookEmbeddings(ctx, embedClient); err != nil {
		return fmt.Errorf("createBookEmbeddings: %w", err)
	}

	// endregion

	// -------------------------------------------------------------------------
	// Initialize the table from the cached embedding dimensionality.
	// region Initialize the table from the cached embedding dimensionality.

	dimensions, err := firstEmbeddingDimensions()
	if err != nil {
		return fmt.Errorf("first embedding dimensions: %w", err)
	}

	fmt.Println("\nInitializing Table")

	if err := initDB(ctx, db, dimensions); err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	// endregion

	// -------------------------------------------------------------------------
	// Stream cached embeddings into pgvector.
	// region Stream cached embeddings into pgvector.

	if err := insertBookEmbeddings(ctx, db); err != nil {
		return fmt.Errorf("insertBookEmbeddings: %w", err)
	}

	// endregion

	return nil
}

func createBookEmbeddings(ctx context.Context, llm *client.LLM) error {
	if _, err := os.Stat(embeddingsPath); err == nil {
		return nil
	}

	data, err := os.ReadFile(chunksPath)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	output, err := os.Create(embeddingsPath)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer output.Close()

	fmt.Print("\n")

	r := regexp.MustCompile(`<CHUNK>[\w\W]*?<\/CHUNK>`)
	chunks := r.FindAllString(string(data), -1)

	for counter, chunk := range chunks {
		fmt.Printf("\rVectorizing Data: %d of %d   ", counter+1, len(chunks))

		chunk = strings.TrimPrefix(chunk, "<CHUNK>")
		chunk = strings.TrimSuffix(chunk, "</CHUNK>")

		embedding, err := llm.EmbedText(ctx, chunk)
		if err != nil {
			return fmt.Errorf("embedding: %w", err)
		}

		doc := document{
			ID:        counter,
			Name:      fmt.Sprintf("Chunk %d", counter),
			Text:      chunk,
			Embedding: embedding,
		}

		raw, err := json.Marshal(doc)
		if err != nil {
			return fmt.Errorf("marshal: %w", err)
		}

		if _, err := output.Write(raw); err != nil {
			return fmt.Errorf("write: %w", err)
		}

		if _, err := output.Write([]byte{'\n'}); err != nil {
			return fmt.Errorf("write crlf: %w", err)
		}
	}

	fmt.Print("\n")

	return nil
}

func firstEmbeddingDimensions() (int, error) {
	f, err := os.Open(embeddingsPath)
	if err != nil {
		return 0, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)

	if !scanner.Scan() {
		return 0, fmt.Errorf("embeddings file is empty")
	}

	var d document
	if err := json.Unmarshal(scanner.Bytes(), &d); err != nil {
		return 0, fmt.Errorf("unmarshal: %w", err)
	}

	return len(d.Embedding), nil
}

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

func insertBookEmbeddings(ctx context.Context, db *sqlx.DB) error {
	input, err := os.Open(embeddingsPath)
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer input.Close()

	query := fmt.Sprintf(`
INSERT INTO %s (id, name, text, embedding)
VALUES ($1, $2, $3, $4::vector)
`, tableName)

	var counter int

	fmt.Print("\n")

	scanner := bufio.NewScanner(input)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)

	for scanner.Scan() {
		counter++

		fmt.Printf("\rInserting Data: %d   ", counter)

		var d document
		if err := json.Unmarshal(scanner.Bytes(), &d); err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		if _, err := db.ExecContext(ctx, query, d.ID, d.Name, d.Text, vector.FormatPGVector(d.Embedding)); err != nil {
			return fmt.Errorf("insert document %d: %w", d.ID, err)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan: %w", err)
	}

	fmt.Print("\n")

	return nil
}
