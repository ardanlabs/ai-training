// This example shows you how to use PostgreSQL with the pgvector extension as
// a vector database to perform a nearest-neighbor vector search. The example
// will create a table with a VECTOR column, store documents with LLM-generated
// embeddings, and run cosine-distance similarity searches.
//
// # Running the example
//
//	$ make example06
//
// # Optional environment overrides
//
//	EMBED_SERVER  embeddings endpoint   (default: http://localhost:11435/v1/embeddings)
//	EMBED_MODEL   embedding model name  (default: embeddinggemma-300m-qat-Q8_0)
//
// # Prerequisites
//
//	$ make compose-up
//	$ make kronk-up

// Example 06 — Vector DB
package main

import (
	"context"
	"fmt"
	"log"
	"os"
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
	tableName  = "step04"
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
	ID        int
	Name      string
	Text      string
	Embedding []float64
}

type searchResult struct {
	ID         int
	Name       string
	Text       string
	Distance   float64
	Similarity float64
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
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
	// Hand-crafted documents.
	// region Hand-crafted documents.

	documents := []document{
		{ID: 1, Name: "Horse   ", Text: "Animal Female"},
		{ID: 2, Name: "Man     ", Text: "Human  Male   Pants Poor Worker"},
		{ID: 3, Name: "Woman   ", Text: "Human  Female Dress Poor Worker"},
		{ID: 4, Name: "King    ", Text: "Human  Male   Pants Rich Ruler"},
		{ID: 5, Name: "Queen   ", Text: "Human  Female Dress Rich Ruler"},
	}

	// endregion

	// -------------------------------------------------------------------------
	// Generate embeddings and capture the dimensionality.
	// region Generate embeddings and capture the dimensionality.

	fmt.Println("\nGenerating Embeddings")
	fmt.Print("\n")

	for i := range documents {
		embedding, err := embedClient.EmbedText(ctx, documents[i].Text)
		if err != nil {
			return fmt.Errorf("embedding %q: %w", documents[i].Name, err)
		}

		documents[i].Embedding = embedding

		fmt.Printf("Vector: Name(%s) len(%d) %v...%v\n",
			documents[i].Name,
			len(embedding),
			embedding[:2],
			embedding[len(embedding)-2:])
	}

	dimensions := len(documents[0].Embedding)

	// endregion

	// -------------------------------------------------------------------------
	// Initialize the table and insert documents.
	// region Initialize the table and insert documents.

	fmt.Println("\nInitializing Table")

	if err := initDB(ctx, db, dimensions); err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	if err := insertDocuments(ctx, db, documents); err != nil {
		return fmt.Errorf("insertDocuments: %w", err)
	}

	// endregion

	// -------------------------------------------------------------------------
	// Vector search.
	// region Vector search.

	fmt.Print("\n---- VECTOR SEARCH ----\n\n")

	search := func(searchDocument string) {
		fmt.Printf("Searching for: %q\n", searchDocument)

		results, err := vectorSearch(ctx, db, embedClient, searchDocument, 10)
		if err != nil {
			fmt.Printf("error while searching: %v\n", err)
			return
		}

		for _, result := range results {
			fmt.Printf("%s -> %s: %.2f%% similar\n",
				result.Name,
				result.Text,
				result.Similarity*100)
		}

		fmt.Printf("\n\n")
	}

	search("worker")
	search("worker woman")
	search("human worker woman")

	// endregion

	return nil
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

func insertDocuments(ctx context.Context, db *sqlx.DB, documents []document) error {
	query := fmt.Sprintf(`
INSERT INTO %s (id, name, text, embedding)
VALUES ($1, $2, $3, $4::vector)
`, tableName)

	for _, doc := range documents {
		if _, err := db.ExecContext(ctx, query, doc.ID, doc.Name, doc.Text, vector.FormatPGVector(doc.Embedding)); err != nil {
			return fmt.Errorf("insert document %d: %w", doc.ID, err)
		}
	}

	return nil
}

func vectorSearch(ctx context.Context, db *sqlx.DB, llm *client.LLM, searchDocument string, limit int) ([]searchResult, error) {
	embedding, err := llm.EmbedText(ctx, searchDocument)
	if err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}

	query := fmt.Sprintf(`
SELECT
	id,
	name,
	text,
	embedding <=> $1::vector AS distance,
	1 - (embedding <=> $1::vector) AS similarity
FROM
	%s
ORDER BY
	embedding <=> $1::vector
LIMIT $2
`, tableName)

	rows, err := db.QueryContext(ctx, query, vector.FormatPGVector(embedding), limit)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	var results []searchResult
	for rows.Next() {
		var r searchResult
		if err := rows.Scan(&r.ID, &r.Name, &r.Text, &r.Distance, &r.Similarity); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}
		results = append(results, r)
	}

	return results, rows.Err()
}
