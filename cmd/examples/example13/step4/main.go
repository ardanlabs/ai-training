package main

import (
	"database/sql"
	"fmt"
	"log"

	"github.com/marcboeker/go-duckdb/v2" // DuckDB driver
)

func main() {
	connector, err := duckdb.NewConnector(":memory:", nil)
	if err != nil {
		log.Fatal(err)
	}

	db := sql.OpenDB(connector)
	defer db.Close()

	// Install and load VSS extension for vector similarity search.
	_, err = db.Exec("INSTALL vss; LOAD vss;")
	if err != nil {
		log.Fatalf("Error loading VSS extension: %v", err)
	}

	// -------------------------------------------------------------------------

	sql := `
		CREATE TABLE items (
			id INTEGER PRIMARY KEY,
			name VARCHAR,
			embedding FLOAT[3]
		);
	`

	if _, err = db.Exec(sql); err != nil {
		log.Fatalf("Error creating table: %v", err)
	}

	sql = `
		INSERT INTO items (id, name, embedding) VALUES
		(1, 'apple', [45.99, 35.98, 25.96]),
		(2, 'orange', [99.45, 99.45, 99.45]),
		(3, 'banana', [0.7, 0.8, 0.9]),
		(4, 'grape', [0.12, 0.22, 0.32]);
	`

	// Insert sample data with vector embeddings.
	if _, err = db.Exec(sql); err != nil {
		log.Fatalf("Error inserting data: %v", err)
	}

	sql = `
		CREATE INDEX idx_embedding ON items 
		USING HNSW (embedding) 
		WITH (metric = 'cosine');
	`

	// Create HNSW index for fast similarity search (using cosine distance)
	if _, err = db.Exec(sql); err != nil {
		log.Fatalf("Error creating HNSW index: %v", err)
	}

	// -------------------------------------------------------------------------
	// Query all items from the db with their embeddings.

	fmt.Println("Items and their embeddings:")

	sql = `
		SELECT id, name, embedding FROM items;
	`

	rows, err := db.Query(sql)
	if err != nil {
		log.Fatalf("Error querying data: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		var id int
		var name string
		var embeddingRaw []any

		if err := rows.Scan(&id, &name, &embeddingRaw); err != nil {
			log.Fatalf("Error scanning row: %v", err)
		}

		embedding := make([]float32, len(embeddingRaw))
		for i, v := range embeddingRaw {
			embedding[i] = v.(float32)
		}

		fmt.Printf("ID: %d, Name: %s, Embedding: %v\n", id, name, embedding)
	}

	// -------------------------------------------------------------------------
	// Perform similarity search using HNSW index.

	queryVector := "[0.11, 0.21, 0.31]"
	fmt.Printf("\nTop 3 similar items to %s:\n", queryVector)

	sql = `
		SELECT id, name, array_cosine_similarity(embedding, ?::FLOAT[3]) as similarity
		FROM items
		ORDER BY array_distance(embedding, ?::FLOAT[3])
		LIMIT 3;
	`

	rows, err = db.Query(sql, queryVector, queryVector)
	if err != nil {
		log.Fatalf("Error querying similar items: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		var id int
		var name string
		var similarity float64

		if err := rows.Scan(&id, &name, &similarity); err != nil {
			log.Fatalf("Error scanning row: %v", err)
		}

		fmt.Printf("ID: %d, Name: %s, Similarity: %.4f\n", id, name, similarity)
	}
}
