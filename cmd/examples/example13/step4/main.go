package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/marcboeker/go-duckdb/v2" // DuckDB driver
)

var modelFile = "zarf/models/bge-m3-q8_0.gguf"

func main() {
	log.Default().SetOutput(os.Stdout)

	em, err := NewEmbeddingModel(modelFile)
	if err != nil {
		log.Fatal(err)
	}
	defer em.Unload()

	connector, err := duckdb.NewConnector(":memory:", nil)
	if err != nil {
		log.Fatal(err)
	}

	db := sql.OpenDB(connector)
	defer db.Close()

	// Install and load VSS extension for vector similarity search.
	sql := `
		INSTALL vss; LOAD vss;
	`

	_, err = db.Exec(sql)
	if err != nil {
		log.Fatalf("Error loading VSS extension: %v", err)
	}

	// -------------------------------------------------------------------------

	sql = `
		CREATE TABLE items (
			id        INTEGER   PRIMARY KEY,
			name      VARCHAR,
			embedding FLOAT[1024]
		);
	`

	if _, err = db.Exec(sql); err != nil {
		log.Fatalf("Error creating table: %v", err)
	}

	// -------------------------------------------------------------------------

	fmt.Print("LOADING DATA...")
	t := time.Now()

	if err := loadData(db, em); err != nil {
		log.Fatalf("Error loading data: %v", err)
	}

	fmt.Printf("Loaded data in %v\n", time.Since(t))

	// -------------------------------------------------------------------------

	sql = `
		CREATE INDEX idx_embedding ON items 
		USING HNSW (embedding) 
		WITH (metric = 'cosine');
	`

	if _, err = db.Exec(sql); err != nil {
		log.Fatalf("Error creating HNSW index: %v", err)
	}

	// -------------------------------------------------------------------------

	question := "How do you declare a variable in Go"

	queryVector, err := em.Embed(question)
	if err != nil {
		log.Fatalf("Error embedding query: %v", err)
	}

	fmt.Printf("\nTop 3 similar items to %q:\n", question)

	sql = `
		SELECT
			id,
			name,
			array_cosine_similarity(embedding, ?::FLOAT[1024]) as similarity
		FROM
			items
		ORDER BY
			array_cosine_distance(embedding, ?::FLOAT[1024])
		LIMIT 3;
	`

	rows, err := db.Query(sql, queryVector, queryVector)
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

		fmt.Printf("ID: %d, Name: %s, Similarity: %.4f\n", id, name[:100], similarity)
	}
}
