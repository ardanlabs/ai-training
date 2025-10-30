package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/marcboeker/go-duckdb/v2" // DuckDB driver
)

func main() {
	log.Default().SetOutput(os.Stdout)

	em, err := NewEmbeddingModel("zarf/models/bge-m3-q8_0.gguf")
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
	fmt.Println("LOADING DATA...")
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

	// Create HNSW index for fast similarity search (using cosine distance)
	if _, err = db.Exec(sql); err != nil {
		log.Fatalf("Error creating HNSW index: %v", err)
	}

	// -------------------------------------------------------------------------
	// Query all items from the db with their embeddings.

	// fmt.Println("Items and their embeddings:")

	// sql = `
	// 	SELECT
	// 		id,
	// 		name,
	// 		embedding
	// 	FROM
	// 		items;
	// `

	// rows, err := db.Query(sql)
	// if err != nil {
	// 	log.Fatalf("Error querying data: %v", err)
	// }
	// defer rows.Close()

	// for rows.Next() {
	// 	var id int
	// 	var name string
	// 	var embeddingRaw []any

	// 	if err := rows.Scan(&id, &name, &embeddingRaw); err != nil {
	// 		log.Fatalf("Error scanning row: %v", err)
	// 	}

	// 	embedding := make([]float32, len(embeddingRaw))
	// 	for i, v := range embeddingRaw {
	// 		embedding[i] = v.(float32)
	// 	}

	// 	fmt.Printf("ID: %d, Name: %s, Embedding: %v\n", id, name, embedding[:1])
	// }

	// -------------------------------------------------------------------------
	// Perform similarity search using HNSW index.

	question := "How do you declare a variable in Go"

	queryVector, err := em.Embed(question)
	if err != nil {
		log.Fatalf("Error embedding query: %v", err)
	}

	// Check query plan to verify index usage
	explainSQL := `
		EXPLAIN SELECT
			id,
			name,
			array_cosine_similarity(embedding, ?::FLOAT[1024]) as similarity
		FROM
			items
		ORDER BY
			array_cosine_distance(embedding, ?::FLOAT[1024])
		LIMIT 3;
	`

	fmt.Println("\nQuery Plan:")
	rows, err := db.Query(explainSQL, queryVector, queryVector)
	if err != nil {
		log.Fatalf("Error explaining query: %v", err)
	}
	for rows.Next() {
		var explainKey, explainValue string
		if err := rows.Scan(&explainKey, &explainValue); err != nil {
			log.Fatalf("Error scanning plan: %v", err)
		}
		fmt.Printf("%s: %s\n", explainKey, explainValue)
	}
	rows.Close()

	// -------------------------------------------------------------------------
	// Perform similarity search using HNSW index.

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

		fmt.Printf("ID: %d, Name: %s, Similarity: %.4f\n", id, name[:100], similarity)
	}
}
