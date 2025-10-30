package main

import (
	"fmt"
	"log"
	"os"
	// DuckDB driver
)

var (
	modelFile = "zarf/models/bge-m3-q8_0.gguf"
	dbPath    = "zarf/data/duck.db"
)

func main() {
	log.Default().SetOutput(os.Stdout)

	em, err := NewEmbeddingModel(modelFile)
	if err != nil {
		log.Fatal(err)
	}
	defer em.Unload()

	db, err := dbConnection(em)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// -------------------------------------------------------------------------

	question := "How do you declare a variable in Go"

	queryVector, err := em.Embed(question)
	if err != nil {
		log.Fatalf("Error embedding query: %v", err)
	}

	fmt.Printf("\nTop 3 similar items to %q:\n", question)

	sql := `
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
