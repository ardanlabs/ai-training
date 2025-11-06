// This example shows you how to use duckDB as an embedding DB and an
// inference model to generate embeddings for a set of items all contained
// in a single Go binary.
//
// # Running the example:
//
//	$ make example13-step4
//
// # This requires running the following command:
//
//	$ make yzma-models // This downloads the needed models

package main

import (
	"fmt"
	"log"
	"os"
)

var (
	modelFile = "zarf/models/embeddinggemma-300m-qat-Q8_0.gguf"
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

	question := "What do interfaces provide in Go"

	queryVector, err := em.Embed(question)
	if err != nil {
		fmt.Printf("Error embedding query: %v", err)
		os.Exit(1)
	}

	fmt.Printf("\nTop 3 similar items to %q:\n\n", question)

	sql := `
		SELECT
			id,
			text,
			array_cosine_similarity(embedding, ?::FLOAT[768]) as similarity
		FROM
			items
		ORDER BY
			similarity DESC
		LIMIT 6;
	`

	rows, err := db.Query(sql, queryVector)
	if err != nil {
		fmt.Printf("Error querying similar items: %v", err)
		os.Exit(1)
	}
	defer rows.Close()

	for rows.Next() {
		var id int
		var text string
		var similarity float64

		if err := rows.Scan(&id, &text, &similarity); err != nil {
			fmt.Printf("Error scanning row: %v", err)
			os.Exit(1)
		}

		fmt.Printf("---\nID: %d\nText: %s...\nSimilarity: %.4f\n\n", id, text[1:min(len(text), 200)], similarity)
	}
}
