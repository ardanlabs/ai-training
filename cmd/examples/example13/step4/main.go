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
	modelFile  = "zarf/models/embeddinggemma-300m-qat-Q8_0.gguf"
	dbPath     = "zarf/data/duck.db"
	dimentions = 768
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	log.Default().SetOutput(os.Stdout)

	em, err := NewEmbeddingModel(modelFile)
	if err != nil {
		return fmt.Errorf("error loading embedding model: %w", err)
	}
	defer em.Unload()

	db, err := dbConnection(em, dimentions)
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer db.Close()

	// -------------------------------------------------------------------------

	question := "What do interfaces provide in Go"

	queryVector, err := em.Embed(question)
	if err != nil {
		return fmt.Errorf("error embedding query: %w", err)
	}

	fmt.Printf("\nTop 3 similar items to %q:\n\n", question)

	if err := dbSearch(db, queryVector); err != nil {
		return fmt.Errorf("error searching database: %w", err)
	}

	return nil
}
