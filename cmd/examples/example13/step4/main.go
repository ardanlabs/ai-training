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

	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
)

var (
	modelURL   = "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf?download=true"
	libPath    = os.Getenv("YZMA_LIB")
	modelPath  = "zarf/models"
	dbPath     = "zarf/data/duck.db" // ":memory:"
	dimentions = 768
)

func main() {
	log.Default().SetOutput(os.Stdout)

	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := llamacpp.InstallLibraries(libPath); err != nil {
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelFile, err := llamacpp.InstallModel(modelURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install model: %w", err)
	}

	fmt.Println("- loading Model", modelFile)
	llm, err := llamacpp.New(libPath, modelFile, llamacpp.Config{
		ContextWindow: 1024 * 32,
		Embeddings:    true,
	})
	if err != nil {
		return fmt.Errorf("unable to create inference model: %w", err)
	}
	defer llm.Unload()

	// -------------------------------------------------------------------------

	db, err := dbConnection(llm, dimentions)
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer db.Close()

	// -------------------------------------------------------------------------

	question := "What do interfaces provide in Go"

	queryVector, err := llm.Embed(question)
	if err != nil {
		return fmt.Errorf("error embedding query: %w", err)
	}

	fmt.Printf("\nTop 3 similar items to %q:\n\n", question)

	if err := dbSearch(db, queryVector); err != nil {
		return fmt.Errorf("error searching database: %w", err)
	}

	return nil
}
