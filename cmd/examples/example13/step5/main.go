package main

import (
	"fmt"
	"os"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
)

var (
	modelURL  = "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf?download=true"
	libPath   = os.Getenv("YZMA_LIB")
	modelPath = "zarf/models"
)

func main() {
	if err := run(); err != nil {
		fmt.Println("error running example:", err)
		os.Exit(1)
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
		ContextWindow: 8196,
		Embeddings:    true,
	})
	if err != nil {
		return fmt.Errorf("unable to create inference model: %w", err)
	}
	defer llm.Unload()

	// -------------------------------------------------------------------------

	query := "What do interfaces provide in Go"
	documents := []string{
		"Eating is the process of consuming food.",
		"Interfaces in Go provide a way to define a set of methods that a type must implement.",
	}

	rankings, err := llm.Rerank(query, documents)
	if err != nil {
		return fmt.Errorf("error embedding query: %w", err)
	}

	for _, rank := range rankings {
		fmt.Printf("Score %f: Document %s\n", rank.Score, rank.Document)
	}

	return nil
}
