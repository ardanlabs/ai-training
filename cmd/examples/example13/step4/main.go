package main

import (
	"fmt"
	"log"
	"os"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/duck"
	"github.com/ardanlabs/ai-training/cmd/examples/example13/install"
	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
	"github.com/hybridgroup/yzma/pkg/download"
)

var (
	modelChatURL  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true"
	modelEmbedURL = "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf?download=true"
	libPath       = "zarf/llamacpp"
	modelPath     = "zarf/models"
	dbPath        = "zarf/data/duck-ex13-step3.db" // ":memory:"
	chunksFile    = "zarf/data/book.chunks"
	dimentions    = 768
)

func main() {
	log.Default().SetOutput(os.Stdout)

	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := install.LlamaCPP(libPath, download.CPU, true); err != nil {
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelEmbedFile, err := install.Model(modelEmbedURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install embedding model: %w", err)
	}

	modelChatFile, err := install.Model(modelChatURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install chat model: %w", err)
	}

	// -------------------------------------------------------------------------

	const concurrency = 5

	llmEmbed, err := llamacpp.New(concurrency, libPath, modelEmbedFile, llamacpp.Config{
		ContextWindow: 1024 * 32,
		Embeddings:    true,
	})
	if err != nil {
		return fmt.Errorf("unable to create embedding model: %w", err)
	}
	defer llmEmbed.Unload()

	llmChat, err := llamacpp.New(concurrency, libPath, modelChatFile, llamacpp.Config{
		ContextWindow: 1024 * 32,
	})
	if err != nil {
		return fmt.Errorf("unable to create chat model: %w", err)
	}
	defer llmChat.Unload()

	// -------------------------------------------------------------------------

	db, err := duck.LoadData(dbPath, llmEmbed, dimentions, chunksFile)
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer db.Close()

	// -------------------------------------------------------------------------

	// NOW ADD ENDPOINT TO SERVE UP CHAT.

	return nil
}
