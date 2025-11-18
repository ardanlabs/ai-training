// This example shows you a web service that provides a chat endpoint for asking
// questions about the Go notebook. It uses the code from step3 for the RAG
// aspects of the application. The code also provides an embedded react app
// that can be used to interact with the chat endpoint. The react app is built
// using vite and the code is in the app directory.
//
// # If you want to rebuild the react app:
//
//	$ make example13-step4-npm-install
//	$ make example13-step4-npm-build
//
// # Running the example:
//
//	$ make example13-step4
//
// # CURL call
//
//	$ make example13-step4-curl

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/duck"
	"github.com/ardanlabs/ai-training/cmd/examples/example13/install"
	"github.com/ardanlabs/ai-training/cmd/examples/example13/step4/website"
	"github.com/ardanlabs/llamacpp"
	"github.com/hybridgroup/yzma/pkg/download"
)

const (
	modelChatURL       = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true"
	modelEmbedURL      = "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf?download=true"
	libPath            = "zarf/llamacpp"
	modelPath          = "zarf/models"
	dbPath             = "zarf/data/duck-ex13-step3.db" // ":memory:"
	chunksFile         = "zarf/data/book.chunks"
	dimentions         = 768
	WebReadTimeout     = 10 * time.Second
	WebWriteTimeout    = 120 * time.Second
	WebIdleTimeout     = 120 * time.Second
	WebShutdownTimeout = 20 * time.Second
	WebAPIHost         = "0.0.0.0:3000"
)

func main() {
	log.Default().SetOutput(os.Stdout)

	if err := run(); err != nil {
		fmt.Println(err)
		os.Exit(1)
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

	fmt.Println()
	fmt.Println("startup: status: initializing V1 API support")

	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, syscall.SIGINT, syscall.SIGTERM)

	cfg := website.Config{
		LLMEmbed: llmEmbed,
		DBMChat:  llmChat,
		DB:       db,
	}

	api := http.Server{
		Addr:         WebAPIHost,
		Handler:      website.WebAPI(cfg),
		ReadTimeout:  WebReadTimeout,
		WriteTimeout: WebWriteTimeout,
		IdleTimeout:  WebIdleTimeout,
	}

	serverErrors := make(chan error, 1)

	go func() {
		fmt.Println("startup: status: api router and website started: host", api.Addr)
		serverErrors <- api.ListenAndServe()
	}()

	select {
	case err := <-serverErrors:
		return fmt.Errorf("server error: %w", err)

	case sig := <-shutdown:
		fmt.Println("shutdown: status: shutdown started: signal", sig)
		defer fmt.Println("shutdown: status: shutdown complete: signal", sig)

		ctx, cancel := context.WithTimeout(context.Background(), WebShutdownTimeout)
		defer cancel()

		if err := api.Shutdown(ctx); err != nil {
			api.Close()
			return fmt.Errorf("could not stop server gracefully: %w", err)
		}
	}

	return nil
}
