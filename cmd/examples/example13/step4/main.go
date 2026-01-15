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
	"github.com/ardanlabs/ai-training/cmd/examples/example13/step4/website"
	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
)

const (
	modelChatURL       = "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf?download=true"
	modelEmbedURL      = "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf?download=true"
	modelInstances     = 1
	dbPath             = "zarf/data/duck-ex13-step3.db" // ":memory:"
	chunksFile         = "zarf/data/book.chunks"
	dimentions         = 768
	WebReadTimeout     = 10 * time.Second
	WebWriteTimeout    = 120 * time.Second
	WebIdleTimeout     = 120 * time.Second
	WebShutdownTimeout = 20 * time.Second
	WebAPIHost         = "0.0.0.0:8080"
)

func main() {
	log.Default().SetOutput(os.Stdout)

	if err := run(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	libs, err := libs.New()
	if err != nil {
		return fmt.Errorf("unable to create libs api: %w", err)
	}

	_, err = libs.Download(ctx, kronk.FmtLogger)
	if err != nil {
		return fmt.Errorf("install-system:unable to install llama.cpp: %w", err)
	}

	mdls, err := models.New()
	if err != nil {
		return fmt.Errorf("unable to create models api: %w", err)
	}

	infoEmbed, err := mdls.Download(context.Background(), kronk.FmtLogger, modelEmbedURL, "")
	if err != nil {
		return fmt.Errorf("unable to install model: %w", err)
	}

	infoChat, err := mdls.Download(context.Background(), kronk.FmtLogger, modelChatURL, "")
	if err != nil {
		return fmt.Errorf("unable to install model: %w", err)
	}

	// -------------------------------------------------------------------------

	if err := kronk.Init(); err != nil {
		return fmt.Errorf("unable to init kronk: %w", err)
	}

	krnEmbed, err := kronk.New(model.Config{
		ModelFiles: infoEmbed.ModelFiles,
	})

	if err != nil {
		return fmt.Errorf("unable to create embedding model: %w", err)
	}

	defer func() {
		if err := krnEmbed.Unload(context.Background()); err != nil {
			fmt.Printf("failed to unload embedding model: %v", err)
		}
	}()

	krnChat, err := kronk.New(model.Config{
		ModelFiles: infoChat.ModelFiles,
		NBatch:     32 * 1024,
	})
	if err != nil {
		return fmt.Errorf("unable to create chat model: %w", err)
	}
	defer func() {
		if err := krnChat.Unload(context.Background()); err != nil {
			fmt.Printf("failed to unload chat model: %v", err)
		}
	}()

	fmt.Print("- system info:\n\t")
	for k, v := range krnChat.SystemInfo() {
		fmt.Printf("%s:%v, ", k, v)
	}
	fmt.Println()

	fmt.Println("- contextWindow:", krnChat.ModelConfig().ContextWindow)
	fmt.Println("- embeddings   :", krnChat.ModelInfo().IsEmbedModel)
	fmt.Println("- isGPT        :", krnChat.ModelInfo().IsGPTModel)

	// -------------------------------------------------------------------------

	db, err := duck.LoadData(dbPath, krnEmbed, dimentions, chunksFile)
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
		KRNEmbed:   krnEmbed,
		KRNChat:    krnChat,
		KRNTimeout: WebWriteTimeout,
		DB:         db,
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
