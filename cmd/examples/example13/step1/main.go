// This example shows you how to use yzma to execute a simple prompt
// against a model using llamacpp directly via a native Go application.
//
// # Running the example:
//
//	$ make example13-step1

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/install"
	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
	"github.com/hybridgroup/yzma/pkg/download"
)

var (
	modelURL  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true"
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
	if err := install.LlamaCPP(libPath, download.CPU, true); err != nil {
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelFile, err := install.Model(modelURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install model: %w", err)
	}

	// -------------------------------------------------------------------------

	const concurrency = 1

	llm, err := llamacpp.New(concurrency, libPath, modelFile, llamacpp.Config{
		ContextWindow: 8196,
	})
	if err != nil {
		return fmt.Errorf("unable to load model: %w", err)
	}
	defer llm.Unload()

	// -------------------------------------------------------------------------

	fmt.Println()

	question := "Write a hello world program in Go?"
	fmt.Printf("Question: %s\n\n", question)

	messages := []llamacpp.ChatMessage{
		{
			Role:    "user",
			Content: question,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	params := llamacpp.Params{
		TopK: 1.0,
		TopP: 0.9,
		Temp: 0.7,
	}

	ch, err := llm.ChatCompletions(ctx, messages, params)
	if err != nil {
		return fmt.Errorf("chat completions: %w", err)
	}

	for msg := range ch {
		if msg.Err != nil {
			return fmt.Errorf("error from model: %w", msg.Err)
		}
		fmt.Print(msg.Response)
	}

	fmt.Println()

	return nil
}
