// This example shows you how to use yzma to execute a simple prompt
// against a model using llamacpp directly via a native Go application.
//
// # Running the example:
//
//	$ make example13-step1

package main

import (
	"fmt"
	"os"

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
	fmt.Println("- check llamacpp installation")
	if err := download.InstallLibraries(libPath, download.CPU, true); err != nil {
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}
	fmt.Println("- llamacpp installed")

	fmt.Println("- check model installation")
	modelFile, err := llamacpp.InstallModel(modelURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install model: %w", err)
	}
	fmt.Printf("- model %q installed\n", modelFile)

	fmt.Println("- load model")
	llm, err := llamacpp.New(libPath, modelFile, llamacpp.Config{
		ContextWindow: 8196,
	})
	if err != nil {
		return fmt.Errorf("unable to load model: %w", err)
	}
	defer llm.Unload()
	fmt.Printf("- model %q loaded\n", modelFile)

	// -------------------------------------------------------------------------

	question := "Write a hello world program in Go?"
	fmt.Printf("Question: %s\n\n", question)

	messages := []llamacpp.ChatMessage{
		{
			Role:    "user",
			Content: question,
		},
	}

	ch := llm.ChatCompletions(messages, llamacpp.Params{
		TopK: 1.0,
		TopP: 0.9,
		Temp: 0.7,
	})

	for msg := range ch {
		if msg.Err != nil {
			return fmt.Errorf("error from model: %w", msg.Err)
		}
		fmt.Print(msg.Response)
	}

	fmt.Println()

	return nil
}
