// This example shows you how to use yzma to execute a simple prompt
// against a model using llamacpp directly via a native Go application.
//
// # Running the example:
//
//	$ make example13-step1

// # This requires running the following command:
//
//	$ make yzma-models // This downloads the needed models

package main

import (
	"fmt"
	"os"

	"github.com/hybridgroup/yzma/pkg/llama"
)

var (
	modelURL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true"
	libPath  = os.Getenv("YZMA_LIB")
)

func main() {
	if err := run(); err != nil {
		fmt.Println("error running example:", err)
		os.Exit(1)
	}
}

func run() error {
	if err := InstallLlamaCPP(); err != nil {
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelFile, err := InstallModel(modelURL)
	if err != nil {
		return fmt.Errorf("unable to install model: %w", err)
	}

	fmt.Println("- loading Model", modelFile)
	im, err := NewInferenceModel(libPath, modelFile, Config{
		ContextWindow: 8196,
	})
	if err != nil {
		return fmt.Errorf("unable to create inference model: %w", err)
	}
	defer im.Unload()

	im.ShowModelInfo(im.model)

	// -------------------------------------------------------------------------

	question := "Write a hello world program in Go?"
	fmt.Printf("Question: %s\n\n", question)

	messages := []llama.ChatMessage{
		llama.NewChatMessage("user", question),
	}

	ch := im.ChatCompletions(messages, Params{
		TopK: 1.0,
		TopP: 0.9,
		Temp: 0.7,
	})

	for msg := range ch {
		fmt.Print(msg)
	}

	fmt.Println()

	return nil
}
