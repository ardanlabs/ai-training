// This example shows you how to use yzma to execute a simple prompt
// against a vision model using llamacpp directly via a native Go application.
//
// # Running the example:
//
//	$ make example13-step2
//
// # This requires running the following command:
//
//	$ make yzma-models // This downloads the needed models

package main

import (
	"fmt"
	"os"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
)

var (
	modelURL  = "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true"
	projURL   = "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true"
	imageFile = "zarf/samples/gallery/giraffe.jpg"
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

	projFile, err := llamacpp.InstallModel(projURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install model: %w", err)
	}

	cfg := llamacpp.Config{
		ContextWindow: 4096,
	}

	llm, err := llamacpp.New(libPath, modelFile, cfg, llamacpp.WithProjection(projFile))
	if err != nil {
		return fmt.Errorf("unable to create inference model: %w", err)
	}
	defer llm.Unload()

	llm.ShowModelInfo()

	// -------------------------------------------------------------------------

	question := "What is in this picture?"
	fmt.Printf("Question: %s\n\n", question)

	message := llamacpp.ChatMessage{
		Role:    "user",
		Content: question,
	}

	params := llamacpp.Params{
		TopK: 1.0,
		TopP: 0.9,
		Temp: 0.7,
	}

	ch, err := llm.ChatVision(message, imageFile, params)
	if err != nil {
		return fmt.Errorf("unable to chat vision: %w", err)
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
