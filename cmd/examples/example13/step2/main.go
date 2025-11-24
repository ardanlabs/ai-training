// This example shows you how to use yzma to execute a simple prompt
// against a vision model using llamacpp directly via yzma and a native Go
// application.
//
// # Running the example:
//
//	$ make example13-step2

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/install"
	"github.com/ardanlabs/kronk"
	"github.com/hybridgroup/yzma/pkg/download"
)

const (
	modelURL  = "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true"
	projURL   = "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true"
	imageFile = "zarf/samples/gallery/giraffe.jpg"
	libPath   = "zarf/llamacpp"
	modelPath = "zarf/models"
)

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelFile, projFile, err := installSystem()
	if err != nil {
		return fmt.Errorf("unable to install system: %w", err)
	}

	krn, err := newKronk(modelFile, projFile)
	if err != nil {
		return fmt.Errorf("unable to init kronk: %w", err)
	}
	defer krn.Unload()

	// -------------------------------------------------------------------------

	question := "What is in this picture?"

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	ch, err := performChat(ctx, krn, question, imageFile)
	if err != nil {
		return fmt.Errorf("perform chat: %w", err)
	}

	if err := modelResponse(krn, ch); err != nil {
		return fmt.Errorf("model response: %w", err)
	}

	return nil
}

func installSystem() (string, string, error) {
	if err := install.LlamaCPP(libPath, download.CPU, true); err != nil {
		return "", "", fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelFile, err := install.Model(modelURL, modelPath)
	if err != nil {
		return "", "", fmt.Errorf("unable to install model: %w", err)
	}

	projFile, err := install.Model(projURL, modelPath)
	if err != nil {
		return "", "", fmt.Errorf("unable to install model: %w", err)
	}

	return modelFile, projFile, nil
}

func newKronk(modelFile string, projFile string) (*kronk.Kronk, error) {
	if err := kronk.Init(libPath, kronk.LogSilent); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	const concurrency = 1

	krn, err := kronk.New(concurrency, modelFile, projFile, kronk.ModelConfig{
		ContextWindow: 0,
		Embeddings:    false,
	})
	if err != nil {
		return nil, fmt.Errorf("unable to create inference model: %w", err)
	}

	fmt.Println("- contextWindow:", krn.ModelConfig().ContextWindow)
	fmt.Println("- embeddings   :", krn.ModelConfig().Embeddings)

	return krn, nil
}

func performChat(ctx context.Context, krn *kronk.Kronk, question string, imageFile string) (<-chan kronk.ChatResponse, error) {
	fmt.Printf("\nQuestion: %s\n\n", question)

	message := kronk.ChatMessage{
		Role:    "user",
		Content: question,
	}

	ch, err := krn.VisionStreaming(ctx, message, imageFile, kronk.Params{
		TopK:      1.0,
		TopP:      0.9,
		Temp:      0.7,
		MaxTokens: 2048,
	})
	if err != nil {
		return nil, fmt.Errorf("vision streaming: %w", err)
	}

	return ch, nil
}

func modelResponse(krn *kronk.Kronk, ch <-chan kronk.ChatResponse) error {
	fmt.Print("MODEL> ")

	var contextTokens int
	var inputTokens int
	var outputTokens int

	now := time.Now()

	for msg := range ch {
		if msg.Err != nil {
			return fmt.Errorf("error from model: %w", msg.Err)
		}

		fmt.Print(msg.Response)

		contextTokens = msg.Tokens.Context
		inputTokens = msg.Tokens.Input
		outputTokens += msg.Tokens.Output
	}

	// ---------------------------------------------------------------------

	elapsedSeconds := time.Since(now).Seconds()
	tokensPerSecond := float64(outputTokens) / elapsedSeconds

	contextWindow := krn.ModelConfig().ContextWindow
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("\n\n\u001b[90mInput: %d  Output: %d  Context: %d (%.0f%% of %.0fK) TPS: %.2f\u001b[0m\n",
		inputTokens, outputTokens, contextTokens, percentage, of, tokensPerSecond)

	return nil
}
