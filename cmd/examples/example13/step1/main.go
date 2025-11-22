// This example shows you how to create a simple chat application against an
// inference model using llamacpp directly via yzma and a native Go application.
//
// # Running the example:
//
//	$ make example13-step1

package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/install"
	"github.com/ardanlabs/kronk"
	"github.com/hybridgroup/yzma/pkg/download"
)

const (
	// modelURL  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf?download=true"
	// modelURL  = "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q8_0.gguf?download=true"
	modelURL  = "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf?download=true"
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
	if err := install.LlamaCPP(libPath, download.CPU, true); err != nil {
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelFile, err := install.Model(modelURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install model: %w", err)
	}

	// -------------------------------------------------------------------------

	if err := kronk.Init(libPath, kronk.LogSilent); err != nil {
		return fmt.Errorf("unable to init kronk: %w", err)
	}

	const concurrency = 1

	krn, err := kronk.New(concurrency, modelFile, kronk.ModelConfig{
		ContextWindow: 0,
		MaxTokens:     0,
		Embeddings:    false,
	})
	if err != nil {
		return fmt.Errorf("unable to create inference model: %w", err)
	}
	defer krn.Unload()

	fmt.Println("- contextWindow:", krn.ModelConfig().ContextWindow)
	fmt.Println("- maxTokens    :", krn.ModelConfig().MaxTokens)
	fmt.Println("- embeddings   :", krn.ModelConfig().Embeddings)

	// -------------------------------------------------------------------------

	var messages []kronk.ChatMessage

	for {
		fmt.Print("\nUSER> ")

		reader := bufio.NewReader(os.Stdin)

		userInput, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("unable to read user input", err.Error())
			os.Exit(1)
		}

		messages = append(messages, kronk.ChatMessage{
			Role:    "user",
			Content: userInput,
		})

		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()

		params := kronk.Params{
			TopK: 1.0,
			TopP: 0.9,
			Temp: 0.7,
		}

		ch, err := krn.ChatStreaming(ctx, messages, params)
		if err != nil {
			return fmt.Errorf("chat streaming: %w", err)
		}

		fmt.Print("\nMODEL> ")

		var finalResponse strings.Builder

		var contextTokens int
		var inputTokens int
		var outputTokens int

		for msg := range ch {
			if msg.Err != nil {
				return fmt.Errorf("error from model: %w", msg.Err)
			}

			fmt.Print(msg.Response)
			finalResponse.WriteString(msg.Response)

			contextTokens = msg.Tokens.Context
			inputTokens = msg.Tokens.Input
			outputTokens += msg.Tokens.Output
		}

		contextWindow := krn.ModelConfig().ContextWindow
		percentage := (float64(contextTokens) / float64(contextWindow)) * 100
		of := float32(contextWindow) / float32(1024)

		fmt.Printf("\n\n\u001b[90mInput: %d  Output: %d  Context: %d (%.0f%% of %.0fK)\u001b[0m",
			inputTokens, outputTokens, contextTokens, percentage, of)

		messages = append(messages, kronk.ChatMessage{
			Role:    "assistant",
			Content: finalResponse.String(),
		})

		fmt.Println()
	}
}
