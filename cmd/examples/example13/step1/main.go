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
	modelFile, err := installSystem()
	if err != nil {
		return fmt.Errorf("unable to installation system: %w", err)
	}

	krn, err := newKronk(modelFile)
	if err != nil {
		return fmt.Errorf("unable to init kronk: %w", err)
	}
	defer krn.Unload()

	// -------------------------------------------------------------------------

	var messages []kronk.ChatMessage

	for {
		messages, err = userInput(messages)
		if err != nil {
			return fmt.Errorf("user input: %w", err)
		}

		messages, err = func() ([]kronk.ChatMessage, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			ch, err := performChat(ctx, krn, messages)
			if err != nil {
				return nil, fmt.Errorf("unable to perform chat: %w", err)
			}

			messages, err = modelResponse(krn, messages, ch)
			if err != nil {
				return nil, fmt.Errorf("model response: %w", err)
			}

			return messages, nil
		}()

		if err != nil {
			return fmt.Errorf("unable to perform chat: %w", err)
		}
	}
}

func installSystem() (string, error) {
	if err := install.LlamaCPP(libPath, download.CPU, true); err != nil {
		return "", fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelFile, err := install.Model(modelURL, modelPath)
	if err != nil {
		return "", fmt.Errorf("unable to install model: %w", err)
	}

	return modelFile, nil
}

func newKronk(modelFile string) (*kronk.Kronk, error) {
	if err := kronk.Init(libPath, kronk.LogSilent); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	const concurrency = 1

	krn, err := kronk.New(concurrency, modelFile, "", kronk.ModelConfig{
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

func userInput(messages []kronk.ChatMessage) ([]kronk.ChatMessage, error) {
	fmt.Print("\nUSER> ")

	reader := bufio.NewReader(os.Stdin)

	userInput, err := reader.ReadString('\n')
	if err != nil {
		return messages, fmt.Errorf("unable to read user input: %w", err)
	}

	messages = append(messages, kronk.ChatMessage{
		Role:    "user",
		Content: userInput,
	})

	return messages, nil
}

func performChat(ctx context.Context, krn *kronk.Kronk, messages []kronk.ChatMessage) (<-chan kronk.ChatResponse, error) {
	ch, err := krn.ChatStreaming(ctx, messages, kronk.Params{
		TopK:      1.0,
		TopP:      0.9,
		Temp:      0.7,
		MaxTokens: 2048,
	})
	if err != nil {
		return nil, fmt.Errorf("chat streaming: %w", err)
	}

	return ch, nil
}

func modelResponse(krn *kronk.Kronk, messages []kronk.ChatMessage, ch <-chan kronk.ChatResponse) ([]kronk.ChatMessage, error) {
	fmt.Print("\nMODEL> ")

	var finalResponse strings.Builder
	var contextTokens int
	var inputTokens int
	var outputTokens int

	now := time.Now()

	for msg := range ch {
		if msg.Err != nil {
			return messages, fmt.Errorf("error from model: %w", msg.Err)
		}

		fmt.Print(msg.Response)
		finalResponse.WriteString(msg.Response)

		contextTokens = msg.Tokens.Context
		inputTokens = msg.Tokens.Input
		outputTokens += msg.Tokens.Output
	}

	// -------------------------------------------------------------------------

	elapsedSeconds := time.Since(now).Seconds()
	tokensPerSecond := float64(outputTokens) / elapsedSeconds

	contextWindow := krn.ModelConfig().ContextWindow
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("\n\n\u001b[90mInput: %d  Output: %d  Context: %d (%.0f%% of %.0fK) TPS: %.2f\u001b[0m\n",
		inputTokens, outputTokens, contextTokens, percentage, of, tokensPerSecond)

	messages = append(messages, kronk.ChatMessage{
		Role:    "assistant",
		Content: finalResponse.String(),
	})

	return messages, nil
}
