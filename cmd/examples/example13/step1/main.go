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
	"github.com/ardanlabs/llamacpp"
	"github.com/hybridgroup/yzma/pkg/download"
)

const (
	modelURL  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf?download=true"
	libPath   = "zarf/llamacpp"
	modelPath = "zarf/models"
)

func main() {
	if err := run(); err != nil {
		fmt.Println(err)
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
		ContextWindow: 1024 * 32,
	})
	if err != nil {
		return fmt.Errorf("unable to create inference model: %w", err)
	}
	defer llm.Unload()

	// -------------------------------------------------------------------------

	var messages []llamacpp.ChatMessage

	for {
		fmt.Print("\nUSER> ")

		reader := bufio.NewReader(os.Stdin)

		userInput, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("unable to read user input", err.Error())
			os.Exit(1)
		}

		messages = append(messages, llamacpp.ChatMessage{
			Role:    "user",
			Content: userInput,
		})

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

		fmt.Print("\nMODEL> ")

		var finalResponse strings.Builder
		for msg := range ch {
			if msg.Err != nil {
				return fmt.Errorf("error from model: %w", msg.Err)
			}

			fmt.Print(msg.Response)
			finalResponse.WriteString(msg.Response)
		}

		messages = append(messages, llamacpp.ChatMessage{
			Role:    "assistant",
			Content: finalResponse.String(),
		})

		fmt.Println()
	}
}
