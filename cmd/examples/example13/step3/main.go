// This example shows you how to create a simple chat application against an
// inference model using llamacpp directly via a native Go application.
//
// # Running the example:
//
//	$ make example13-step3
//
// # This requires running the following command:
//
//	$ make yzma-models // This downloads the needed models

package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
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
	if err := llamacpp.InstallLibraries(libPath); err != nil {
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelFile, err := llamacpp.InstallModel(modelURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install model: %w", err)
	}

	fmt.Println("- loading Model", modelFile)
	llm, err := llamacpp.New(libPath, modelFile, llamacpp.Config{
		ContextWindow: 1024 * 32,
	})
	if err != nil {
		return fmt.Errorf("unable to create inference model: %w", err)
	}
	defer llm.Unload()

	// -------------------------------------------------------------------------

	fmt.Println()

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

		ch := llm.ChatCompletions(messages, llamacpp.Params{
			TopK: 1.0,
			TopP: 0.9,
			Temp: 0.7,
		})

		fmt.Print("\nMODEL> ")

		for msg := range ch {
			if msg.Err != nil {
				return fmt.Errorf("error from model: %w", msg.Err)
			}
			fmt.Print(msg.Response)
		}

		fmt.Println()
	}
}
