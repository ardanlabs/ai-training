// This example shows you how to connect to an OpenAI-compatible LLM server
// from Go and make a standard, non-streaming chat completions request. The
// program submits a single prompt, waits for the full response to come back,
// and prints it once.
//
// # Running the example
//
//	$ make example04-step1
//
// # Optional environment overrides
//
//  LLM_SERVER  chat completions endpoint (default: http://localhost:11435/v1/chat/completions)
//  LLM_MODEL   chat model name           (default: Qwen3-8B-Q8_0)
//
// # Prerequisites
//
//	$ make kronk-up
//
//	First step — establishes the LLM connection and shows the request/response
//	shape using the simplest possible (blocking) chat call.

// Example 04 — Step 1 — Non-Streaming Chat
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

var (
	llmURL   = "http://localhost:11435/v1/chat/completions"
	llmModel = "Qwen3-8B-Q8_0"
)

func init() {
	if v := os.Getenv("LLM_SERVER"); v != "" {
		llmURL = v
	}

	if v := os.Getenv("LLM_MODEL"); v != "" {
		llmModel = v
	}
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	const prompt = `/no_think

What is a goroutine in Go?`

	fmt.Printf("\nServer:\n%s\n", llmURL)
	fmt.Printf("\nModel:\n%s\n", llmModel)
	fmt.Printf("\nPrompt:\n\n%s\n", prompt)

	chatClient := client.NewLLM(llmURL, llmModel)

	// -------------------------------------------------------------------------
	// Non-streaming chat completion. The call blocks until the model has
	// finished generating, then returns the full answer in one string.
	// region Non-streaming chat completion.

	answer, err := chatClient.ChatCompletions(ctx, prompt)
	if err != nil {
		return fmt.Errorf("chat completions: %w", err)
	}

	fmt.Print("\nAnswer (Non-Streaming):\n\n")
	fmt.Println(answer)

	// endregion

	return nil
}
