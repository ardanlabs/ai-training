// This example takes step1 and adds a streaming chat completion via SSE
// alongside the non-streaming call. The same prompt is sent twice so you can
// see the latency difference between waiting for the full answer and watching
// tokens arrive in real time.
//
// # Running the example
//
//	$ make example04-step2
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
// # What this step adds over step1
//
//	Streaming chat completion via the SSE channel. Same prompt as step1, but
//	tokens are rendered as they arrive, making perceived latency dramatically
//	lower.

// Example 04 — Step 2 — Streaming Chat
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

	// -------------------------------------------------------------------------
	// Step 02
	// Streaming chat completion via SSE. The channel yields chunks as the
	// model produces them.
	// region Step 02

	ch, err := chatClient.ChatCompletionsSSE(ctx, prompt)
	if err != nil {
		return fmt.Errorf("chat completions sse: %w", err)
	}

	fmt.Print("\nAnswer (Streaming):\n\n")

	for resp := range ch {
		if len(resp.Choices) == 0 {
			continue
		}

		fmt.Print(resp.Choices[0].Delta.Content)
	}

	fmt.Print("\n")

	// endregion

	return nil
}
