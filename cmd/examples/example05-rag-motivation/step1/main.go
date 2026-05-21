// This example asks the model a question grounded in a specific Go book
// (the Ultimate Go Notebook) WITHOUT providing any context. The question
// targets Bill Kennedy's "never, ever, never" rule about switching from
// pointer semantics back to value semantics in a call chain. This is a
// Kennedy-specific stylistic and opinionated rule (not general Go folklore),
// so the base model will not be able to cite the exact phrasing or the
// specific reasoning he gives in the book. The model may produce general
// Go advice that sounds plausible but does not match what the book says.
// This sets up the motivation for retrieval-augmented generation (RAG) in
// step2, where we inject the relevant book excerpt and re-ask the same
// question.
//
// # Running the example
//
//	$ make example05-step1
//
// # Optional environment overrides
//
//  LLM_SERVER  chat completions endpoint (default: http://localhost:11435/v1/chat/completions)
//  LLM_MODEL   chat model name           (default: Qwen3-8B-Q8_0)
//
// # Prerequisites
//
//	$ make kronk-up

// Example 05 — Step 1 — Question Without Context
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

	const question = `According to the Ultimate Go Notebook, what is Bill Kennedy's "never, ever, never" rule about switching between data semantics in a call chain, and what does he say is never safe?`

	fmt.Printf("\nServer:\n%s\n", llmURL)
	fmt.Printf("\nModel:\n%s\n", llmModel)

	chatClient := client.NewLLM(llmURL, llmModel)

	// -------------------------------------------------------------------------
	// 1) Same question without context.
	// region 1) Same question without context.

	fmt.Print("\n============================================================\n")
	fmt.Print("1) Same Question Without Context\n")
	fmt.Print("============================================================\n")

	ch, err := chatClient.ChatCompletionsSSE(ctx, "/no_think\n\n"+question)
	if err != nil {
		return fmt.Errorf("chat completions sse without context: %w", err)
	}

	fmt.Print("\n==============================\n")
	fmt.Print("-----\n")
	fmt.Printf("> Prompt: %s\n", question)
	fmt.Print("-----\n")
	fmt.Print("> Response: ")

	for resp := range ch {
		if len(resp.Choices) == 0 {
			continue
		}

		fmt.Print(resp.Choices[0].Delta.Content)
	}

	fmt.Print("\n-----\n")

	// endregion

	return nil
}
