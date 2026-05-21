// This example takes step1 and asks the SAME question a second time, this
// time with the relevant excerpt from the Ultimate Go Notebook (section
// 4.4 — Data Semantic Guideline For Struct Types) injected into the
// prompt. With the excerpt in context, the model can quote Kennedy's
// "never, ever, never" rule exactly and explain why he says it is never
// safe to copy a value that a pointer points to. This is the core RAG
// idea: retrieve relevant context, then augment the prompt with it
// before asking the model to answer.
//
// # Running the example
//
//	$ make example05-step2
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
//	Second call to the model with the relevant book excerpt injected as
//	context. Side-by-side with step1's answer, the contrast makes the value
//	of retrieval grounding obvious.

// Example 05 — Step 2 — Question With Context
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

	// Step 02
	const prompt = `/no_think

Use the following pieces of information to answer the user's question.
If you don't know the answer, say that you don't know.

Context: %s

Question: %s

Answer the question and provide additional helpful information, but be concise.

Responses should be properly formatted to be easily read.
`

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

	// -------------------------------------------------------------------------
	// Step 02
	// 2) Same question with the relevant book excerpt injected as context.
	// region Step 02

	fmt.Print("\n============================================================\n")
	fmt.Print("2) Same Question With Context\n")
	fmt.Print("============================================================\n")

	fmt.Printf("\nBook Excerpt:\n\n%s\n", bookExcerpt)

	finalPrompt := fmt.Sprintf(prompt, bookExcerpt, question)

	ch, err = chatClient.ChatCompletionsSSE(ctx, finalPrompt)
	if err != nil {
		return fmt.Errorf("chat completions sse with context: %w", err)
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

// Step 02
const bookExcerpt = `4.4 Data Semantic Guideline For Struct Types

These four methods from the Time package seem to break the rules for data
semantic consistency. They are using pointer semantics, why? Because they
are implementing an interface where the method signature is locked in.
Since the implementation requires a mutation, pointer semantics are the
only choice.

Here is a guideline: If value semantics are at play, I can switch to
pointer semantics for some functions as long as I don't let the data in
the remaining call chain switch back to value semantics. Once I switch
to pointer semantics, all future calls from that point need to stick to
pointer semantics.

I can never, ever, never, go from pointer to value. It's never safe to
make a copy of a value that a pointer points to.`
