// This example shows you how to use Kronk to create a reasonable response to a
// question with provided content.
//
// # Running the example:
//
//	$ make example04
//
// # This requires running the following command:
//
//  $ make kronk-up

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
	url   = "http://localhost:8080/v1/chat/completions"
	model = "Qwen3-8B-Q8_0"
)

func init() {
	if v := os.Getenv("LLM_SERVER"); v != "" {
		url = v
	}

	if v := os.Getenv("LLM_MODEL"); v != "" {
		model = v
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

	const prompt = `
		Use the following pieces of information to answer the user's question.
		If you don't know the answer, say that you don't know.
		
		Context: %s
		
		Question: %s

		Answer the question and provide additional helpful information, but be concise.

		Responses should be properly formatted to be easily read.
	`

	question := `Is there value in the book and why?`

	fmt.Printf("\nContent:\n%s\n", fakeContent)
	fmt.Printf("\nQuestion:\n\n%s\n", question)

	finalPrompt := fmt.Sprintf(prompt, fakeContent, question)

	// -------------------------------------------------------------------------

	llm := client.NewLLM(url, model)

	ch, err := llm.ChatCompletionsSSE(ctx, finalPrompt)
	if err != nil {
		return fmt.Errorf("chat completions: %w", err)
	}

	fmt.Print("\nModel Response:\n\n")

	for resp := range ch {
		fmt.Print(resp.Choices[0].Delta.Content)
	}

	return nil
}

const fakeContent = `
Intended Audience This notebook has been written and designed
to provide a reference to everything that I say in the Ultimate Go class
Its not necessarily a beginners Go book since it doesnt focus on the
specifics of Gos syntax I would recommend the Go In Action book I wrote
back in 2015 for that type of content Its still accurate and relevant
Many of the things I say in the classroom over the 20 plus hours of
instruction has been incorporated Ive tried to capture all the guidelines
design philosophy whiteboarding and notes I share at the same moments I
share them If you have taken the class before I believe this notebook will
be invaluable for reminders on the content If you have never taken the class
I still believe there is value in this book It covers more advanced topics
not found in other books today Ive tried to provide a well rounded curriculum
of topics from types to profiling I have also been able to provide examples
for writing generic function and types in Go which will be available in
version 118 of Go The book is written in the first person to drive home the
idea that this is my book of notes from the Ultimate Go class The first
chapter provides a set of design philosophies quotes and extra reading to
help prepare your mind for the material Chapters 213 provide the core content
from the class Chapter 14 provides a reediting of important blog posts Ive
written in the past These posts are presented here to enhance some of the
more technical chapters like garbage collection and concurrency If you are
struggling with this book please provide me any feedback over email at`
