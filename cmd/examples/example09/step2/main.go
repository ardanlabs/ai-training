// This example shows you how to add a system prompt and better UI formatting
// from the chat agent in step 1.
//
// # Running the example:
//
//	$ make example09-step2
//
// # This requires running the following commands:
//
//	$ make kronk-up  // This starts the Kronk service.

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

var (
	url   = "http://localhost:8080/v1/chat/completions"
	model = "Qwen3-8B-Q8_0"
)

func init() {
	if v := os.Getenv("KRONK_SERVER"); v != "" {
		url = v
	}

	if v := os.Getenv("KRONK_MODEL"); v != "" {
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
	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	agent, err := NewAgent(getUserMessage)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	return agent.Run(context.TODO())
}

// =============================================================================

type Agent struct {
	sseClient      *client.SSEClient[client.ChatSSE]
	getUserMessage func() (string, bool)
}

func NewAgent(getUserMessage func() (string, bool)) (*Agent, error) {
	agent := Agent{
		sseClient:      client.NewSSE[client.ChatSSE](client.StdoutLogger),
		getUserMessage: getUserMessage,
	}

	return &agent, nil
}

// WE WILL ADD A SYSTEM PROMPT TO THE AGENT TO HELP WITH CLARIFYING INSTRUCTIONS.

const systemPrompt = `You are a helpful coding assistant that has tools to assist
you in coding.

Reasoning: high
`

func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D

	// WE WILL ADD THE SYSTEM PROMPT TO THE CONVERSATION.
	conversation = append(conversation, client.D{
		"role":    "system",
		"content": systemPrompt,
	})

	fmt.Printf("\nChat with %s (use 'ctrl-c' to quit)\n", model)

	for {
		fmt.Print("\u001b[94m\nYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		conversation = append(conversation, client.D{
			"role":    "user",
			"content": userInput,
		})

		d := client.D{
			"model":       model,
			"messages":    conversation,
			"temperature": 0.0,
			"top_p":       0.1,
			"top_k":       1,
			"stream":      true,
		}

		fmt.Printf("\u001b[93m\n%s\u001b[0m: ", model)

		ch := make(chan client.ChatSSE, 100)
		ctx, cancelContext := context.WithTimeout(ctx, time.Minute*5)

		if err := a.sseClient.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			cancelContext()
			fmt.Printf("\n\n\u001b[91mERROR:%s\u001b[0m\n\n", err)
			continue
		}

		var chunks []string

		// WE WILL CREATE FLAGS TO KNOW WHEN WE ARE PROCESSING REASONING CONTENT.

		reasonThinking := false  // GPT models provide a Reasoning field.
		contentThinking := false // Other reasoning models use <think> tags.

		// WE WILL ADD SOME IMPROVED FORMATTING.
		fmt.Print("\n")

		for resp := range ch {
			if len(resp.Choices) == 0 {
				continue
			}

			switch {
			case resp.Choices[0].Delta.Content != "":

				// WE NEED TO RESET THE REASONING FLAG ONCE THE MODEL IS
				// DONE REASONING.
				if reasonThinking {
					reasonThinking = false
					fmt.Print("\n\n")
				}

				// WE NEED TO CHECK IF THE REASONING IS HAPPENING VIA
				// <think> TAGS.
				switch resp.Choices[0].Delta.Content {
				case "<think>":
					contentThinking = true
					continue
				case "</think>":
					contentThinking = false
					continue
				}

				// WE NEED TO ADJUST OUR ORIGINAL SWITCH TO TAKE INTO ACCOUNT
				// WE MIGHT HAVE BEEN PROCESSING <think> TAGS.
				switch {
				case !contentThinking:
					fmt.Print(resp.Choices[0].Delta.Content)
					chunks = append(chunks, resp.Choices[0].Delta.Content)

				case contentThinking:
					fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Content)
				}

			// WE NEED TO CHECK IF THE MODEL IS THINKING VIA THIS REASONING
			// FIELD AND FORMAT THE RESPONSE PROPERLY.
			case resp.Choices[0].Delta.Reasoning != "":
				if !reasonThinking {
					fmt.Print("\n")
				}

				reasonThinking = true

				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
			}
		}

		cancelContext()

		if len(chunks) > 0 {
			fmt.Print("\n")

			// REMOVING <think> TAGS FROM THE CONTENT WILL LEAVE EXTRA CRLF
			// CHARACTERS WE NEED TO REMOVE.
			content := strings.Join(chunks, " ")
			content = strings.TrimLeft(content, "\n")

			// WE NEED TO CHECK IF THE CONTENT IS EMPTY AFTER REMOVING CRLF.
			if content != "" {
				conversation = append(conversation, client.D{
					"role":    "assistant",
					"content": strings.Join(chunks, " "),
				})
			}
		}
	}

	return nil
}
