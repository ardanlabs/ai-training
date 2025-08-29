// This example shows you how to create a terminal based chat agent.
// using the Ollama service.
//
// # Running the example:
//
//	$ make example10-step1
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
//
// # Extra reading and watching:
//
//  https://ampcode.com/how-to-build-an-agent

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

var (
	url   = "http://localhost:11434/v1/chat/completions"
	model = "gpt-oss:latest"

	// The context window represents the maximum number of tokens that can be sent
	// and received by the model. The default for Ollama is 4K. In the makefile
	// it has been increased to 64K.
	contextWindow = 1024 * 4
)

func init() {
	if v := os.Getenv("OLLAMA_CONTEXT_LENGTH"); v != "" {
		var err error
		contextWindow, err = strconv.Atoi(v)
		if err != nil {
			log.Fatal(err)
		}
	}

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

// Agent represents the chat agent that can use tools to perform tasks.
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

func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D

	fmt.Printf("Chat with %s (use 'ctrl-c' to quit)\n", model)

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
			"max_tokens":  contextWindow,
			"temperature": 0.1,
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

		for resp := range ch {
			if len(resp.Choices) == 0 {
				continue
			}

			switch {
			case resp.Choices[0].Delta.Content != "":
				fmt.Print(resp.Choices[0].Delta.Content)
				chunks = append(chunks, resp.Choices[0].Delta.Content)

			case resp.Choices[0].Delta.Reasoning != "":
				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
			}
		}

		cancelContext()

		if len(chunks) > 0 {
			fmt.Print("\n")

			conversation = append(conversation, client.D{
				"role":    "assistant",
				"content": strings.Join(chunks, " "),
			})
		}
	}

	return nil
}
