// This example shows you how to create a terminal based chat agent
// using the Kronk service.
//
// # Running the example:
//
//	$ make example09-step1
//
// # This requires running the following commands:
//
//	$ make kronk-up  // This starts the Kronk service.
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
	"strings"
	"sync"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

var (
	url   = "http://localhost:8080/v1/chat/completions"
	model = "gpt-oss-20b-Q8_0"
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

// Agent represents the chat agent that can communicate with the model.
type Agent struct {
	sseClient      *client.SSEClient[client.ChatSSE]
	getUserMessage func() (string, bool)
	modelInfo      modelInfo
}

type modelInfo struct {
	id string
}

// NewAgent creates a new instance of Agent.
func NewAgent(getUserMessage func() (string, bool)) (*Agent, error) {
	agent := Agent{
		sseClient:      client.NewSSE[client.ChatSSE](client.StdoutLogger),
		getUserMessage: getUserMessage,
		modelInfo: modelInfo{
			id: model,
		},
	}

	return &agent, nil
}

// systemPrompt defines how the agent should behave when responding to user queries.
const systemPrompt = `You are a helpful assistant that provides accurate and concise information.

Reasoning: high
`

// Run starts the agent and runs the chat loop.
func (a *Agent) Run(ctx context.Context) error {
	conversation := []client.D{
		{"role": "system", "content": systemPrompt},
	}

	fmt.Printf("\nChat with %s (use 'ctrl-c' to quit)\n", a.modelInfo.id)

	for {
		// ---------------------------------------------------------------------
		// Prompt the user for their next question or request.

		if ok := a.promptUser(&conversation); !ok {
			return nil
		}

		// ---------------------------------------------------------------------
		// Make a streaming call to the model. This returns the assembled text
		// content from the response.

		content, usage, err := a.streamModelTurn(ctx, conversation)
		if err != nil {
			return err
		}

		a.printUsage(usage)

		// ---------------------------------------------------------------------
		// The model produced a text response. Add it to the conversation
		// and go back to asking the user for input.

		a.appendAssistant(&conversation, content)
	}
}

// promptUser asks the user for input and appends it to the conversation.
func (a *Agent) promptUser(conversation *[]client.D) bool {
	fmt.Print("\u001b[94m\nYou\u001b[0m: ")

	userInput, ok := a.getUserMessage()
	if !ok {
		return false
	}

	*conversation = append(*conversation, client.D{
		"role":    "user",
		"content": userInput,
	})

	return true
}

// streamModelTurn sends the conversation to the model and streams back the
// response. It returns the assembled text content and usage.
func (a *Agent) streamModelTurn(ctx context.Context, conversation []client.D) (string, *client.Usage, error) {
	d := client.D{
		"model":       model,
		"messages":    conversation,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       1,
		"stream":      true,
	}

	fmt.Printf("\u001b[93m\n%s\u001b[0m: 0.000", a.modelInfo.id)

	callCtx, cancelCall := context.WithTimeout(ctx, 5*time.Minute)
	defer cancelCall()

	ch := make(chan client.ChatSSE, 100)

	if err := a.sseClient.Do(callCtx, http.MethodPost, url, d, ch); err != nil {
		return "", nil, fmt.Errorf("error streaming: %w", err)
	}

	// Start the latency printer and ensure it stops.
	stopPrinter := a.startLatencyPrinter(ctx)
	defer stopPrinter()

	var chunks []string
	var lastResp client.ChatSSE
	reasonFirstChunk := true
	reasonThinking := false

	for resp := range ch {
		lastResp = resp

		if len(resp.Choices) == 0 {
			continue
		}

		// On the first real chunk, stop the latency printer.
		stopPrinter()

		switch resp.Choices[0].FinishReason {
		case "error":
			return "", lastResp.Usage, fmt.Errorf("error from model: %s", resp.Choices[0].Delta.Content)

		case "stop":
			text := strings.TrimLeft(strings.Join(chunks, " "), "\n")
			return text, lastResp.Usage, nil

		default:
			delta := resp.Choices[0].Delta

			switch {
			case delta.Reasoning != "":
				reasonThinking = true

				if reasonFirstChunk {
					reasonFirstChunk = false
					fmt.Print("\n")
				}

				fmt.Printf("\u001b[91m%s\u001b[0m", delta.Reasoning)

			case delta.Content != "":
				if reasonThinking {
					reasonThinking = false
					fmt.Print("\n\n")
				}

				fmt.Print(delta.Content)
				chunks = append(chunks, delta.Content)
			}
		}
	}

	// Stream ended without an explicit finish reason.
	text := strings.TrimLeft(strings.Join(chunks, " "), "\n")
	return text, lastResp.Usage, nil
}

// startLatencyPrinter starts a goroutine that displays elapsed time while
// waiting for the model's first response chunk. The returned function stops
// the printer; it is safe to call multiple times.
func (a *Agent) startLatencyPrinter(ctx context.Context) (stop func()) {
	modelID := a.modelInfo.id
	start := time.Now()

	ticker := time.NewTicker(100 * time.Millisecond)
	done := make(chan struct{})
	exited := make(chan struct{})

	var once sync.Once
	stop = func() {
		once.Do(func() {
			close(done)
			<-exited
		})
	}

	go func() {
		defer ticker.Stop()
		defer close(exited)

		for {
			select {
			case <-ticker.C:
				m := time.Since(start).Milliseconds()
				fmt.Printf("\r\u001b[93m%s %d.%03d\u001b[0m:  ", modelID, m/1000, m%1000)

			case <-done:
				fmt.Print("\n")
				return

			case <-ctx.Done():
				fmt.Print("\n")
				return
			}
		}
	}()

	return stop
}

// appendAssistant adds the assistant's text response to the conversation.
func (a *Agent) appendAssistant(conversation *[]client.D, content string) {
	if content == "" {
		return
	}

	fmt.Print("\n")
	*conversation = append(*conversation, client.D{"role": "assistant", "content": content})
}

// printUsage displays token usage information after each model call.
func (a *Agent) printUsage(usage *client.Usage) {
	if usage == nil {
		return
	}

	contextTokens := usage.PromptTokens + usage.CompletionTokens
	contextWindow := 32 * 1024 // TODO: Get this from model config when available
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("\n\n\u001b[90mInput: %d  Reasoning: %d  Completion: %d  Output: %d  Window: %d (%.0f%% of %.0fK) TPS: %.2f\u001b[0m",
		usage.PromptTokens, usage.ReasoningTokens, usage.CompletionTokens, usage.OutputTokens, contextTokens, percentage, of, usage.TokensPerSecond)
}