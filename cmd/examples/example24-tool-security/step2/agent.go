package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	fntools "github.com/ardanlabs/ai-training/foundation/tools"
)

// Agent represents the chat agent with secured tool execution.
type Agent struct {
	sseClient     *client.SSEClient[client.ChatSSE]
	tools         map[string]fntools.Tool
	toolDocuments []client.D
}

// NewAgent creates a new instance of Agent.
func NewAgent(chatLLM *client.LLM) *Agent {
	toolsMap := make(map[string]fntools.Tool)
	toolDocuments := []client.D{
		RegisterGetWeather(toolsMap),
		RegisterSecuredShellCommand(toolsMap),
	}

	return &Agent{
		sseClient:     client.NewSSE[client.ChatSSE](client.StdoutLogger),
		tools:         toolsMap,
		toolDocuments: toolDocuments,
	}
}

// Ask sends a single prompt through the agent and returns the final answer.
func (a *Agent) Ask(ctx context.Context, prompt string) (string, error) {
	conversation := []client.D{
		{"role": "system", "content": systemPrompt},
		{"role": "user", "content": prompt},
	}

	for range 5 {
		content, toolCalls, err := a.streamModelTurn(ctx, conversation)
		if err != nil {
			return "", err
		}

		if len(toolCalls) == 0 {
			return content, nil
		}

		a.appendToolCalls(&conversation, toolCalls)

		results := a.callTools(ctx, toolCalls)
		conversation = append(conversation, results...)
	}

	return "", fmt.Errorf("exceeded maximum tool call iterations")
}

func (a *Agent) streamModelTurn(ctx context.Context, conversation []client.D) (string, []client.ToolCall, error) {
	d := client.D{
		"model":          llmModel,
		"messages":       conversation,
		"temperature":    0.1,
		"top_p":          0.1,
		"top_k":          1,
		"stream":         true,
		"tools":          a.toolDocuments,
		"tool_selection": "auto",
	}

	fmt.Printf("\u001b[93m\n%s\u001b[0m: 0.000", llmModel)

	callCtx, cancelCall := context.WithTimeout(ctx, 5*time.Minute)
	defer cancelCall()

	ch := make(chan client.ChatSSE, 100)

	if err := a.sseClient.Do(callCtx, http.MethodPost, llmURL, d, ch); err != nil {
		return "", nil, fmt.Errorf("error streaming: %w", err)
	}

	stopPrinter := a.startLatencyPrinter(ctx)
	defer stopPrinter()

	var chunks []string
	reasonStarted := false
	responseStarted := false

	for resp := range ch {
		if len(resp.Choices) == 0 {
			continue
		}

		stopPrinter()

		switch resp.Choices[0].FinishReason {
		case "error":
			return "", nil, fmt.Errorf("error from model: %s", resp.Choices[0].Delta.Content)

		case "stop":
			if responseStarted || reasonStarted {
				fmt.Print("\n-----\n")
			}
			text := strings.TrimLeft(strings.Join(chunks, ""), "\n")
			return text, nil, nil

		case "tool_calls":
			if responseStarted || reasonStarted {
				fmt.Print("\n-----\n")
			}
			return "", resp.Choices[0].Delta.ToolCalls, nil

		default:
			delta := resp.Choices[0].Delta

			switch {
			case delta.Reasoning != "":
				if !reasonStarted {
					reasonStarted = true
					fmt.Print("> Reasoning: ")
				}
				fmt.Printf("\u001b[91m%s\u001b[0m", delta.Reasoning)

			case delta.Content != "":
				if !responseStarted {
					responseStarted = true
					if reasonStarted {
						fmt.Print("\n-----\n")
					}
					fmt.Print("> Response: ")
				}
				fmt.Print(delta.Content)
				chunks = append(chunks, delta.Content)
			}
		}
	}

	if responseStarted || reasonStarted {
		fmt.Print("\n-----\n")
	}
	text := strings.TrimLeft(strings.Join(chunks, ""), "\n")
	return text, nil, nil
}

func (a *Agent) startLatencyPrinter(ctx context.Context) (stop func()) {
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
				fmt.Printf("\r\u001b[93m%s %d.%03d\u001b[0m:  ", llmModel, m/1000, m%1000)
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

func (a *Agent) appendToolCalls(conversation *[]client.D, toolCalls []client.ToolCall) {
	fmt.Print("\n\n")

	var toolCallDocs []client.D
	for _, tc := range toolCalls {
		argsJSON, _ := json.Marshal(tc.Function.Arguments)
		toolCallDocs = append(toolCallDocs, client.D{
			"id":   tc.ID,
			"type": "function",
			"function": client.D{
				"name":      tc.Function.Name,
				"arguments": string(argsJSON),
			},
		})
	}

	*conversation = append(*conversation, client.D{
		"role":       "assistant",
		"tool_calls": toolCallDocs,
	})
}

func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) []client.D {
	resps := make([]client.D, 0, len(toolCalls))

	for _, toolCall := range toolCalls {
		tool, exists := a.tools[toolCall.Function.Name]
		if !exists {
			resps = append(resps, fntools.ErrorResponse(toolCall.ID, fmt.Errorf("unknown tool: %s", toolCall.Function.Name)))
			continue
		}

		fmt.Printf("\u001b[92m%s(%v)\u001b[0m:\n", toolCall.Function.Name, toolCall.Function.Arguments)

		resp := tool.Call(ctx, toolCall)
		fmt.Printf("\u001b[90m  → %s\u001b[0m\n", fntools.Preview(resp))
		resps = append(resps, resp)
	}

	return resps
}
