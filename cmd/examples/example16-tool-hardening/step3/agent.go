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
	"github.com/jmoiron/sqlx"
)

// Agent represents the chat agent. Tool execution is routed through
// safeCallTool so panic recovery (step 02) and a per-tool timeout
// (step 03) can be layered in without changing call sites.
type Agent struct {
	sseClient      *client.SSEClient[client.ChatSSE]
	getUserMessage func() (string, bool)
	tools          map[string]fntools.Tool
	toolDocuments  []client.D
}

// NewAgent creates a new instance of Agent.
func NewAgent(getUserMessage func() (string, bool), chatLLM *client.LLM, embedLLM *client.LLM, db *sqlx.DB) (*Agent, error) {
	toolsMap := make(map[string]fntools.Tool)
	toolDocuments := []client.D{
		RegisterQueryDB(toolsMap, chatLLM, db),
		RegisterSearchBook(toolsMap, embedLLM, db),
		RegisterReadingProgress(toolsMap),
		RegisterCrash(toolsMap),
	}

	agent := Agent{
		sseClient:      client.NewSSE[client.ChatSSE](client.StdoutLogger),
		getUserMessage: getUserMessage,
		tools:          toolsMap,
		toolDocuments:  toolDocuments,
	}

	return &agent, nil
}

// Run starts the agent and runs the chat loop.
func (a *Agent) Run(ctx context.Context) error {
	conversation := []client.D{
		{"role": "system", "content": systemPrompt},
	}

	fmt.Printf("\nChat with GoNotebook AI [%s] (use 'ctrl-c' to quit)\n", llmModel)
	fmt.Println("Try: 'What does the Go Notebook say about interfaces?' or 'How many highlights do I have for concurrency?' or 'Where am I in the Ultimate Go Notebook?'")
	fmt.Println("To exercise recover() + timeout: 'Crash a tool to show me what happens.' (REPL stays up; FAILED envelope is returned within toolTimeout.)")

	needUserInput := true

	for {
		if needUserInput {
			if ok := a.promptUser(&conversation); !ok {
				return nil
			}
		}

		content, toolCalls, usage, err := a.streamModelTurn(ctx, conversation)
		if err != nil {
			return err
		}

		a.printUsage(usage)

		if len(toolCalls) > 0 {
			a.appendToolCalls(&conversation, toolCalls)

			results := a.callTools(ctx, toolCalls)
			if len(results) > 0 {
				conversation = append(conversation, results...)
			}

			needUserInput = false
			continue
		}

		a.appendAssistant(&conversation, content)
		needUserInput = true
	}
}

// =============================================================================

func (a *Agent) promptUser(conversation *[]client.D) bool {
	fmt.Print("\n==============================\n")
	fmt.Print("-----\n")
	fmt.Print("> Prompt: ")

	userInput, ok := a.getUserMessage()
	if !ok {
		return false
	}

	fmt.Print("-----\n")

	*conversation = append(*conversation, client.D{
		"role":    "user",
		"content": userInput,
	})

	return true
}

func (a *Agent) streamModelTurn(ctx context.Context, conversation []client.D) (string, []client.ToolCall, *client.Usage, error) {
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
		return "", nil, nil, fmt.Errorf("error streaming: %w", err)
	}

	stopPrinter := a.startLatencyPrinter(ctx)
	defer stopPrinter()

	var chunks []string
	var lastResp client.ChatSSE
	reasonStarted := false
	responseStarted := false

	for resp := range ch {
		lastResp = resp

		if len(resp.Choices) == 0 {
			continue
		}

		stopPrinter()

		switch resp.Choices[0].FinishReason {
		case "error":
			return "", nil, lastResp.Usage, fmt.Errorf("error from model: %s", resp.Choices[0].Delta.Content)

		case "stop":
			if responseStarted || reasonStarted {
				fmt.Print("\n-----\n")
			}
			text := strings.TrimLeft(strings.Join(chunks, ""), "\n")
			return text, nil, lastResp.Usage, nil

		case "tool_calls":
			if responseStarted || reasonStarted {
				fmt.Print("\n-----\n")
			}
			return "", resp.Choices[0].Delta.ToolCalls, lastResp.Usage, nil

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
	return text, nil, lastResp.Usage, nil
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

func (a *Agent) printUsage(usage *client.Usage) {
	if usage == nil {
		return
	}

	contextTokens := usage.PromptTokens + usage.CompletionTokens
	contextWindow := 32 * 1024
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("\n\n\u001b[90mInput: %d  Reasoning: %d  Completion: %d  Output: %d  Window: %d (%.0f%% of %.0fK) TPS: %.2f\u001b[0m",
		usage.PromptTokens, usage.ReasoningTokens, usage.CompletionTokens, usage.OutputTokens, contextTokens, percentage, of, usage.TokensPerSecond)
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

// callTools looks up requested tools by name and dispatches each one
// through safeCallTool, which centralizes panic recovery (step 02) and
// per-tool timeout (step 03).
func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) []client.D {
	resps := make([]client.D, 0, len(toolCalls))

	for _, toolCall := range toolCalls {
		tool, exists := a.tools[toolCall.Function.Name]
		if !exists {
			resps = append(resps, fntools.ErrorResponse(toolCall.ID, fmt.Errorf("unknown tool: %s", toolCall.Function.Name)))
			continue
		}

		fmt.Printf("\u001b[92m%s(%v)\u001b[0m:\n", toolCall.Function.Name, toolCall.Function.Arguments)

		resp := a.safeCallTool(ctx, tool, toolCall)
		fmt.Printf("\u001b[90m  → %s\u001b[0m\n", fntools.Preview(resp))
		resps = append(resps, resp)
	}

	return resps
}

// safeCallTool executes a tool call. Step 02 adds panic recovery so a
// panicking tool returns a structured FAILED envelope instead of
// crashing the agent, and step 03 wraps the call in context.WithTimeout
// so a slow or hung tool cannot wedge the loop.
func (a *Agent) safeCallTool(ctx context.Context, tool fntools.Tool, toolCall client.ToolCall) (resp client.D) {
	// Step 02
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("  \u001b[91mPANIC recovered: %v\u001b[0m\n", r)
			resp = fntools.ErrorResponse(toolCall.ID, fmt.Errorf("tool panicked: %v", r))
		}
	}()

	// Step 03
	ctx, cancel := context.WithTimeout(ctx, toolTimeout)
	defer cancel()

	return tool.Call(ctx, toolCall)
}

func (a *Agent) appendAssistant(conversation *[]client.D, content string) {
	if content == "" {
		return
	}

	fmt.Print("\n")
	*conversation = append(*conversation, client.D{"role": "assistant", "content": content})
}
