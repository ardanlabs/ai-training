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

const toolTimeout = 30 * time.Second

// ModelConfig holds the parameters for a single model configuration used by
// the cascading router.
type ModelConfig struct {
	Name         string
	URL          string
	Model        string
	Temperature  float64
	TopP         float64
	TopK         int
	SystemPrompt string
}

// Agent represents the chat agent with cascading model routing.
type Agent struct {
	sseClient      *client.SSEClient[client.ChatSSE]
	getUserMessage func() (string, bool)
	tools          map[string]fntools.Tool
	toolDocuments  []client.D
	conversation   []client.D
	fastCfg        ModelConfig
	detailedCfg    ModelConfig
}

// NewAgent creates a new instance of Agent with cascading routing configs.
func NewAgent(getUserMessage func() (string, bool), chatLLM *client.LLM, embedLLM *client.LLM, db *sqlx.DB, fastCfg, detailedCfg ModelConfig) (*Agent, error) {
	toolsMap := make(map[string]fntools.Tool)
	toolDocuments := []client.D{
		RegisterQueryDB(toolsMap, chatLLM, db),
		RegisterSearchBook(toolsMap, embedLLM, db),
		RegisterReadingProgress(toolsMap),
	}

	agent := Agent{
		sseClient:      client.NewSSE[client.ChatSSE](client.StdoutLogger),
		getUserMessage: getUserMessage,
		tools:          toolsMap,
		toolDocuments:  toolDocuments,
		conversation: []client.D{
			{"role": "system", "content": systemPrompt},
		},
		fastCfg:     fastCfg,
		detailedCfg: detailedCfg,
	}

	return &agent, nil
}

// RunOnce processes a single user prompt through the full agent loop (including
// any tool-call rounds) and returns.
func (a *Agent) RunOnce(ctx context.Context) error {
	if ok := a.promptUser(); !ok {
		return nil
	}

	for {
		content, toolCalls, usage, usedCfg, err := a.streamModelTurn(ctx)
		if err != nil {
			return err
		}

		a.printUsage(usage)
		fmt.Printf("\n\u001b[96mHandled by: %s\u001b[0m\n", usedCfg)

		if len(toolCalls) > 0 {
			a.appendToolCalls(toolCalls)

			results := a.callTools(ctx, toolCalls)
			if len(results) > 0 {
				a.conversation = append(a.conversation, results...)
			}

			continue
		}

		a.appendAssistant(content)
		return nil
	}
}

// =============================================================================

func (a *Agent) promptUser() bool {
	userInput, ok := a.getUserMessage()
	if !ok {
		return false
	}

	fmt.Print("\n==============================\n")
	fmt.Print("-----\n")
	fmt.Printf("> Prompt: %s\n", userInput)
	fmt.Print("-----\n")

	a.conversation = append(a.conversation, client.D{
		"role":    "user",
		"content": userInput,
	})

	return true
}

// streamModelTurn performs the cascading router logic: fast pass first, then
// escalate if low confidence is detected.
func (a *Agent) streamModelTurn(ctx context.Context) (string, []client.ToolCall, *client.Usage, string, error) {

	// Fast pass.
	fastStart := time.Now()
	content, toolCalls, usage, err := a.doStreamTurn(ctx, a.fastCfg)
	if err != nil {
		return "", nil, nil, "", err
	}
	fastElapsed := time.Since(fastStart)

	// Tool calls are never subject to confidence checking.
	if len(toolCalls) > 0 {
		fmt.Printf("\n  \u001b[90m(fast pass, %s — tool call)\u001b[0m", fastElapsed)
		return content, toolCalls, usage, a.fastCfg.Name, nil
	}

	fmt.Printf("\n  \u001b[90m(fast pass, %s)\u001b[0m", fastElapsed)

	if !isLowConfidence(content) {
		return content, nil, usage, a.fastCfg.Name, nil
	}

	// Escalation — retry with the detailed config.
	fmt.Printf("\n  \u001b[93m→ Low confidence detected, escalating to %s...\u001b[0m\n", a.detailedCfg.Name)

	// Inject the detailed system prompt for this turn by replacing the
	// leading system message with the escalation persona, then nudge the
	// model with a separate user message.
	escalatedConv := make([]client.D, len(a.conversation))
	copy(escalatedConv, a.conversation)
	if a.detailedCfg.SystemPrompt != "" {
		if len(escalatedConv) > 0 && escalatedConv[0]["role"] == "system" {
			escalatedConv[0] = client.D{
				"role":    "system",
				"content": a.detailedCfg.SystemPrompt,
			}
		} else {
			escalatedConv = append([]client.D{{
				"role":    "system",
				"content": a.detailedCfg.SystemPrompt,
			}}, escalatedConv...)
		}
		escalatedConv = append(escalatedConv, client.D{
			"role":    "user",
			"content": "Please provide a comprehensive and detailed answer.",
		})
	}

	origConv := a.conversation
	a.conversation = escalatedConv

	escalateStart := time.Now()
	content2, toolCalls2, usage2, err := a.doStreamTurn(ctx, a.detailedCfg)
	a.conversation = origConv

	if err != nil {
		return "", nil, nil, "", err
	}

	escalateElapsed := time.Since(escalateStart)
	fmt.Printf("\n  \u001b[90m(escalated pass, %s)\u001b[0m", escalateElapsed)

	if len(toolCalls2) > 0 {
		return content2, toolCalls2, usage2, a.detailedCfg.Name, nil
	}

	return content2, nil, usage2, a.detailedCfg.Name, nil
}

// doStreamTurn performs a single streaming turn with the given model config.
func (a *Agent) doStreamTurn(ctx context.Context, cfg ModelConfig) (string, []client.ToolCall, *client.Usage, error) {
	d := client.D{
		"model":          cfg.Model,
		"messages":       a.conversation,
		"temperature":    cfg.Temperature,
		"top_p":          cfg.TopP,
		"top_k":          cfg.TopK,
		"stream":         true,
		"tools":          a.toolDocuments,
		"tool_selection": "auto",
	}

	fmt.Printf("\u001b[93m\n%s [%s]\u001b[0m: 0.000", cfg.Model, cfg.Name)

	callCtx, cancelCall := context.WithTimeout(ctx, 5*time.Minute)
	defer cancelCall()

	ch := make(chan client.ChatSSE, 100)

	if err := a.sseClient.Do(callCtx, http.MethodPost, cfg.URL, d, ch); err != nil {
		return "", nil, nil, fmt.Errorf("error streaming: %w", err)
	}

	stopPrinter := a.startLatencyPrinter(ctx, cfg)
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

func (a *Agent) startLatencyPrinter(ctx context.Context, cfg ModelConfig) (stop func()) {
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
				fmt.Printf("\r\u001b[93m%s [%s] %d.%03d\u001b[0m:  ", cfg.Model, cfg.Name, m/1000, m%1000)
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

func (a *Agent) appendToolCalls(toolCalls []client.ToolCall) {
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

	a.conversation = append(a.conversation, client.D{
		"role":       "assistant",
		"tool_calls": toolCallDocs,
	})
}

func (a *Agent) appendAssistant(content string) {
	if content == "" {
		return
	}

	fmt.Print("\n")
	a.conversation = append(a.conversation, client.D{"role": "assistant", "content": content})
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

// callTools looks up requested tools by name and executes them with
// panic recovery and a per-tool timeout.
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

// safeCallTool executes a tool call with panic recovery and a timeout.
func (a *Agent) safeCallTool(ctx context.Context, tool fntools.Tool, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("  \u001b[91mPANIC recovered: %v\u001b[0m\n", r)
			resp = fntools.ErrorResponse(toolCall.ID, fmt.Errorf("tool panicked: %v", r))
		}
	}()

	toolCtx, cancel := context.WithTimeout(ctx, toolTimeout)
	defer cancel()

	return tool.Call(toolCtx, toolCall)
}

// =============================================================================
// Confidence detection

const minConfidentLength = 30

var lowConfidenceSignals = []string{
	"i'm not sure",
	"i am not sure",
	"i don't know",
	"i do not know",
	"not certain",
	"hard to say",
	"it depends",
	"unclear",
	"out of scope",
}

func isLowConfidence(answer string) bool {
	cleaned := strings.TrimSpace(answer)

	if len(cleaned) < minConfidentLength {
		return true
	}

	lower := strings.ToLower(cleaned)
	for _, signal := range lowConfidenceSignals {
		if strings.Contains(lower, signal) {
			return true
		}
	}

	return false
}
