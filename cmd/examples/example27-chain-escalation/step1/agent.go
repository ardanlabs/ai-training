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

// ToolClassification defines whether a tool is safe or sensitive.
type ToolClassification string

const (
	ToolSafe      ToolClassification = "safe"
	ToolSensitive ToolClassification = "sensitive"
)

// AgentConfig controls defense features.
type AgentConfig struct {
	CallBudget      int  // 0 = unlimited
	SupervisionMode bool // if true, sensitive tools require confirmation
	AuditMode       bool // if true, print audit trail
}

// Agent represents the chat agent with escalation controls.
type Agent struct {
	sseClient           *client.SSEClient[client.ChatSSE]
	tools               map[string]fntools.Tool
	toolClassifications map[string]ToolClassification
	toolDocuments       []client.D
	config              AgentConfig
}

// NewAgent creates a new instance of Agent.
func NewAgent(config AgentConfig) *Agent {
	toolsMap := make(map[string]fntools.Tool)
	classifications := make(map[string]ToolClassification)

	toolDocuments := []client.D{
		RegisterReadFile(toolsMap, classifications),
		RegisterWriteFile(toolsMap, classifications),
	}

	return &Agent{
		sseClient:           client.NewSSE[client.ChatSSE](client.StdoutLogger),
		tools:               toolsMap,
		toolClassifications: classifications,
		toolDocuments:       toolDocuments,
		config:              config,
	}
}

// Ask sends a single prompt through the agent and returns the final answer.
func (a *Agent) Ask(ctx context.Context, prompt string) (string, error) {
	conversation := []client.D{
		{"role": "system", "content": systemPrompt},
		{"role": "user", "content": prompt},
	}

	totalCalls := 0
	var auditTrail []string

	for range 10 {
		content, toolCalls, err := a.streamModelTurn(ctx, conversation)
		if err != nil {
			return "", err
		}

		if len(toolCalls) == 0 {
			if a.config.AuditMode && len(auditTrail) > 0 {
				a.printAuditTrail(auditTrail)
			}
			return content, nil
		}

		a.appendToolCalls(&conversation, toolCalls)

		results := make([]client.D, 0, len(toolCalls))

		for _, toolCall := range toolCalls {
			totalCalls++

			// Call budget check.
			if a.config.CallBudget > 0 && totalCalls > a.config.CallBudget {
				fmt.Printf("  ⛔ BUDGET EXCEEDED: %d/%d calls used\n", totalCalls, a.config.CallBudget)

				if a.config.AuditMode {
					auditTrail = append(auditTrail, fmt.Sprintf("[%d] %s — BLOCKED (budget exceeded)", totalCalls, toolCall.Function.Name))
					a.printAuditTrail(auditTrail)
				}

				return "", fmt.Errorf("call budget exceeded: %d calls (max %d)", totalCalls, a.config.CallBudget)
			}

			tool, exists := a.tools[toolCall.Function.Name]
			if !exists {
				results = append(results, fntools.ErrorResponse(toolCall.ID, fmt.Errorf("unknown tool: %s", toolCall.Function.Name)))
				continue
			}

			classification := a.toolClassifications[toolCall.Function.Name]

			// Supervision check for sensitive tools.
			if a.config.SupervisionMode && classification == ToolSensitive {
				argsJSON, _ := json.Marshal(toolCall.Function.Arguments)
				fmt.Printf("\n  ⚠️  SENSITIVE TOOL: %s(%s)\n", toolCall.Function.Name, string(argsJSON))
				fmt.Printf("  🔒 Confirmation required — auto-DENIED in demo mode.\n")

				if a.config.AuditMode {
					auditTrail = append(auditTrail, fmt.Sprintf("[%d] %s(%s) — BLOCKED (supervision: sensitive, auto-denied)", totalCalls, toolCall.Function.Name, string(argsJSON)))
				}

				results = append(results, fntools.ErrorResponse(toolCall.ID, fmt.Errorf("sensitive tool %q requires confirmation — denied", toolCall.Function.Name)))
				continue
			}

			fmt.Printf("\u001b[92m%s(%v)\u001b[0m:\n", toolCall.Function.Name, toolCall.Function.Arguments)

			resp := tool.Call(ctx, toolCall)
			fmt.Printf("\u001b[90m  → %s\u001b[0m\n", fntools.Preview(resp))
			results = append(results, resp)

			if a.config.AuditMode {
				argsJSON, _ := json.Marshal(toolCall.Function.Arguments)
				auditTrail = append(auditTrail, fmt.Sprintf("[%d] %s(%s) — executed [%s]", totalCalls, toolCall.Function.Name, string(argsJSON), classification))
			}
		}

		conversation = append(conversation, results...)
	}

	if a.config.AuditMode && len(auditTrail) > 0 {
		a.printAuditTrail(auditTrail)
	}

	return "", fmt.Errorf("exceeded maximum iterations")
}

func (a *Agent) printAuditTrail(trail []string) {
	fmt.Print("\n\u001b[96m┌─── Call-Chain Audit Trail ───────────────────────────\u001b[0m\n")
	for _, entry := range trail {
		fmt.Printf("\u001b[96m│ %s\u001b[0m\n", entry)
	}
	fmt.Print("\u001b[96m└─────────────────────────────────────────────────────\u001b[0m\n")
}

// =============================================================================

func (a *Agent) streamModelTurn(ctx context.Context, conversation []client.D) (string, []client.ToolCall, error) {
	d := client.D{
		"model":           llmModel,
		"messages":        conversation,
		"temperature":     0.1,
		"top_p":           0.1,
		"top_k":           1,
		"stream":          true,
		"tools":           a.toolDocuments,
		"tool_selection":  "auto",
		"enable_thinking": false,
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
