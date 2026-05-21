package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	fntools "github.com/ardanlabs/ai-training/foundation/tools"
	"github.com/jmoiron/sqlx"
)

// Agent represents the chat agent that can use tools to perform tasks.
type Agent struct {
	cln            *client.Client
	getUserMessage func() (string, bool)
	tools          map[string]fntools.Tool
	toolDocuments  []client.D
}

// NewAgent creates a new instance of Agent.
func NewAgent(getUserMessage func() (string, bool), embedLLM *client.LLM, db *sqlx.DB) (*Agent, error) {
	toolsMap := make(map[string]fntools.Tool)
	toolDocuments := []client.D{
		RegisterSearchBook(toolsMap, embedLLM, db),
		RegisterReadingProgress(toolsMap),
	}

	agent := Agent{
		cln:            client.New(client.StdoutLogger),
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
	fmt.Println("Try: 'How do interfaces work in Go?' or 'Where am I in the Ultimate Go Notebook?'")

	needUserInput := true

	for {
		if needUserInput {
			if ok := a.promptUser(&conversation); !ok {
				return nil
			}
		}

		content, toolCalls, err := a.modelTurn(ctx, conversation)
		if err != nil {
			return err
		}

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

// modelTurn sends the conversation to the model and returns either the
// final text content or any tool calls the model wants executed.
func (a *Agent) modelTurn(ctx context.Context, conversation []client.D) (string, []client.ToolCall, error) {
	d := client.D{
		"model":          llmModel,
		"messages":       conversation,
		"temperature":    0.1,
		"top_p":          0.1,
		"top_k":          1,
		"tools":          a.toolDocuments,
		"tool_selection": "auto",
	}

	callCtx, cancelCall := context.WithTimeout(ctx, 5*time.Minute)
	defer cancelCall()

	var chat agentChat
	if err := a.cln.Do(callCtx, http.MethodPost, llmURL, d, &chat); err != nil {
		return "", nil, fmt.Errorf("error calling model: %w", err)
	}

	if len(chat.Choices) == 0 {
		return "", nil, fmt.Errorf("no response from model")
	}

	choice := chat.Choices[0]

	if choice.FinishReason == "tool_calls" || len(choice.Message.ToolCalls) > 0 {
		return "", choice.Message.ToolCalls, nil
	}

	fmt.Printf("> Response: %s\n", choice.Message.Content)
	fmt.Print("-----\n")

	return choice.Message.Content, nil, nil
}

func (a *Agent) appendToolCalls(conversation *[]client.D, toolCalls []client.ToolCall) {
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

func (a *Agent) appendAssistant(conversation *[]client.D, content string) {
	if content == "" {
		return
	}

	*conversation = append(*conversation, client.D{"role": "assistant", "content": content})
}

// =============================================================================

type agentMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	ToolCalls []client.ToolCall `json:"tool_calls,omitempty"`
}

type agentChoice struct {
	Index        int          `json:"index"`
	Message      agentMessage `json:"message"`
	FinishReason string       `json:"finish_reason"`
}

type agentChat struct {
	ID      string        `json:"id"`
	Choices []agentChoice `json:"choices"`
}
