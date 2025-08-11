// https://ampcode.com/how-to-build-an-agent
//
// This example shows you a final example of the coding agent with support
// to read, list, and edit files.
//
// # Running the example:
//
//	$ make example10-step4
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/tiktoken"
)

const (
	url   = "http://localhost:11434/v1/chat/completions"
	model = "gpt-oss:latest"
)

// The context window represents the maximum number of tokens that can be sent
// and received by the model. The default for Ollama is 8K. In the makefile
// it has been increased to 64K.
var contextWindow = 1024 * 8

func init() {
	if v := os.Getenv("OLLAMA_CONTEXT_LENGTH"); v != "" {
		var err error
		contextWindow, err = strconv.Atoi(v)
		if err != nil {
			log.Fatal(err)
		}
	}
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	// -------------------------------------------------------------------------
	// Declare a function that can accept user input which the agent will use
	// when it's the users turn.

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	// -------------------------------------------------------------------------
	// Construct the agent and get it started.

	agent, err := NewAgent(getUserMessage)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	return agent.Run(context.TODO())
}

// =============================================================================

// Tool describes the features which all tools must implement.
type Tool interface {
	Call(ctx context.Context, toolCall client.ToolCall) client.D
}

// =============================================================================

// Agent represents the chat agent that can use tools to perform tasks.
type Agent struct {
	sseClient      *client.SSEClient[client.ChatSSE]
	getUserMessage func() (string, bool)
	tke            *tiktoken.Tiktoken
	tools          map[string]Tool
	toolDocuments  []client.D
}

// NewAgent creates a new instance of Agent.
func NewAgent(getUserMessage func() (string, bool)) (*Agent, error) {

	// -------------------------------------------------------------------------
	// Construct the SSE client to make model calls.

	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		log.Println(s)
	}

	sseClient := client.NewSSE[client.ChatSSE](logger)

	// -------------------------------------------------------------------------
	// Construct the tokenizer.

	tke, err := tiktoken.NewTiktoken()
	if err != nil {
		return nil, fmt.Errorf("failed to create tiktoken: %w", err)
	}

	// -------------------------------------------------------------------------
	// Construct the agent.

	tools := map[string]Tool{}

	agent := Agent{
		sseClient:      sseClient,
		getUserMessage: getUserMessage,
		tke:            tke,
		tools:          tools,
		toolDocuments: []client.D{
			NewReadFile(tools),
			NewSearchFiles(tools),
			NewCreateFile(tools),
			NewGoCodeEditor(tools),
		},
	}

	return &agent, nil
}

// The system prompt for the model so it behaves as expected.
var systemPrompt = `You are a helpful coding assistant that has tools to assist
you in coding.

After you request a tool call, you will receive a JSON document with two fields,
"status" and "data". Always check the "status" field to know if the call "SUCCEED"
or "FAILED". The information you need to respond will be provided under the "data"
field. If the called "FAILED", just inform the user and don't try using the tool
again for the current response.

When reading Go source code always start counting lines of code from the top of
the source code file.

If you get back results from a tool call, do not verify the results.

Reasoning: high
`

// Run starts the agent and runs the chat loop.
func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D        // History of the conversation
	var reasonContent []string         // Reasoning content per model call
	var inToolCall bool                // Need to know we are inside a tool call request
	var lastToolCall []client.ToolCall // Last tool call to identify call dups

	conversation = append(conversation, client.D{
		"role":    "system",
		"content": systemPrompt,
	})

	fmt.Printf("Chat with %s (use 'ctrl-c' to quit)\n", model)

	for {
		// ---------------------------------------------------------------------
		// If we are not in a tool call then we can ask the user
		// to provide their next question or request.

		if !inToolCall {
			fmt.Print("\u001b[94m\nYou\u001b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}

			conversation = append(conversation, client.D{
				"role":    "user",
				"content": userInput,
			})
		}

		inToolCall = false

		// ---------------------------------------------------------------------
		// Now we will make a call to the model, we could be responding to a
		// tool call or providing a user request.

		d := client.D{
			"model":          model,
			"messages":       conversation,
			"max_tokens":     contextWindow,
			"temperature":    0.0,
			"top_p":          0.1,
			"top_k":          1,
			"stream":         true,
			"tools":          a.toolDocuments,
			"tool_selection": "auto",
		}

		fmt.Printf("\u001b[93m\n%s\u001b[0m: ", model)

		ch := make(chan client.ChatSSE, 100)
		ctx, cancelContext := context.WithTimeout(ctx, time.Minute*5)

		if err := a.sseClient.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			cancelContext()
			fmt.Printf("\n\n\u001b[91mERROR:%s\u001b[0m\n\n", err)
			inToolCall = false
			lastToolCall = nil
			continue
		}

		// ---------------------------------------------------------------------
		// Now we will make a call to the model.

		var chunks []string      // Store the response chunks since we are streaming.
		reasonThinking := false  // GPT models provide a Reasoning field.
		contentThinking := false // Other reasoning models use <think> tags.
		reasonContent = nil      // Reset the reasoning content for this next call.

		fmt.Print("\n")

		// ---------------------------------------------------------------------
		// Process the response which comes in as chunks. So we need to process
		// and save each chunk.

		for resp := range ch {
			switch {

			// Did the model ask us to execute a tool call?
			case len(resp.Choices[0].Delta.ToolCalls) > 0:
				fmt.Print("\n\n")

				conversation = a.addToConversation(reasonContent, conversation, client.D{
					"role":    "assistant",
					"content": fmt.Sprintf("Tool call %s: %s(%v)", resp.Choices[0].Delta.ToolCalls[0].ID, resp.Choices[0].Delta.ToolCalls[0].Function.Name, resp.Choices[0].Delta.ToolCalls[0].Function.Arguments),
				})

				toolID := resp.Choices[0].Delta.ToolCalls[0].ID

				result := compareToolCalls(toolID, lastToolCall, resp.Choices[0].Delta.ToolCalls)
				if len(result) > 0 {
					conversation = a.addToConversation(reasonContent, conversation, result)
					inToolCall = true
					continue
				}

				results := a.callTools(ctx, resp.Choices[0].Delta.ToolCalls)
				if len(results) > 0 {
					conversation = a.addToConversation(reasonContent, conversation, results...)
					inToolCall = true
					lastToolCall = resp.Choices[0].Delta.ToolCalls
				}

			// Did we get content? With some models a <think> tag could exist to
			// indicate reasoning. We need to filter that out and display it as
			// a different color.
			case resp.Choices[0].Delta.Content != "":
				if reasonThinking {
					reasonThinking = false
					fmt.Print("\n\n")
				}

				switch resp.Choices[0].Delta.Content {
				case "<think>":
					contentThinking = true
					continue
				case "</think>":
					contentThinking = false
					continue
				}

				switch {
				case !contentThinking:
					fmt.Print(resp.Choices[0].Delta.Content)
					chunks = append(chunks, resp.Choices[0].Delta.Content)

				case contentThinking:
					reasonContent = append(reasonContent, resp.Choices[0].Delta.Content)
					fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Content)
				}

				lastToolCall = nil

			// Did we get reasoning content? ChatGPT models provide reasoning in
			// the Delta.Reasoning field. Display it as a different color.
			case resp.Choices[0].Delta.Reasoning != "":
				reasonThinking = true

				if len(reasonContent) == 0 {
					fmt.Print("\n")
				}

				reasonContent = append(reasonContent, resp.Choices[0].Delta.Reasoning)
				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
			}
		}

		cancelContext()

		// ---------------------------------------------------------------------
		// We processed all the chunks from the response so we need to add
		// this to the conversation history.

		if !inToolCall && len(chunks) > 0 {
			fmt.Print("\n")

			content := strings.Join(chunks, " ")
			content = strings.TrimLeft(content, "\n")

			if content != "" {
				conversation = a.addToConversation(reasonContent, conversation, client.D{
					"role":    "assistant",
					"content": content,
				})
			}
		}
	}

	return nil
}

// callTools will lookup a requested tool by name and call it.
func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) []client.D {
	var resps []client.D

	for _, toolCall := range toolCalls {
		tool, exists := a.tools[toolCall.Function.Name]
		if !exists {
			continue
		}

		fmt.Printf("\u001b[92mtool: %s(%v)\u001b[0m:\n", toolCall.Function.Name, toolCall.Function.Arguments)

		resp := tool.Call(ctx, toolCall)
		resps = append(resps, resp)

		fmt.Printf("%#v\n", resps)
	}

	return resps
}

// addToConversation will add new messages to the conversation history and
// calculate the different tokens used in the conversation and display it to the
// user. It will also check the amount of input tokens currently in history
// and remove the oldest messages if we are over.
func (a *Agent) addToConversation(reasoning []string, conversation []client.D, newMessages ...client.D) []client.D {
	conversation = append(conversation, newMessages...)

	fmt.Print("\n")

	for {
		var currentWindow int
		for _, c := range conversation {
			currentWindow += a.tke.TokenCount(c["content"].(string))
		}

		r := strings.Join(reasoning, "")
		reasonTokens := a.tke.TokenCount(r)

		totalTokens := currentWindow + reasonTokens
		percentage := (float64(currentWindow) / float64(contextWindow)) * 100
		of := float32(contextWindow) / float32(1024)

		fmt.Printf("\u001b[90mTokens Total[%d] Reason[%d] Window[%d] (%.0f%% of %.0fK)\u001b[0m\n", totalTokens, reasonTokens, currentWindow, percentage, of)

		// ---------------------------------------------------------------------
		// Check if we have too many input tokens and start removing messages.

		if currentWindow > contextWindow {
			fmt.Print("\u001b[90mRemoving conversation history\u001b[0m\n")
			conversation = slices.Delete(conversation, 1, 2)
			continue
		}

		break
	}

	return conversation
}

// =============================================================================

// compareToolCalls will try and detect if the model is asking us to call the
// same tool twice. This function is not accurate because the arguments are in a
// map. We need to fix that.
func compareToolCalls(toolID string, last []client.ToolCall, current []client.ToolCall) client.D {
	if len(last) != len(current) {
		return client.D{}
	}

	for i := range last {
		if last[i].Function.Name != current[i].Function.Name {
			return client.D{}
		}

		if fmt.Sprintf("%v", last[i].Function.Arguments) != fmt.Sprintf("%v", current[i].Function.Arguments) {
			return client.D{}
		}
	}

	fmt.Printf("\u001b[92mtool\u001b[0m: %s(%v)\n", current[0].Function.Name, current[0].Function.Arguments)
	fmt.Printf("\u001b[92mtool\u001b[0m: %s\n", "Same tool call")

	return toolErrorResponse(toolID, current[0].Function.Name, errors.New("data already provided in a previous response, please review the conversation history"))
}

// toolSuccessResponse returns a successful structured tool response.
func toolSuccessResponse(toolID string, toolName string, values ...any) client.D {
	data := make(map[string]any)
	for i := 0; i < len(values); i = i + 2 {
		data[values[i].(string)] = values[i+1]
	}

	info := struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: "SUCCESS",
		Data:   data,
	}

	json, err := json.Marshal(info)
	if err != nil {
		return client.D{
			"role":         "tool",
			"tool_call_id": toolID,
			"tool_name":    toolName,
			"content":      `{"status": "FAILED", "data": "error marshaling tool response"}`,
		}
	}

	return client.D{
		"role":         "tool",
		"tool_call_id": toolID,
		"tool_name":    toolName,
		"content":      string(json),
	}
}

// toolErrorResponse returns a failed structured tool response.
func toolErrorResponse(toolID string, toolName string, err error) client.D {
	data := map[string]any{"error": err.Error()}

	info := struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: "FAILED",
		Data:   data,
	}

	json, err := json.Marshal(info)
	if err != nil {
		return client.D{
			"role":         "tool",
			"tool_call_id": toolID,
			"tool_name":    toolName,
			"content":      `{"status": "FAILED", "data": "error marshaling tool response"}`,
		}
	}

	content := string(json)

	return client.D{
		"role":         "tool",
		"tool_call_id": toolID,
		"tool_name":    toolName,
		"content":      content,
	}
}
