// This example shows you how to add tool calling to the chat agent from
// steps 1 and 2.
//
// # Running the example:
//
//	$ make example09-step3
//
// # This requires running the following commands:
//
//	$ make kronk-up  // This starts the Kronk service.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

var (
	url           = "http://localhost:8080/v1/chat/completions"
	model         = "gpt-oss-20b-Q8_0"
	contextWindow = 32 * 1024
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

// DECLARE A TOOL INTERFACE TO ALLOW THE AGENT TO CALL ANY TOOL FUNCTION
// WE DEFINE WITHOUT THE AGENT KNOWING THE EXACT TOOL IT IS USING.

type Tool interface {
	Call(ctx context.Context, toolCall client.ToolCall) client.D
}

// =============================================================================

type Agent struct {
	sseClient      *client.SSEClient[client.ChatSSE]
	getUserMessage func() (string, bool)

	// WE NEED TO ADD TOOL SUPPORT TO THE AGENT. WE NEED TO HAVE A SET OF
	// TOOLS THAT THE AGENT CAN USE TO PERFORM TASKS AND THE CORRESPONDING
	// DOCUMENTATION FOR THE MODEL.
	tools         map[string]Tool
	toolDocuments []client.D
}

func NewAgent(getUserMessage func() (string, bool)) (*Agent, error) {

	// CONSTRUCT THE TOOLS MAP HERE BECAUSE IT IS PASSED ON TOOL CONSTRUCTION
	// SO TOOLS CAN REGISTER THEMSELVES IN THIS MAP OF AVAILABLE TOOLS.
	tools := map[string]Tool{}

	agent := Agent{
		sseClient:      client.NewSSE[client.ChatSSE](client.StdoutLogger),
		getUserMessage: getUserMessage,

		// ADD THE TOOLNG SUPPORT TO THE AGENT.
		tools: tools,
		toolDocuments: []client.D{
			RegisterGetWeather(tools),
		},
	}

	return &agent, nil
}

// WE NEED TO EXTEND THE SYSTEM PROMPT TO INCLUDE THE TOOLING INSTRUCTIONS.

const systemPrompt = `
You are a helpful coding assistant that has tools to assist you in coding.

After you request a tool call, you will receive a JSON document with two fields,
"status" and "data". Always check the "status" field to know if the call "SUCCEED"
or "FAILED". The information you need to respond will be provided under the "data"
field. If the called "FAILED", just inform the user and don't try using the tool
again for the current response.

When reading Go source code always start counting lines of code from the top of
the source code file.

If you get back results from a tool call, do not verify the results.
`

func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D

	// WE WILL KEEP TRACK OF WHETHER WE ARE IN A TOOL CALL.
	var inToolCall bool

	conversation = append(conversation, client.D{
		"role":    "system",
		"content": systemPrompt,
	})

	fmt.Printf("Chat with %s (use 'ctrl-c' to quit)\n", model)

	for {
		// CHECK IF WE ARE IN A TOOL CALL BEFORE ASKING FOR INPUT.
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

		// WE NEED TO RESET THE TOOL CALL FLAG ON EACH ITERATION.
		inToolCall = false

		d := client.D{
			"model":       model,
			"messages":    conversation,
			"max_tokens":  contextWindow,
			"temperature": 0.1,
			"top_p":       0.1,
			"top_k":       50,
			"stream":      true,

			// ADDING TOOL CALLING TO THE REQUEST.
			"tools":          a.toolDocuments,
			"tool_selection": "auto",
		}

		fmt.Printf("\u001b[93m\n%s\u001b[0m: ", model)

		ch := make(chan client.ChatSSE, 100)
		ctx, cancelDoCall := context.WithTimeout(ctx, time.Minute*5)

		if err := a.sseClient.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			fmt.Printf("\n\n\u001b[91mERROR:%s\u001b[0m\n\n", err)
			cancelDoCall()
			continue
		}

		var chunks []string
		var pendingToolCalls []client.ToolCall
		reasonThinking := false

		fmt.Print("\n")

	loop:
		for resp := range ch {
			if len(resp.Choices) == 0 {
				continue
			}

			switch {

			// WE NEED TO CHECK IF WE ARE ASKING TO MAKE A TOOL CALL.
			case len(resp.Choices[0].Delta.ToolCalls) > 0:

				// STORE TOOL CALLS FOR PROCESSING AFTER THE LOOP SO WE DON'T
				// HOLD THE CONNECTION HOSTAGE.
				pendingToolCalls = resp.Choices[0].Delta.ToolCalls

				break loop

			case resp.Choices[0].Delta.Content != "":
				if reasonThinking {
					reasonThinking = false
					fmt.Print("\n")
				}

				fmt.Print(resp.Choices[0].Delta.Content)
				chunks = append(chunks, resp.Choices[0].Delta.Content)

			case resp.Choices[0].Delta.Reasoning != "":
				if !reasonThinking {
					fmt.Print("\n")
				}

				reasonThinking = true

				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
			}
		}

		cancelDoCall()

		// WE NEED TO EXECUTE THE TOOL CALLS AFTER THE LOOP SO WE DON'T HOLD
		// THE CONNECTION HOSTAGE.
		if len(pendingToolCalls) > 0 {

			// ADD THE TOOL CALL TO THE CONVERSATION SO THE MODEL HAS
			// CONTEXT OF THE TOOL CALL.
			argsJSON, _ := json.Marshal(pendingToolCalls[0].Function.Arguments)
			conversation = append(conversation, client.D{
				"role": "assistant",
				"tool_calls": []client.D{
					{
						"id":   pendingToolCalls[0].ID,
						"type": "function",
						"function": client.D{
							"name":      pendingToolCalls[0].Function.Name,
							"arguments": string(argsJSON),
						},
					},
				},
			})

			results := a.callTools(ctx, pendingToolCalls)

			if len(results) > 0 {
				conversation = append(conversation, results...)
				inToolCall = true
			}
		}

		// WE NEED TO CHECK IF WE ARE IN A TOOL CALL BECAUSE WE NEED TO GIVE THE
		// MODEL THE RESULTS WITHOUT ANY NOISE. THE CHUNKS SHOULD BE EMPTY IN
		// THIS CASE BUT I DON'T TRUST MODELS.

		if !inToolCall && len(chunks) > 0 {
			fmt.Print("\n")

			content := strings.Join(chunks, " ")
			content = strings.TrimLeft(content, "\n")

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

// WE NEED A FUNCTION THAT LOOKS UP THE REQUESTED TOOL BY NAME AND CALLS IT
// WITH THE MODEL PROVIDED PARAMETERS.

func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) []client.D {
	var resps []client.D

	for _, toolCall := range toolCalls {
		tool, exists := a.tools[toolCall.Function.Name]
		if !exists {
			continue
		}

		fmt.Printf("\n\n\u001b[92m%s(%v)\u001b[0m:\n\n", toolCall.Function.Name, toolCall.Function.Arguments)

		resp := tool.Call(ctx, toolCall)
		resps = append(resps, resp)

		fmt.Printf("%#v\n", resps)
	}

	return resps
}
