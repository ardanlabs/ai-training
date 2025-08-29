// This example shows you the workflow and mechanics for tool calling.
//
// # Running the example:
//
//	$ make example10-step3
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

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
	if err := weatherQuestion(context.TODO()); err != nil {
		return fmt.Errorf("weatherQuestion: %w", err)
	}

	return nil
}

func weatherQuestion(ctx context.Context) error {
	cln := client.NewSSE[client.ChatSSE](client.StdoutLogger)

	// -------------------------------------------------------------------------
	// Start by asking what the weather is like in New York City

	q := "What is the weather like in New York City?"

	fmt.Printf("\nQuestion:\n\n%s\n", q)

	conversation := []client.D{
		{
			"role":    "user",
			"content": q,
		},
	}

	toolName := "tool_get_weather"

	d := client.D{
		"model":       model,
		"messages":    conversation,
		"max_tokens":  contextWindow,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       1,
		"stream":      true,
		"tools": []client.D{
			{
				"type": "function",
				"function": client.D{
					"name":        toolName,
					"description": "Get the current weather for a location",
					"parameters": client.D{
						"type": "object",
						"properties": client.D{
							"location": client.D{
								"type":        "string",
								"description": "The location to get the weather for, e.g. San Francisco, CA",
							},
						},
						"required": []string{"location"},
					},
				},
			},
		},
		"tool_selection": "auto",
	}

	ch := make(chan client.ChatSSE, 100)
	if err := cln.Do(ctx, http.MethodPost, url, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	// -------------------------------------------------------------------------
	// The model will respond asking us to make the get_current_weather function
	// call. We will make the call and then send the response back to the model.

	fmt.Print("\n")

	for resp := range ch {
		switch {
		case len(resp.Choices[0].Delta.ToolCalls) > 0:
			toolCall := resp.Choices[0].Delta.ToolCalls[0]

			fmt.Printf("\n\n\u001b[92mModel Asking For Tool Call:\n\nToolID[%s]: %s(%s)\u001b[0m\n\n",
				toolCall.ID,
				toolCall.Function.Name,
				toolCall.Function.Arguments)

			conversation = append(conversation, client.D{
				"role": "assistant",
				"content": fmt.Sprintf("Tool call %s: %s(%v)",
					toolCall.ID,
					toolCall.Function.Name,
					toolCall.Function.Arguments),
			})

			resp := GetWeatherTool(ctx, toolName, toolCall)
			conversation = append(conversation, resp)

			fmt.Printf("%s\n\n", resp)

		case resp.Choices[0].Delta.Content != "":
			fmt.Print(resp.Choices[0].Delta.Content)

		case resp.Choices[0].Delta.Reasoning != "":
			fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
		}
	}

	// -------------------------------------------------------------------------
	// Send the result of the tool call back to the model

	d = client.D{
		"model":       model,
		"messages":    conversation,
		"max_tokens":  contextWindow,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       50,
		"stream":      true,
	}

	ch = make(chan client.ChatSSE, 100)
	if err := cln.Do(ctx, http.MethodPost, url, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	// -------------------------------------------------------------------------
	// The model should provide the answer based on the tool call

	fmt.Print("Final Result:\n\n")

	var reasoning bool

	for resp := range ch {
		switch {
		case resp.Choices[0].Delta.Content != "":
			if reasoning {
				fmt.Print("\n\n")
				reasoning = false
			}

			fmt.Print(resp.Choices[0].Delta.Content)

		case resp.Choices[0].Delta.Reasoning != "":
			reasoning = true
			fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
		}
	}

	return nil
}

// =============================================================================

// GetWeatherTool is the function that is called by the agent to get the weather
// when the model requests the tool with the specified parameters.
func GetWeatherTool(ctx context.Context, toolName string, toolCall client.ToolCall) (resp client.D) {

	// We are going to hardcode a result for now so we can test the tool.

	location := toolCall.Function.Arguments["location"].(string)

	info := struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: "SUCCESS",
		Data: map[string]any{
			"temperature": 28,
			"humidity":    80,
			"wind_speed":  10,
			"description": fmt.Sprintln("The weather in", location, "is hot and humid"),
		},
	}

	// Return the weather information as structured data using JSON which is
	// easier for the model to interpret.

	d, err := json.Marshal(info)
	if err != nil {
		return client.D{
			"role":         "tool",
			"tool_call_id": toolCall.ID,
			"tool_name":    toolName,
			"content":      fmt.Sprintf(`{"status": "FAILED", "data": "%s"}`, err),
		}
	}

	return client.D{
		"role":         "tool",
		"tool_call_id": toolCall.ID,
		"tool_name":    toolName,
		"content":      string(d),
	}
}
