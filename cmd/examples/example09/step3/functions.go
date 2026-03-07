package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/client"
)

// toolSuccessResponse returns a successful structured tool response.
func toolSuccessResponse(toolID string, keyValues ...any) client.D {
	data := make(map[string]any, len(keyValues)/2)
	for i := 0; i < len(keyValues); i += 2 {
		data[keyValues[i].(string)] = keyValues[i+1]
	}

	return toolResponse(toolID, data, "SUCCESS")
}

// toolErrorResponse returns a failed structured tool response.
func toolErrorResponse(toolID string, err error) client.D {
	data := map[string]any{"error": err.Error()}

	return toolResponse(toolID, data, "FAILED")
}

// toolResponse creates a structured tool response.
func toolResponse(toolID string, data map[string]any, status string) client.D {
	info := struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: status,
		Data:   data,
	}

	content, err := json.Marshal(info)
	if err != nil {
		return client.D{
			"role":         "tool",
			"tool_call_id": toolID,
			"content":      `{"status": "FAILED", "data": "error marshaling tool response"}`,
		}
	}

	return client.D{
		"role":         "tool",
		"tool_call_id": toolID,
		"content":      string(content),
	}
}

// =============================================================================
// GetWeather Tool

// GetWeather represents a tool that can be used to get weather information.
type GetWeather struct {
	name string
}

// RegisterGetWeather creates a new instance of the GetWeather tool and loads it
// into the provided tools map.
func RegisterGetWeather(tools map[string]Tool) client.D {
	gw := GetWeather{
		name: "tool_get_weather",
	}
	tools[gw.name] = &gw

	return gw.toolDocument()
}

// ToolDocument defines the metadata for the tool that is provided to the model.
func (gw *GetWeather) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        gw.name,
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
	}
}

// Call is the function that is called by the agent to get weather information when the model
// requests the tool with the specified parameters.
func (gw *GetWeather) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(toolCall.ID, fmt.Errorf("%s", r))
		}
	}()

	location := toolCall.Function.Arguments["location"].(string)

	return toolSuccessResponse(toolCall.ID,
		"temperature", 28,
		"humidity", 80,
		"wind_speed", 10,
		"description", "The weather in "+location+" is hot and humid",
	)
}