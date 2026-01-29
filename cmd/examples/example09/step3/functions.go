package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/client"
)

// WE WILL ADD SUPPORT FOR STRUCTURED TOOL RESPONSES.

func toolSuccessResponse(toolID string, keyValues ...any) client.D {
	data := make(map[string]any)
	for i := 0; i < len(keyValues); i = i + 2 {
		data[keyValues[i].(string)] = keyValues[i+1]
	}

	return toolResponse(toolID, data, "SUCCESS")
}

func toolErrorResponse(toolID string, err error) client.D {
	data := map[string]any{"error": err.Error()}

	return toolResponse(toolID, data, "FAILED")
}

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

// WE WILL DEFINE A TYPE FOR THE TOOL.

type GetWeather struct {
	name string
}

func RegisterGetWeather(tools map[string]Tool) client.D {
	gw := GetWeather{
		name: "tool_get_weather",
	}
	tools[gw.name] = &gw

	return gw.toolDocument()
}

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
		"description", fmt.Sprintln("The weather in", location, "is hot and humid"),
	)
}
