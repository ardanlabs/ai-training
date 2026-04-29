// Package tools provides tool protocol primitives for the agentic steps.
package tools

import (
	"context"
	"encoding/json"

	"github.com/ardanlabs/ai-training/foundation/client"
)

// Tool describes the features which all tools must implement.
type Tool interface {
	Call(ctx context.Context, toolCall client.ToolCall) client.D
}

// SuccessResponse returns a successful structured tool response.
func SuccessResponse(toolID string, keyValues ...any) client.D {
	data := make(map[string]any, len(keyValues)/2)
	for i := 0; i < len(keyValues); i += 2 {
		data[keyValues[i].(string)] = keyValues[i+1]
	}

	return response(toolID, data, "SUCCESS")
}

// ErrorResponse returns a failed structured tool response.
func ErrorResponse(toolID string, err error) client.D {
	data := map[string]any{"error": err.Error()}

	return response(toolID, data, "FAILED")
}

// response creates a structured tool response.
func response(toolID string, data map[string]any, status string) client.D {
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
