// Package tools provides tool protocol primitives for the agentic steps.
package tools

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
)

// PreviewMaxChars is the maximum number of characters to show in a tool
// response preview when printing call results for educational tracing.
const PreviewMaxChars = 150

// Tool describes the features which all tools must implement.
type Tool interface {
	Call(ctx context.Context, toolCall client.ToolCall) client.D
}

// Preview returns a short, single-line preview of the "content" field of a
// tool response, truncated to PreviewMaxChars with an ellipsis if cut.
// It is intended to give learners visibility into what a tool returned to
// the model without flooding the terminal.
func Preview(resp client.D) string {
	content, _ := resp["content"].(string)
	content = strings.Join(strings.Fields(content), " ")
	if len(content) > PreviewMaxChars {
		return content[:PreviewMaxChars] + "…"
	}
	return content
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
