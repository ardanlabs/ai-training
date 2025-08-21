package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/client"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

// toolSuccessResponse returns a successful structured tool response.
func toolSuccessResponse(toolID string, toolName string, keyValues ...any) client.D {
	data := make(map[string]any)
	for i := 0; i < len(keyValues); i = i + 2 {
		data[keyValues[i].(string)] = keyValues[i+1]
	}

	return toolResponse(toolID, toolName, data, "SUCCESS")
}

// toolErrorResponse returns a failed structured tool response.
func toolErrorResponse(toolID string, toolName string, err error) client.D {
	data := map[string]any{"error": err.Error()}

	return toolResponse(toolID, toolName, data, "FAILED")
}

// toolResponse creates a structured tool response.
func toolResponse(toolID string, toolName string, data map[string]any, status string) client.D {
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
			"tool_name":    toolName,
			"content":      `{"status": "FAILED", "data": "error marshaling tool response"}`,
		}
	}

	return client.D{
		"role":         "tool",
		"tool_call_id": toolID,
		"tool_name":    toolName,
		"content":      string(content),
	}
}

// =============================================================================
// DBSearch Tool

// DBSearch represents a tool that can be used to search the database
// for a given document.
type DBSearch struct {
	name     string
	llmEmbed *client.LLM
	col      *mongo.Collection
}

// RegisterDBSearch creates a new instance of the DBSearch tool and loads it
// into the provided tools map.
func RegisterDBSearch(tools map[string]Tool, llmEmbed *client.LLM, col *mongo.Collection) client.D {
	dbs := DBSearch{
		name:     "tool_database_search",
		llmEmbed: llmEmbed,
		col:      col,
	}
	tools[dbs.name] = &dbs

	return dbs.toolDocument()
}

// ToolDocument defines the metadata for the tool that is provided to the model.
func (dbs *DBSearch) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        dbs.name,
			"description": "Searches in the database for the given content.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"content": client.D{
						"type":        "string",
						"description": "The content to look for in the database.",
					},
				},
				"required": []string{"content"},
			},
		},
	}
}

// Call is the function that is called by the agent to read the contents of a
// file when the model requests the tool with the specified parameters.
func (dbs *DBSearch) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(toolCall.ID, dbs.name, fmt.Errorf("%s", r))
		}
	}()

	content := ""
	if toolCall.Function.Arguments["content"] != "" {
		content = toolCall.Function.Arguments["content"].(string)
	}

	if content == "" {
		return toolErrorResponse(toolCall.ID, dbs.name, fmt.Errorf("content is required"))
	}

	searchResults, err := vectorSearch(ctx, dbs.llmEmbed, dbs.col, content)
	if err != nil {
		return toolErrorResponse(toolCall.ID, dbs.name, fmt.Errorf("vectorSearch returned and error: %w", err))
	}

	returnValue := map[string]string{}

	for _, result := range searchResults {
		returnValue[result.FileName] = result.Description
	}

	return toolSuccessResponse(toolCall.ID, dbs.name, "return_values", returnValue)
}

func vectorSearch(ctx context.Context, llm *client.LLM, col *mongo.Collection, question string) ([]searchResult, error) {
	vector, err := llm.EmbedText(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("embed text: %w", err)
	}

	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":       "vector_index",
				"exact":       true,
				"path":        "embedding",
				"queryVector": vector,
				"limit":       5,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"file_name":   1,
				"description": 1,
				"embedding":   1,
				"score": bson.M{
					"$meta": "vectorSearchScore",
				},
			}},
		},
	}

	cur, err := col.Aggregate(ctx, pipeline)
	if err != nil {
		return nil, fmt.Errorf("aggregate: %w", err)
	}
	defer cur.Close(ctx)

	var results []searchResult
	if err := cur.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("all: %w", err)
	}

	return results, nil
}
