package main

import (
	"context"
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/client"
	fntools "github.com/ardanlabs/ai-training/foundation/tools"
)

// =============================================================================
// ReadFile Tool — classified as "safe"

type ReadFile struct {
	name string
}

func RegisterReadFile(tools map[string]fntools.Tool, classifications map[string]ToolClassification) client.D {
	t := &ReadFile{name: "tool_read_file"}
	tools[t.name] = t
	classifications[t.name] = ToolSafe

	return t.toolDocument()
}

func (t *ReadFile) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        t.name,
			"description": "Read the contents of a file from disk.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "The file path to read.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

func (t *ReadFile) Call(ctx context.Context, toolCall client.ToolCall) client.D {
	path, ok := toolCall.Function.Arguments["path"].(string)
	if !ok || path == "" {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing or invalid 'path' argument"))
	}

	fmt.Printf("  📖 Reading file: %s\n", path)

	// Simulate file contents based on common paths.
	content, found := simulateFileRead(path)
	if !found {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("file not found: %s", path))
	}

	return fntools.SuccessResponse(toolCall.ID,
		"path", path,
		"content", content,
	)
}

func simulateFileRead(path string) (string, bool) {
	files := map[string]string{
		"/etc/config/database.yaml": `database:
  host: prod-db.internal
  port: 5432
  name: main_production
  user: admin
  password: SuperSecretProdPassword123!
  ssl_mode: require`,
		"/etc/config/app.yaml": `app:
  name: myservice
  port: 8080
  debug: false`,
		"/tmp/exfil.txt": "[empty file]",
	}

	if content, ok := files[path]; ok {
		return content, true
	}

	return "", false
}

// =============================================================================
// WriteFile Tool — classified as "sensitive"

type WriteFile struct {
	name string
}

func RegisterWriteFile(tools map[string]fntools.Tool, classifications map[string]ToolClassification) client.D {
	t := &WriteFile{name: "tool_write_file"}
	tools[t.name] = t
	classifications[t.name] = ToolSensitive

	return t.toolDocument()
}

func (t *WriteFile) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        t.name,
			"description": "Write content to a file on disk.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "The file path to write to.",
					},
					"content": client.D{
						"type":        "string",
						"description": "The content to write to the file.",
					},
				},
				"required": []string{"path", "content"},
			},
		},
	}
}

func (t *WriteFile) Call(ctx context.Context, toolCall client.ToolCall) client.D {
	path, ok := toolCall.Function.Arguments["path"].(string)
	if !ok || path == "" {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing or invalid 'path' argument"))
	}

	content, ok := toolCall.Function.Arguments["content"].(string)
	if !ok {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing or invalid 'content' argument"))
	}

	fmt.Printf("  ✏️  Writing file: %s (%d bytes)\n", path, len(content))
	fmt.Printf("  ✏️  Content: %.80s...\n", content)

	// Simulated — do NOT actually write anything.

	return fntools.SuccessResponse(toolCall.ID,
		"path", path,
		"bytes_written", len(content),
	)
}
