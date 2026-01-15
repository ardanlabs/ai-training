package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// mcpClient is a client for the MCP server.
type mcpClient struct {
	client *mcp.Client
}

// newMCPClient constructs a new MCP client.
func newMCPClient() *mcpClient {
	client := mcp.NewClient(&mcp.Implementation{Name: "mcp-client", Version: "v1.0.0"}, nil)

	return &mcpClient{
		client: client,
	}
}

// Call executes an MCP tool call using the provided transport and parameters.
func (cln *mcpClient) Call(ctx context.Context, transport *mcp.SSEClientTransport, params *mcp.CallToolParams) ([]mcp.Content, error) {
	fmt.Print("\u001b[92mtool: connecting to MCP Server\u001b[0m\n")

	session, err := cln.client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MCP server: %w", err)
	}
	defer session.Close()

	fmt.Printf("\u001b[92mtool: calling tool: %s(%v)\u001b[0m\n\n", params.Name, params.Arguments)

	res, err := session.CallTool(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to call tool: %w", err)
	}

	if res.IsError {
		return nil, fmt.Errorf("tool call failed: %s", res.Content)
	}

	return res.Content, nil
}

// =============================================================================

// toolSuccessResponse returns a successful structured tool response.
func toolSuccessResponse(toolID string, keyValues ...any) client.D {
	data := make(map[string]any)
	for i := 0; i < len(keyValues); i = i + 2 {
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
// ReadFile Tool

// ReadFile represents a tool that can be used to read the contents of a file.
type ReadFile struct {
	name      string
	mcpClient *mcpClient
	transport *mcp.SSEClientTransport
}

// RegisterReadFile creates a new instance of the ReadFile tool and loads it
// into the provided tools map.
func RegisterReadFile(mcpClient *mcpClient, tools map[string]Tool) client.D {
	toolName := "tool_read_file"

	addr := fmt.Sprintf("http://%s/%s", mcpHost, toolName)
	transport := mcp.SSEClientTransport{
		Endpoint: addr,
	}

	rf := ReadFile{
		name:      toolName,
		mcpClient: mcpClient,
		transport: &transport,
	}
	tools[rf.name] = &rf

	return rf.toolDocument()
}

// ToolDocument defines the metadata for the tool that is provied to the model.
func (rf *ReadFile) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        rf.name,
			"description": "Read the contents of a given file path or search for files containing a pattern. When searching file contents, returns line numbers where the pattern is found.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "The relative path of a file in the working directory. If pattern is provided, this can be a directory path to search in.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

// Call is the function that is called by the agent to read the contents of a
// file when the model requests the tool with the specified parameters.
func (rf *ReadFile) Call(ctx context.Context, tool client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(tool.ID, fmt.Errorf("%s", r))
		}
	}()

	params := &mcp.CallToolParams{
		Name:      rf.name,
		Arguments: tool.Function.Arguments,
	}

	results, err := rf.mcpClient.Call(ctx, rf.transport, params)
	if err != nil {
		return toolErrorResponse(tool.ID, fmt.Errorf("failed to call tool: %w", err))
	}

	data := results[0].(*mcp.TextContent).Text

	var info struct {
		Contents string `json:"contents"`
	}

	if err := json.Unmarshal([]byte(data), &info); err != nil {
		return toolErrorResponse(tool.ID, fmt.Errorf("failed to unmarshal response: %w", err))
	}

	return toolSuccessResponse(tool.ID, "file_contents", info.Contents)
}

// =============================================================================
// SearchFiles Tool

// SearchFiles represents a tool that can be used to search for files.
type SearchFiles struct {
	name      string
	mcpClient *mcpClient
	transport *mcp.SSEClientTransport
}

// RegisterSearchFiles creates a new instance of the SearchFiles tool and loads it
// into the provided tools map.
func RegisterSearchFiles(mcpClient *mcpClient, tools map[string]Tool) client.D {
	toolName := "tool_search_files"

	addr := fmt.Sprintf("http://%s/%s", mcpHost, toolName)
	transport := mcp.SSEClientTransport{
		Endpoint: addr,
	}

	sf := SearchFiles{
		name:      toolName,
		mcpClient: mcpClient,
		transport: &transport,
	}
	tools[sf.name] = &sf

	return sf.toolDocument()
}

// toolDocument defines the metadata for the tool that is provied to the model.
func (sf *SearchFiles) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        sf.name,
			"description": "Search a directory at a given path for files that match a given file name or contain a given string. If no path is provided, search files will look in the current directory.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "Relative path to search files from. Defaults to current directory if not provided.",
					},
					"filter": client.D{
						"type":        "string",
						"description": "The filter to apply to the file names. It supports golang regex syntax. If not provided, will filtering with take place. If provided, only return files that match the filter.",
					},
					"contains": client.D{
						"type":        "string",
						"description": "A string to search for inside files. It supports golang regex syntax. If not provided, no search will be performed. If provided, only return files that contain the string.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

// Call is the function that is called by the agent to list files when the model
// requests the tool with the specified parameters.
func (sf *SearchFiles) Call(ctx context.Context, tool client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(tool.ID, fmt.Errorf("%s", r))
		}
	}()

	params := &mcp.CallToolParams{
		Name:      sf.name,
		Arguments: tool.Function.Arguments,
	}

	results, err := sf.mcpClient.Call(ctx, sf.transport, params)
	if err != nil {
		return toolErrorResponse(tool.ID, fmt.Errorf("failed to call tool: %w", err))
	}

	data := results[0].(*mcp.TextContent).Text

	var info struct {
		Files []string `json:"files"`
	}

	if err := json.Unmarshal([]byte(data), &info); err != nil {
		return toolErrorResponse(tool.ID, fmt.Errorf("failed to unmarshal response: %w", err))
	}

	return toolSuccessResponse(tool.ID, "files", info.Files)
}

// =============================================================================
// CreateFile Tool

// CreateFile represents a tool that can be used to search for files.
type CreateFile struct {
	name      string
	mcpClient *mcpClient
	transport *mcp.SSEClientTransport
}

// RegisterCreateFile creates a new instance of the CreateFile tool and loads it
// into the provided tools map.
func RegisterCreateFile(mcpClient *mcpClient, tools map[string]Tool) client.D {
	toolName := "tool_create_file"

	addr := fmt.Sprintf("http://%s/%s", mcpHost, toolName)
	transport := mcp.SSEClientTransport{
		Endpoint: addr,
	}

	cf := CreateFile{
		name:      toolName,
		mcpClient: mcpClient,
		transport: &transport,
	}
	tools[cf.name] = &cf

	return cf.toolDocument()
}

// toolDocument defines the metadata for the tool that is provied to the model.
func (cf *CreateFile) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        cf.name,
			"description": "Creates a new file",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "Relative path and name of the file to create.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

// Call is the function that is called by the agent to create a file when the model
// requests the tool with the specified parameters.
func (cf *CreateFile) Call(ctx context.Context, tool client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(tool.ID, fmt.Errorf("%s", r))
		}
	}()

	params := &mcp.CallToolParams{
		Name:      cf.name,
		Arguments: tool.Function.Arguments,
	}

	results, err := cf.mcpClient.Call(ctx, cf.transport, params)
	if err != nil {
		return toolErrorResponse(tool.ID, fmt.Errorf("failed to call tool: %w", err))
	}

	data := results[0].(*mcp.TextContent).Text

	var info struct {
		Status string `json:"status"`
	}

	if err := json.Unmarshal([]byte(data), &info); err != nil {
		return toolErrorResponse(tool.ID, fmt.Errorf("failed to unmarshal response: %w", err))
	}

	return toolSuccessResponse(tool.ID, "status", info.Status)
}

// =============================================================================
// GoCodeEditor Tool

// GoCodeEditor represents a tool that can be used to edit Go files.
type GoCodeEditor struct {
	name      string
	mcpClient *mcpClient
	transport *mcp.SSEClientTransport
}

// RegisterGoCodeEditor creates a new instance of the GoCodeEditor tool and loads it
// into the provided tools map.
func RegisterGoCodeEditor(mcpClient *mcpClient, tools map[string]Tool) client.D {
	toolName := "tool_go_code_editor"

	addr := fmt.Sprintf("http://%s/%s", mcpHost, toolName)
	transport := mcp.SSEClientTransport{
		Endpoint: addr,
	}

	gce := GoCodeEditor{
		name:      toolName,
		mcpClient: mcpClient,
		transport: &transport,
	}
	tools[gce.name] = &gce

	return gce.toolDocument()
}

// toolDocument defines the metadata for the tool that is provied to the model.
func (gce *GoCodeEditor) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        gce.name,
			"description": "Edit Golang source code files including adding, replacing, and deleting lines.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "Relative path and name of the Golang file",
					},
					"line_number": client.D{
						"type":        "integer",
						"description": "The line number for the code change",
					},
					"type_change": client.D{
						"type":        "string",
						"description": "The type of change to make: add, replace, delete",
					},
					"line_change": client.D{
						"type":        "string",
						"description": "The text to add, replace, delete",
					},
				},
				"required": []string{"path", "line_number", "type_change", "line_change"},
			},
		},
	}
}

// Call is the function that is called by the agent to edit a file when the model
// requests the tool with the specified parameters.
func (gce *GoCodeEditor) Call(ctx context.Context, tool client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(tool.ID, fmt.Errorf("%s", r))
		}
	}()

	params := &mcp.CallToolParams{
		Name:      gce.name,
		Arguments: tool.Function.Arguments,
	}

	results, err := gce.mcpClient.Call(ctx, gce.transport, params)
	if err != nil {
		return toolErrorResponse(tool.ID, fmt.Errorf("failed to call tool: %w", err))
	}

	data := results[0].(*mcp.TextContent).Text

	var info struct {
		Message string `json:"message"`
	}

	if err := json.Unmarshal([]byte(data), &info); err != nil {
		return toolErrorResponse(tool.ID, fmt.Errorf("failed to unmarshal response: %w", err))
	}

	return toolSuccessResponse(tool.ID, "message", info.Message)
}
