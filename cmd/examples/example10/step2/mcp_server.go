package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/format"
	"go/parser"
	"go/token"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func init() {
	// -------------------------------------------------------------------------
	// Runs the MCP server locally for our example purposes. This could be
	// replaced with a MCP server that is running in a different process.

	go func() {
		mcpListenAndServe(mcpHost)
	}()
}

// mcpListenAndServe starts the MCP server for all the tooling we support.
func mcpListenAndServe(host string) {
	fmt.Printf("\nServer: MCP servers serving at %s\n", host)

	fileOperations := mcp.NewServer(&mcp.Implementation{Name: "file_operations", Version: "v1.0.0"}, nil)

	f := func(request *http.Request) *mcp.Server {
		url := request.URL.Path

		switch url {
		case RegisterReadFileTool(fileOperations),
			RegisterSearchFilesTool(fileOperations),
			RegisterCreateFileTool(fileOperations),
			RegisterGoCodeEditorTool(fileOperations):
			return fileOperations

		default:
			return mcp.NewServer(&mcp.Implementation{Name: "unknown_tool", Version: "v1.0.0"}, nil)
		}
	}

	handler := mcp.NewSSEHandler(f, &mcp.SSEOptions{})
	fmt.Println(http.ListenAndServe(host, handler))
}

// =============================================================================

// RegisterReadFileTool registers the read_file tool with the given MCP server.
func RegisterReadFileTool(mcpServer *mcp.Server) string {
	const toolName = "tool_read_file"
	const tooDescription = "Read the contents of a given file path or search for files containing a pattern. When searching file contents, returns line numbers where the pattern is found."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, ReadFileHandler)

	return "/" + toolName
}

// ReadFileToolParams represents the parameters for this tool call.
type ReadFileToolParams struct {
	Path string `json:"path" jsonschema:"a possible filter to use"`
}

// ReadFileHandler reads the contents of a given file path.
func ReadFileHandler(ctx context.Context, req *mcp.CallToolRequest, params ReadFileToolParams) (*mcp.CallToolResult, any, error) {
	dir := "."
	if params.Path != "" {
		dir = params.Path
	}

	content, err := os.ReadFile(dir)
	if err != nil {
		return nil, nil, err
	}

	info := struct {
		Contents string `json:"contents"`
	}{
		Contents: string(content),
	}

	data, err := json.Marshal(info)
	if err != nil {
		return nil, nil, err
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(data),
		}},
	}, nil, nil
}

// =============================================================================

// RegisterSearchFilesTool registers the search_files tool with the given MCP server.
func RegisterSearchFilesTool(mcpServer *mcp.Server) string {
	const toolName = "tool_search_files"
	const tooDescription = "Read the contents of a given file path or search for files containing a pattern. When searching file contents, returns line numbers where the pattern is found."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, SearchFilesHandler)

	return "/" + toolName
}

// SearchFilesToolParams represents the parameters for this tool call.
type SearchFilesToolParams struct {
	Path     string `json:"path" jsonschema:"Relative path to search files from. Defaults to current directory if not provided."`
	Filter   string `json:"filter" jsonschema:"The filter to apply to the file names. It supports golang regex syntax. If not provided, will filtering with take place. If provided, only return files that match the filter."`
	Contains string `json:"contains" jsonschema:"A string to search for inside files. It supports golang regex syntax. If not provided, no search will be performed. If provided, only return files that contain the string."`
}

// SearchFilesHandler searches for files in a given directory that match a
// given filter and contain a given string.
func SearchFilesHandler(ctx context.Context, req *mcp.CallToolRequest, params SearchFilesToolParams) (*mcp.CallToolResult, any, error) {
	dir := "."
	if params.Path != "" {
		dir = params.Path
	}

	filter := params.Filter
	contains := params.Contains

	var files []string
	err := filepath.WalkDir(dir, func(path string, info fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, filepath.SkipDir) {
				return nil
			}
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if strings.Contains(relPath, "zarf") ||
			strings.Contains(relPath, "vendor") ||
			strings.Contains(relPath, ".venv") ||
			strings.Contains(relPath, ".idea") ||
			strings.Contains(relPath, ".vscode") ||
			strings.Contains(relPath, "libw2v") ||
			strings.Contains(relPath, ".git") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if relPath == "." {
			return nil
		}

		if filter != "" {
			if matched, _ := regexp.MatchString(filter, relPath); !matched {
				return nil
			}
		}

		if contains != "" {
			content, err := os.ReadFile(relPath)
			if err != nil {
				return nil
			}

			if matched, _ := regexp.MatchString(contains, string(content)); !matched {
				return nil
			}
		}

		switch {
		case info.IsDir():
			files = append(files, relPath+"/")

		default:
			files = append(files, relPath)
		}

		return nil
	})

	if err != nil {
		return nil, nil, err
	}

	info := struct {
		Files []string `json:"files"`
	}{
		Files: files,
	}

	data, err := json.Marshal(info)
	if err != nil {
		return nil, nil, err
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(data),
		}},
	}, nil, nil
}

// =============================================================================

// RegisterCreateFileTool registers the search_files tool with the given MCP server.
func RegisterCreateFileTool(mcpServer *mcp.Server) string {
	const toolName = "tool_create_file"
	const tooDescription = "Creates a new file"

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, CreateFileHandler)

	return "/" + toolName
}

// CreateFileToolParams represents the parameters for this tool call.
type CreateFileToolParams struct {
	Path string `json:"path" jsonschema:"Relative path and name of the file to create."`
}

// CreateFileHandler creates a new file at the specified path.
func CreateFileHandler(ctx context.Context, req *mcp.CallToolRequest, params CreateFileToolParams) (*mcp.CallToolResult, any, error) {
	filePath := "."
	if params.Path != "" {
		filePath = params.Path
	}

	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		return nil, nil, err
	}

	dir := path.Dir(filePath)
	if dir != "." {
		os.MkdirAll(dir, 0755)
	}

	f, err := os.Create(filePath)
	if err != nil {
		return nil, nil, err
	}
	f.Close()

	info := struct {
		Status string `json:"status"`
	}{
		Status: "SUCCESS",
	}

	data, err := json.Marshal(info)
	if err != nil {
		return nil, nil, err
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(data),
		}},
	}, nil, nil
}

// =============================================================================

// RegisterGoCodeEditorTool registers the go_code_editor tool with the given MCP server.
func RegisterGoCodeEditorTool(mcpServer *mcp.Server) string {
	const toolName = "tool_go_code_editor"
	const tooDescription = "Edit Golang source code files including adding, replacing, and deleting lines."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, GoCodeEditorHandler)

	return "/" + toolName
}

// GoCodeEditorToolParams represents the parameters for this tool call.
type GoCodeEditorToolParams struct {
	Path       string `json:"path" jsonschema:"Relative path and name of the file to create."`
	LineNumber int    `json:"line_number" jsonschema:"Relative path and name of the Golang file"`
	TypeChange string `json:"type_change" jsonschema:"Type of change to make to the file."`
	LineChange string `json:"line_change" jsonschema:"Line of code to add, replace, or delete."`
}

// GoCodeEditorHandler can make add, updates, and deletes to go code.
func GoCodeEditorHandler(ctx context.Context, req *mcp.CallToolRequest, params GoCodeEditorToolParams) (*mcp.CallToolResult, any, error) {
	path := "."
	if params.Path != "" {
		path = params.Path
	}

	lineNumber := params.LineNumber
	typeChange := strings.TrimSpace(params.TypeChange)
	lineChange := strings.TrimSpace(params.LineChange)

	content, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}

	fset := token.NewFileSet()
	lines := strings.Split(string(content), "\n")

	if lineNumber < 1 || lineNumber > len(lines) {
		return nil, nil, fmt.Errorf("line number %d is out of range (1-%d)", lineNumber, len(lines))
	}

	switch typeChange {
	case "add":
		newLines := make([]string, 0, len(lines)+1)
		newLines = append(newLines, lines[:lineNumber-1]...)
		newLines = append(newLines, lineChange)
		newLines = append(newLines, lines[lineNumber-1:]...)
		lines = newLines

	case "replace":
		lines[lineNumber-1] = lineChange

	case "delete":
		if len(lines) == 1 {
			lines = []string{""}
		} else {
			lines = append(lines[:lineNumber-1], lines[lineNumber:]...)
		}

	default:
		return nil, nil, fmt.Errorf("unsupported change type: %s, please inform the user", typeChange)
	}

	modifiedContent := strings.Join(lines, "\n")

	_, err = parser.ParseFile(fset, path, modifiedContent, parser.ParseComments)
	if err != nil {
		return nil, nil, fmt.Errorf("syntax error after modification: %s, please inform the user", err)
	}

	formattedContent, err := format.Source([]byte(modifiedContent))
	if err != nil {
		formattedContent = []byte(modifiedContent)
	}

	err = os.WriteFile(path, formattedContent, 0644)
	if err != nil {
		return nil, nil, fmt.Errorf("write file: %s", err)
	}

	var action string
	switch typeChange {
	case "add":
		action = fmt.Sprintf("Added line at position %d", lineNumber)
	case "replace":
		action = fmt.Sprintf("Replaced line %d", lineNumber)
	case "delete":
		action = fmt.Sprintf("Deleted line %d", lineNumber)
	}

	info := struct {
		Message string `json:"message"`
	}{
		Message: action,
	}

	data, err := json.Marshal(info)
	if err != nil {
		return nil, nil, err
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(data),
		}},
	}, nil, nil
}
