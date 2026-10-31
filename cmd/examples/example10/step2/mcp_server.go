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
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// mcpListenAndServe starts the MCP server for all the tooling we support. For
// our example purposes the server runs locally, inside this program. This could
// be replaced with a MCP server that is running in a different process.
//
// The listener is opened before returning, so the server is ready to accept
// connections as soon as this function returns without an error, and a port
// that is already in use is reported instead of the tool calls being sent to
// whatever else is listening there.
func mcpListenAndServe(host string) error {
	ln, err := net.Listen("tcp", host)
	if err != nil {
		return fmt.Errorf("mcp server: %w", err)
	}

	fmt.Printf("\nServer: MCP servers serving at %s\n", host)

	fileOperations := mcp.NewServer(&mcp.Implementation{Name: "file_operations", Version: "v1.0.0"}, nil)

	// Register the tools once, up front. Registering them inside the request
	// handler below would add them again on every request.
	readFile := RegisterReadFileTool(fileOperations)
	searchFiles := RegisterSearchFilesTool(fileOperations)
	createFile := RegisterCreateFileTool(fileOperations)
	goCodeEditor := RegisterGoCodeEditorTool(fileOperations)

	f := func(request *http.Request) *mcp.Server {
		url := request.URL.Path

		switch url {
		case readFile, searchFiles, createFile, goCodeEditor:
			return fileOperations

		default:
			return mcp.NewServer(&mcp.Implementation{Name: "unknown_tool", Version: "v1.0.0"}, nil)
		}
	}

	handler := mcp.NewSSEHandler(f, &mcp.SSEOptions{})

	go func() {
		fmt.Println(http.Serve(ln, handler))
	}()

	return nil
}

// =============================================================================

// RegisterReadFileTool registers the read_file tool with the given MCP server.
func RegisterReadFileTool(mcpServer *mcp.Server) string {
	const toolName = "tool_read_file"
	const tooDescription = "Read the contents of a given relative file path. Use this when you want to see what's inside a file."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, ReadFileHandler)

	return "/" + toolName
}

// ReadFileToolParams represents the parameters for this tool call.
type ReadFileToolParams struct {
	Path *string `json:"path,omitempty" jsonschema:"The relative path of a file in the working directory."`
}

// ReadFileHandler reads the contents of a given file path.
func ReadFileHandler(ctx context.Context, req *mcp.CallToolRequest, params ReadFileToolParams) (*mcp.CallToolResult, any, error) {
	if params.Path == nil || *params.Path == "" {
		return nil, nil, errors.New("the path argument is missing")
	}

	content, err := os.ReadFile(*params.Path)
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
	const tooDescription = "Search a directory at a given path for files that match a given file name or contain a given string. If no path is provided, search files will look in the current directory."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, SearchFilesHandler)

	return "/" + toolName
}

// SearchFilesToolParams represents the parameters for this tool call.
type SearchFilesToolParams struct {
	Path     *string `json:"path,omitempty" jsonschema:"Relative path to search files from. Defaults to current directory if not provided."`
	Filter   *string `json:"filter,omitempty" jsonschema:"The filter to apply to the file names. It supports golang regex syntax. If not provided, no filtering will take place. If provided, only return files that match the filter."`
	Contains *string `json:"contains,omitempty" jsonschema:"A string to search for inside files. It supports golang regex syntax. If not provided, no search will be performed. If provided, only return files that contain the string."`
}

// SearchFilesHandler searches for files in a given directory that match a
// given filter and contain a given string.
func SearchFilesHandler(ctx context.Context, req *mcp.CallToolRequest, params SearchFilesToolParams) (*mcp.CallToolResult, any, error) {
	dir := "."
	if params.Path != nil && *params.Path != "" {
		dir = *params.Path
	}

	var filter, contains string
	if params.Filter != nil {
		filter = *params.Filter
	}
	if params.Contains != nil {
		contains = *params.Contains
	}

	// Compile the expressions once, and tell the model when one is invalid
	// instead of silently matching nothing.

	var filterRE, containsRE *regexp.Regexp
	if filter != "" {
		re, err := regexp.Compile(filter)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid filter regex: %w", err)
		}
		filterRE = re
	}

	if contains != "" {
		re, err := regexp.Compile(contains)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid contains regex: %w", err)
		}
		containsRE = re
	}

	var files []string
	err := filepath.WalkDir(dir, func(path string, info fs.DirEntry, err error) error {
		if err != nil {
			// A bad starting directory is an error the model needs to see.
			// Anything below it that can't be read is skipped.
			if path == dir {
				return err
			}
			return nil
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

		if filterRE != nil && !filterRE.MatchString(relPath) {
			return nil
		}

		if containsRE != nil {
			// relPath is relative to dir, not to the working directory, so the
			// file has to be read using path.
			content, err := os.ReadFile(path)
			if err != nil {
				return nil
			}

			if !containsRE.Match(content) {
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

// RegisterCreateFileTool registers the create_file tool with the given MCP server.
func RegisterCreateFileTool(mcpServer *mcp.Server) string {
	const toolName = "tool_create_file"
	const tooDescription = "Creates a new file"

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, CreateFileHandler)

	return "/" + toolName
}

// CreateFileToolParams represents the parameters for this tool call.
type CreateFileToolParams struct {
	Path *string `json:"path,omitempty" jsonschema:"Relative path and name of the file to create."`
}

// CreateFileHandler creates a new file at the specified path.
func CreateFileHandler(ctx context.Context, req *mcp.CallToolRequest, params CreateFileToolParams) (*mcp.CallToolResult, any, error) {
	if params.Path == nil || *params.Path == "" {
		return nil, nil, errors.New("the path argument is missing")
	}
	filePath := *params.Path

	switch _, err := os.Stat(filePath); {
	case err == nil:
		return nil, nil, errors.New("file already exists")

	case !errors.Is(err, fs.ErrNotExist):
		return nil, nil, err
	}

	dir := filepath.Dir(filePath)
	if dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, nil, err
		}
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
	Path       *string `json:"path,omitempty" jsonschema:"Relative path and name of the Golang file"`
	LineNumber int     `json:"line_number" jsonschema:"The line number for the code change"`
	TypeChange *string `json:"type_change,omitempty" jsonschema:"The type of change to make: add, replace, delete"`
	LineChange *string `json:"line_change,omitempty" jsonschema:"The text to add, replace, delete"`
}

// GoCodeEditorHandler can make add, updates, and deletes to go code.
func GoCodeEditorHandler(ctx context.Context, req *mcp.CallToolRequest, params GoCodeEditorToolParams) (*mcp.CallToolResult, any, error) {
	if params.Path == nil || *params.Path == "" {
		return nil, nil, errors.New("the path argument is missing")
	}
	filePath := *params.Path

	lineNumber := params.LineNumber
	var typeChange, lineChange string
	if params.TypeChange != nil {
		typeChange = strings.ToLower(strings.TrimSpace(*params.TypeChange))
	}
	if params.LineChange != nil {
		lineChange = strings.TrimSpace(*params.LineChange)
	}

	content, err := os.ReadFile(filePath)
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

	_, err = parser.ParseFile(fset, filePath, modifiedContent, parser.ParseComments)
	if err != nil {
		return nil, nil, fmt.Errorf("syntax error after modification: %s, please inform the user", err)
	}

	formattedContent, err := format.Source([]byte(modifiedContent))
	if err != nil {
		formattedContent = []byte(modifiedContent)
	}

	err = os.WriteFile(filePath, formattedContent, 0644)
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
