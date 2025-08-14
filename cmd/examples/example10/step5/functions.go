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
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
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
// ReadFile Tool

// ReadFile represents a tool that can be used to read the contents of a file.
type ReadFile struct {
	name string
}

// RegisterReadFile creates a new instance of the ReadFile tool and loads it
// into the provided tools map.
func RegisterReadFile(tools map[string]Tool) client.D {
	rf := ReadFile{
		name: "tool_read_file",
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
func (rf *ReadFile) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(toolCall.ID, rf.name, fmt.Errorf("%s", r))
		}
	}()

	dir := "."
	if toolCall.Function.Arguments["path"] != "" {
		dir = toolCall.Function.Arguments["path"].(string)
	}

	content, err := os.ReadFile(dir)
	if err != nil {
		return toolErrorResponse(toolCall.ID, rf.name, err)
	}

	return toolSuccessResponse(toolCall.ID, rf.name, "file_contents", string(content))
}

// =============================================================================
// SearchFiles Tool

// SearchFiles represents a tool that can be used to list files.
type SearchFiles struct {
	name string
}

// RegisterSearchFiles creates a new instance of the SearchFiles tool and loads it
// into the provided tools map.
func RegisterSearchFiles(tools map[string]Tool) client.D {
	sf := SearchFiles{
		name: "tool_search_files",
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
func (sf *SearchFiles) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(toolCall.ID, sf.name, fmt.Errorf("%s", r))
		}
	}()

	dir := "."
	if v, exists := toolCall.Function.Arguments["path"]; exists && v != "" {
		dir = v.(string)
	}

	filter := ""
	if v, exists := toolCall.Function.Arguments["filter"]; exists {
		filter = v.(string)
	}

	contains := ""
	if v, exists := toolCall.Function.Arguments["contains"]; exists {
		contains = v.(string)
	}

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
		return toolErrorResponse(toolCall.ID, sf.name, err)
	}

	return toolSuccessResponse(toolCall.ID, sf.name, "files", files)
}

// =============================================================================
// CreateFile Tool

// CreateFile represents a tool that can be used to create files.
type CreateFile struct {
	name string
}

// RegisterCreateFile creates a new instance of the CreateFile tool and loads it
// into the provided tools map.
func RegisterCreateFile(tools map[string]Tool) client.D {
	cf := CreateFile{
		name: "tool_create_file",
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
func (cf *CreateFile) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(toolCall.ID, cf.name, fmt.Errorf("%s", r))
		}
	}()

	filePath := toolCall.Function.Arguments["path"].(string)

	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		return toolErrorResponse(toolCall.ID, cf.name, errors.New("file already exists"))
	}

	dir := path.Dir(filePath)
	if dir != "." {
		os.MkdirAll(dir, 0755)
	}

	f, err := os.Create(filePath)
	if err != nil {
		return toolErrorResponse(toolCall.ID, cf.name, err)
	}
	f.Close()

	return toolSuccessResponse(toolCall.ID, cf.name, "status", "SUCCESS")
}

// =============================================================================
// GoCodeEditor Tool

// GoCodeEditor represents a tool that can be used to edit Go source code files.
type GoCodeEditor struct {
	name string
}

// RegisterGoCodeEditor creates a new instance of the GoCodeEditor tool and
// loads it into the provided tools map.
func RegisterGoCodeEditor(tools map[string]Tool) client.D {
	gce := GoCodeEditor{
		name: "tool_go_code_editor",
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
func (gce *GoCodeEditor) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(toolCall.ID, gce.name, fmt.Errorf("%s", r))
		}
	}()

	path := toolCall.Function.Arguments["path"].(string)
	lineNumber := int(toolCall.Function.Arguments["line_number"].(float64))
	typeChange := strings.TrimSpace(toolCall.Function.Arguments["type_change"].(string))
	lineChange := strings.TrimSpace(toolCall.Function.Arguments["line_change"].(string))

	content, err := os.ReadFile(path)
	if err != nil {
		return toolErrorResponse(toolCall.ID, gce.name, err)
	}

	fset := token.NewFileSet()
	lines := strings.Split(string(content), "\n")

	if lineNumber < 1 || lineNumber > len(lines) {
		return toolErrorResponse(toolCall.ID, gce.name, fmt.Errorf("line number %d is out of range (1-%d)", lineNumber, len(lines)))
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
		return toolErrorResponse(toolCall.ID, gce.name, fmt.Errorf("unsupported change type: %s, please inform the user", typeChange))
	}

	modifiedContent := strings.Join(lines, "\n")

	_, err = parser.ParseFile(fset, path, modifiedContent, parser.ParseComments)
	if err != nil {
		return toolErrorResponse(toolCall.ID, gce.name, fmt.Errorf("syntax error after modification: %s, please inform the user", err))
	}

	formattedContent, err := format.Source([]byte(modifiedContent))
	if err != nil {
		formattedContent = []byte(modifiedContent)
	}

	err = os.WriteFile(path, formattedContent, 0644)
	if err != nil {
		return toolErrorResponse(toolCall.ID, gce.name, fmt.Errorf("write file: %s", err))
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

	return toolSuccessResponse(toolCall.ID, gce.name, "message", action)
}
