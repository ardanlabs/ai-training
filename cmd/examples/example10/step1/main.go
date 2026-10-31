// This example shows you how to create a basic MCP interaction where the Server
// runs as a service and extends the set of tools as endpoints. The Client makes
// a call to the Server via the MCP SSE protocol. The makefile shows you the
// raw CURL calls that are used to make the client/server interaction.
//
// # Running the example:
//
//	$ make example10-step1
//
// # This doesn't require you to run any additional services.
//
// # Extra reading and watching:
//
//  https://github.com/modelcontextprotocol/go-sdk
//  https://github.com/modelcontextprotocol/go-sdk/blob/main/design/design.md
//  https://github.com/orgs/modelcontextprotocol/discussions/364

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	host := flag.String("host", "localhost", "host to listen on")
	port := flag.String("port", "8082", "port to listen on")
	flag.Parse()

	if err := run(*host, *port); err != nil {
		log.Fatal(err)
	}
}

func run(host string, port string) error {

	// -------------------------------------------------------------------------
	// Start a MCP Server and listen for requests.

	if err := mcpListenAndServe(host, port); err != nil {
		return err
	}

	fmt.Println("\nTesting MCP Client coded against the MCP Server on Port:", port)

	// -------------------------------------------------------------------------
	// Make three MCP based tool calls.

	if err := mcpClientCall(host, port, "tool_list_files", map[string]any{"filter": "*.go"}); err != nil {
		return err
	}

	if err := mcpClientCall(host, port, "tool_read_files", map[string]any{"path": "file1.txt"}); err != nil {
		return err
	}

	if err := mcpClientCall(host, port, "tool_shell_command", map[string]any{"command": []string{"find", ".", "-name", "*.go", "-not", "-path", "./vendor/*"}}); err != nil {
		return err
	}

	fmt.Print("\nTest Successful\n\n")

	// -------------------------------------------------------------------------
	// Hold the server open for extended testing, such as using CURL.

	fmt.Println("Holding the server open for extended testing.\n\nPress Ctrl+C to exit.")

	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt)
	<-ch

	fmt.Println("\nServer Down")

	return nil
}

// =============================================================================
// Basic MCP server implementation

// mcpListenAndServe opens the listener before returning, so the server is
// ready to accept connections when the client calls below are made.
func mcpListenAndServe(host string, port string) error {
	fileOperations := mcp.NewServer(&mcp.Implementation{Name: "file operations", Version: "v1.0.0"}, nil)
	mcp.AddTool(fileOperations, &mcp.Tool{Name: "tool_list_files", Description: "lists files"}, ListFilesMCPHandler)
	mcp.AddTool(fileOperations, &mcp.Tool{Name: "tool_read_files", Description: "reads a file"}, ReadFilesMCPHandler)
	mcp.AddTool(fileOperations, &mcp.Tool{Name: "tool_shell_command", Description: "runs a shell command"}, ShellCommandMCPHandler)

	// -------------------------------------------------------------------------

	addr := net.JoinHostPort(host, port)

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("mcp server: %w", err)
	}

	log.Printf("Server: MCP servers serving at %s", addr)

	// -------------------------------------------------------------------------

	f := func(request *http.Request) *mcp.Server {
		url := request.URL.Path
		log.Printf("Server: Handling request for URL %s\n", url)

		switch url {
		case "/tool_list_files":
			return fileOperations

		case "/tool_read_files":
			return fileOperations

		case "/tool_shell_command":
			return fileOperations

		default:
			return mcp.NewServer(&mcp.Implementation{Name: "unknown_tool", Version: "v1.0.0"}, nil)
		}
	}

	handler := mcp.NewSSEHandler(f, &mcp.SSEOptions{})

	go func() {
		log.Fatal(http.Serve(ln, handler))
	}()

	return nil
}

// =============================================================================

type ListFilesParams struct {
	Filter string `json:"filter" jsonschema:"a possible filter to use"`
}

func ListFilesMCPHandler(ctx context.Context, req *mcp.CallToolRequest, params ListFilesParams) (*mcp.CallToolResult, any, error) {
	data := struct {
		Status string   `json:"status"`
		Filter string   `json:"filter"`
		Files  []string `json:"files"`
	}{
		Status: "SUCCESS",
		Filter: params.Filter,
		Files: []string{
			"file1.txt",
			"file2.txt",
			"file3.txt",
		},
	}

	d, err := json.Marshal(data)
	if err != nil {
		return nil, nil, err
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(d),
		}},
	}, nil, nil
}

// =============================================================================

type ReadFilesParams struct {
	Path string `json:"path" jsonschema:"the path to the file to read"`
}

func ReadFilesMCPHandler(ctx context.Context, req *mcp.CallToolRequest, params ReadFilesParams) (*mcp.CallToolResult, any, error) {
	data := struct {
		Status  string `json:"status"`
		Path    string `json:"path"`
		Content string `json:"content"`
	}{
		Status:  "SUCCESS",
		Path:    params.Path,
		Content: "Hello World",
	}

	d, err := json.Marshal(data)
	if err != nil {
		return nil, nil, err
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(d),
		}},
	}, nil, nil
}

// =============================================================================

type ShellCommandParams struct {
	Command []string `json:"command" jsonschema:"the command and arguments to execute"`
}

// ShellCommandMCPHandler is a VERY DANGEROUS tool that should never be implemented
// like this. I am showing this because you could leverage CLI tooling to do
// things like list files, read files, etc, but you need some way to limit the
// commands that can be executed with a level of security.
func ShellCommandMCPHandler(ctx context.Context, req *mcp.CallToolRequest, params ShellCommandParams) (*mcp.CallToolResult, any, error) {
	if len(params.Command) == 0 {
		return nil, nil, fmt.Errorf("no command provided")
	}

	var out, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, params.Command[0], params.Command[1:]...)
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("%w: %s", err, strings.TrimSpace(stderr.String()))
	}

	data := struct {
		Command []string `json:"command"`
		Output  string   `json:"output"`
	}{
		Command: params.Command,
		Output:  out.String(),
	}

	d, err := json.Marshal(data)
	if err != nil {
		return nil, nil, err
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(d),
		}},
	}, nil, nil
}

// =============================================================================
// Basic MCP client code

func mcpClientCall(host string, port string, tool string, arguments map[string]any) error {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// Connect to the MCP server.

	fmt.Println("Establish client connection on Port:", port)

	addr := fmt.Sprintf("http://%s:%s/%s", host, port, tool)
	client := mcp.NewClient(&mcp.Implementation{Name: "mcp-client", Version: "v1.0.0"}, nil)
	transport := mcp.SSEClientTransport{
		Endpoint: addr,
	}

	fmt.Print("Client: Connecting to MCP Server\n\n")

	session, err := client.Connect(ctx, &transport, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to MCP server: %w", err)
	}
	defer session.Close()

	// -------------------------------------------------------------------------
	// Call the tool.

	params := &mcp.CallToolParams{
		Name:      tool,
		Arguments: arguments,
	}

	fmt.Printf("\nClient: Calling Tool: %s(%v)\n\n", params.Name, params.Arguments)

	res, err := session.CallTool(ctx, params)
	if err != nil {
		return fmt.Errorf("failed to call tool: %w", err)
	}

	// -------------------------------------------------------------------------
	// Check for errors and display the result.

	// A tool that fails returns the reason as text content, so the text is
	// pulled out of the content either way. Printing res.Content directly
	// would only show pointer addresses.
	var text strings.Builder
	for _, c := range res.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			text.WriteString(tc.Text)
		}
	}

	if res.IsError {
		return fmt.Errorf("tool call failed: %s", text.String())
	}

	fmt.Print(text.String())
	fmt.Print("\n")

	return nil
}
