package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// startMCPServer creates and runs an MCP server that exposes tool_search_book
// and tool_query_db, both backed by the real PostgreSQL + LLM dependencies
// in mcpDeps.
func startMCPServer(host string, port string) {
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "gonotebook-tools",
		Version: "v1.0.0",
	}, nil)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "tool_search_book",
		Description: "Search the Ultimate Go Notebook for information about Go programming concepts. Returns the most relevant passages from the book.",
	}, SearchBookMCPHandler)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "tool_query_db",
		Description: "Query the notebook database for highlights, bookmarks, and chapters of the Ultimate Go Notebook. Provide a natural-language question and the tool will generate a SELECT statement, execute it, and return the rows.",
	}, QueryDBMCPHandler)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "tool_get_reading_progress",
		Description: "Get the current reading progress for the user — which chapter and page they last read, and what percentage of the book they have completed.",
	}, ReadingProgressMCPHandler)

	addr := fmt.Sprintf("%s:%s", host, port)
	log.Printf("MCP Server: listening at %s", addr)

	handler := mcp.NewSSEHandler(func(_ *http.Request) *mcp.Server {
		return server
	}, &mcp.SSEOptions{})

	log.Fatal(http.ListenAndServe(addr, handler))
}
