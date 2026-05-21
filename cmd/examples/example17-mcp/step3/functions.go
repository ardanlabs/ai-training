package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/rag"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

var (
	//go:embed prompts/query.txt
	queryPrompt string

	//go:embed sql/schema.sql
	schemaSQL string

	//go:embed sql/insert.sql
	insertSQL string
)

// =============================================================================
// SearchBook MCP handler — runs the same RAG search as example16-tool-hardening.

// SearchBookParams defines the parameters for the search book tool.
type SearchBookParams struct {
	Query string `json:"query" jsonschema:"the search query about Go programming concepts"`
}

// SearchBookMCPHandler embeds the query, performs a pgvector similarity
// search against the documents table, and returns the top passages.
func SearchBookMCPHandler(ctx context.Context, _ *mcp.CallToolRequest, params SearchBookParams) (*mcp.CallToolResult, any, error) {
	if strings.TrimSpace(params.Query) == "" {
		return mcpFailure("missing or empty 'query' argument")
	}

	results, err := rag.SearchDocuments(ctx, mcpDeps.db, mcpDeps.embedLLM, params.Query, 5)
	if err != nil {
		return mcpFailure(fmt.Sprintf("search: %v", err))
	}

	var passages []string
	for _, r := range results {
		if r.Similarity >= 0.50 {
			passages = append(passages, r.Text)
		}
	}

	if len(passages) == 0 {
		return mcpSuccess(map[string]any{
			"answer":  "No relevant passages found in the Go Notebook for this query.",
			"matches": 0,
		})
	}

	return mcpSuccess(map[string]any{
		"context": strings.Join(passages, "\n---\n"),
		"matches": len(passages),
	})
}

// =============================================================================
// QueryDB MCP handler — translates NL → SQL via the LLM, executes against
// PostgreSQL, returns the rows.

// QueryDBParams defines the parameters for the query database tool.
type QueryDBParams struct {
	Question string `json:"question" jsonschema:"the natural-language question to answer using the database"`
}

// QueryDBMCPHandler asks the chat LLM to generate SQL from the question,
// runs that SQL, and returns the result rows as a JSON string.
func QueryDBMCPHandler(ctx context.Context, _ *mcp.CallToolRequest, params QueryDBParams) (*mcp.CallToolResult, any, error) {
	if strings.TrimSpace(params.Question) == "" {
		return mcpFailure("missing or empty 'question' argument")
	}

	sqlQuery, err := mcpDeps.chatLLM.ChatCompletions(ctx, fmt.Sprintf(queryPrompt, params.Question))
	if err != nil {
		return mcpFailure(fmt.Sprintf("generate SQL: %v", err))
	}

	fmt.Printf("  Generated SQL: %s\n", strings.TrimSpace(sqlQuery))

	rows := []map[string]any{}
	if err := sqldb.QueryMap(ctx, mcpDeps.db, sqlQuery, &rows); err != nil {
		return mcpFailure(fmt.Sprintf("execute SQL: %v", err))
	}

	rowsJSON, err := json.Marshal(rows)
	if err != nil {
		return mcpFailure(fmt.Sprintf("marshal results: %v", err))
	}

	return mcpSuccess(map[string]any{
		"results":   string(rowsJSON),
		"row_count": len(rows),
	})
}

// =============================================================================
// ReadingProgress MCP handler — stub that returns fixed reading-progress data.

// ReadingProgressParams defines the (optional) parameters for the reading
// progress tool.
type ReadingProgressParams struct {
	UserID string `json:"user_id" jsonschema:"ID of the user whose progress to retrieve. Any non-empty string is accepted in this demo."`
}

// ReadingProgressMCPHandler returns a fixed progress payload demonstrating
// the third MCP tool without touching the database.
func ReadingProgressMCPHandler(ctx context.Context, _ *mcp.CallToolRequest, params ReadingProgressParams) (*mcp.CallToolResult, any, error) {
	return mcpSuccess(map[string]any{
		"current_chapter":  "Concurrency",
		"chapter_number":   6,
		"current_page":     142,
		"total_pages":      240,
		"percent_complete": 59.2,
		"last_read":        "2026-05-15T14:30:00Z",
	})
}

// =============================================================================
// MCP envelope helpers — produce the {"status","data"} JSON the agent expects.

func mcpSuccess(data map[string]any) (*mcp.CallToolResult, any, error) {
	return mcpEnvelope("SUCCESS", data)
}

func mcpFailure(message string) (*mcp.CallToolResult, any, error) {
	return mcpEnvelope("FAILED", map[string]any{"error": message})
}

func mcpEnvelope(status string, data map[string]any) (*mcp.CallToolResult, any, error) {
	body, err := json.Marshal(struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: status,
		Data:   data,
	})
	if err != nil {
		return nil, nil, err
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: string(body)}},
	}, nil, nil
}

// Step 02 =====================================================================
// MCP Client helpers

// Step 02
// mcpListTools connects to the MCP server and returns the list of available tools.
func mcpListTools(ctx context.Context, host, port string) ([]*mcp.Tool, error) {
	addr := fmt.Sprintf("http://%s:%s/", host, port)

	cln := mcp.NewClient(&mcp.Implementation{
		Name:    "mcp-client",
		Version: "v1.0.0",
	}, nil)

	transport := mcp.SSEClientTransport{Endpoint: addr}

	session, err := cln.Connect(ctx, &transport, nil)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}
	defer session.Close()

	res, err := session.ListTools(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("list tools: %w", err)
	}

	return res.Tools, nil
}

// Step 03
// mcpClientCall connects to the MCP server, calls the named tool, and returns
// the text content of the result.
func mcpClientCall(ctx context.Context, host string, port string, tool string, arguments map[string]any) (string, error) {
	addr := fmt.Sprintf("http://%s:%s/%s", host, port, tool)

	cln := mcp.NewClient(&mcp.Implementation{
		Name:    "mcp-client",
		Version: "v1.0.0",
	}, nil)

	transport := mcp.SSEClientTransport{
		Endpoint: addr,
	}

	session, err := cln.Connect(ctx, &transport, nil)
	if err != nil {
		return "", fmt.Errorf("connect to MCP server: %w", err)
	}
	defer session.Close()

	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name:      tool,
		Arguments: arguments,
	})
	if err != nil {
		return "", fmt.Errorf("call tool: %w", err)
	}

	if res.IsError {
		return "", fmt.Errorf("tool call failed: %v", res.Content)
	}

	var result strings.Builder
	for _, c := range res.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			result.WriteString(tc.Text)
		}
	}

	return result.String(), nil
}
