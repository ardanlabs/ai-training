package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/rag"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	fntools "github.com/ardanlabs/ai-training/foundation/tools"
	"github.com/jmoiron/sqlx"
)

var (
	//go:embed sql/schema.sql
	schemaSQL string

	//go:embed sql/insert.sql
	insertSQL string

	//go:embed prompts/query.txt
	queryPrompt string
)

// =============================================================================
// QueryDB Tool — translates natural language to SQL, executes, returns results.

type QueryDB struct {
	name string
	llm  *client.LLM
	db   *sqlx.DB
}

func RegisterQueryDB(tools map[string]fntools.Tool, llm *client.LLM, db *sqlx.DB) client.D {
	t := QueryDB{
		name: "tool_query_db",
		llm:  llm,
		db:   db,
	}
	tools[t.name] = &t

	return t.toolDocument()
}

func (t *QueryDB) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        t.name,
			"description": "Query the notebook database for highlights, bookmarks, and chapters of the Ultimate Go Notebook. Provide a natural-language question and the tool will generate a SELECT statement, execute it, and return the rows.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"question": client.D{
						"type":        "string",
						"description": "The natural-language question to answer using the database, e.g. 'How many highlights do I have for concurrency?'",
					},
				},
				"required": []string{"question"},
			},
		},
	}
}

func (t *QueryDB) Call(ctx context.Context, toolCall client.ToolCall) client.D {
	question, _ := toolCall.Function.Arguments["question"].(string)
	if question == "" {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing question argument"))
	}

	// Ask the LLM to generate a SQL query.
	sqlQuery, err := t.llm.ChatCompletions(ctx, fmt.Sprintf(queryPrompt, question))
	if err != nil {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("generate SQL: %w", err))
	}

	fmt.Printf("  Generated SQL: %s\n", strings.TrimSpace(sqlQuery))

	// Execute the SQL query.
	data := []map[string]any{}
	if err := sqldb.QueryMap(ctx, t.db, sqlQuery, &data); err != nil {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("execute SQL: %w", err))
	}

	rowsJSON, err := json.Marshal(data)
	if err != nil {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("marshal results: %w", err))
	}

	return fntools.SuccessResponse(toolCall.ID,
		"results", string(rowsJSON),
		"row_count", len(data),
	)
}

// =============================================================================
// SearchBook Tool — wraps rag.SearchDocuments for retrieval.

type SearchBook struct {
	name     string
	embedLLM *client.LLM
	db       *sqlx.DB
}

func RegisterSearchBook(tools map[string]fntools.Tool, embedLLM *client.LLM, db *sqlx.DB) client.D {
	t := SearchBook{
		name:     "tool_search_book",
		embedLLM: embedLLM,
		db:       db,
	}
	tools[t.name] = &t

	return t.toolDocument()
}

func (t *SearchBook) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        t.name,
			"description": "Search the Ultimate Go Notebook for information about Go programming concepts. Returns the most relevant passages from the book.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"query": client.D{
						"type":        "string",
						"description": "The search query about Go programming concepts, e.g. 'How do interfaces work in Go?'",
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

func (t *SearchBook) Call(ctx context.Context, toolCall client.ToolCall) client.D {
	query, _ := toolCall.Function.Arguments["query"].(string)
	if query == "" {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing query argument"))
	}

	results, err := rag.SearchDocuments(ctx, t.db, t.embedLLM, query, 5)
	if err != nil {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("search: %w", err))
	}

	var passages []string
	for _, r := range results {
		if r.Similarity >= 0.50 {
			passages = append(passages, r.Text)
		}
	}

	if len(passages) == 0 {
		return fntools.SuccessResponse(toolCall.ID,
			"answer", "No relevant passages found in the Go Notebook for this query.",
			"matches", 0,
		)
	}

	return fntools.SuccessResponse(toolCall.ID,
		"context", strings.Join(passages, "\n---\n"),
		"matches", len(passages),
	)
}

// =============================================================================
// ReadingProgress Tool — stub that returns fixed reading-progress data.

type ReadingProgress struct {
	name string
}

func RegisterReadingProgress(tools map[string]fntools.Tool) client.D {
	t := ReadingProgress{
		name: "tool_get_reading_progress",
	}
	tools[t.name] = &t

	return t.toolDocument()
}

func (t *ReadingProgress) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        t.name,
			"description": "Get the current reading progress for the user — which chapter and page they last read, and what percentage of the book they have completed.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"user_id": client.D{
						"type":        "string",
						"description": "ID of the user whose progress to retrieve. Any non-empty string is accepted in this demo.",
					},
				},
				"required": []string{"user_id"},
			},
		},
	}
}

func (t *ReadingProgress) Call(ctx context.Context, toolCall client.ToolCall) client.D {
	return fntools.SuccessResponse(toolCall.ID,
		"current_chapter", "Concurrency",
		"chapter_number", 6,
		"current_page", 142,
		"total_pages", 240,
		"percent_complete", 59.2,
		"last_read", "2026-05-15T14:30:00Z",
	)
}
