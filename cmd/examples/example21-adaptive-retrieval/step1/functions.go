package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	fntools "github.com/ardanlabs/ai-training/foundation/tools"
	"github.com/jmoiron/sqlx"
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
// QueryDB Tool

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
						"description": "The natural-language question to answer using the database.",
					},
				},
				"required": []string{"question"},
			},
		},
	}
}

func (t *QueryDB) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	question, ok := toolCall.Function.Arguments["question"].(string)
	if !ok || question == "" {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing or invalid 'question' argument: expected a string"))
	}

	sqlQuery, err := t.llm.ChatCompletions(ctx, fmt.Sprintf(queryPrompt, question))
	if err != nil {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("generate SQL: %w", err))
	}

	fmt.Printf("  Generated SQL: %s\n", strings.TrimSpace(sqlQuery))

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
// SearchBook Tool (with adaptive retrieval gate)

type SearchBook struct {
	name     string
	chatLLM  *client.LLM
	embedLLM *client.LLM
	db       *sqlx.DB
}

func RegisterSearchBook(tools map[string]fntools.Tool, chatLLM *client.LLM, embedLLM *client.LLM, db *sqlx.DB) client.D {
	t := SearchBook{
		name:     "tool_search_book",
		chatLLM:  chatLLM,
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
						"description": "The search query about Go programming concepts.",
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

const classifierPrompt = `You are a question classifier. Your job is to determine if a question
requires specific domain knowledge from the Ultimate Go Notebook, or if it
can be answered from general knowledge.

Respond with exactly one of these two labels:
- NEEDS_CONTEXT — if the question is about Go programming concepts, the
  Ultimate Go Notebook, or specific technical details about Go.
- GENERAL_KNOWLEDGE — if the question is about general knowledge, math,
  geography, or anything not specifically about Go programming.

Do not explain your reasoning. Just output the label.

Question: %s`

func (t *SearchBook) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	query, ok := toolCall.Function.Arguments["query"].(string)
	if !ok || query == "" {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing or invalid 'query' argument: expected a string"))
	}

	// Step 1: Classify the query using the chat LLM.
	classification, err := t.classifyQuery(ctx, query)
	if err != nil {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("classify query: %w", err))
	}

	fmt.Printf("  Retrieval gate: %s\n", classification)

	// Step 2: If general knowledge, skip the vector search.
	if classification == "GENERAL_KNOWLEDGE" {
		return fntools.SuccessResponse(toolCall.ID,
			"answer", "This is a general knowledge question, no book context needed.",
			"classification", classification,
		)
	}

	// Step 3: Proceed with normal vector search.
	results, err := searchDocuments(ctx, t.db, t.embedLLM, query, 5)
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
			"classification", classification,
		)
	}

	return fntools.SuccessResponse(toolCall.ID,
		"context", strings.Join(passages, "\n---\n"),
		"matches", len(passages),
		"classification", classification,
	)
}

func (t *SearchBook) classifyQuery(ctx context.Context, query string) (string, error) {
	prompt := fmt.Sprintf(classifierPrompt, query)

	answer, err := t.chatLLM.ChatCompletions(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("chat completions: %w", err)
	}

	answer = strings.TrimSpace(answer)

	if strings.Contains(answer, "NEEDS_CONTEXT") {
		return "NEEDS_CONTEXT", nil
	}

	return "GENERAL_KNOWLEDGE", nil
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
