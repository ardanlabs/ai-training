// This example builds on the example16-tool-hardening agent architecture by adding an adaptive
// retrieval gate to the tool_search_book tool. Before performing a vector
// search in pgvector, a classifier prompt determines whether the question
// actually needs domain context from the Ultimate Go Notebook or can be
// answered from general knowledge. This avoids injecting irrelevant RAG
// context into the model's prompt for questions like "What is 2+2?".
//
// The demo runs four hardcoded prompts through the agent — two general
// knowledge and two domain-specific — to demonstrate the classification
// decision and final answer for each.
//
// # Running the example
//
//	$ make example21-step1
//
// # Optional environment overrides
//
//  LLM_SERVER    chat completions endpoint (default: http://localhost:11435/v1/chat/completions)
//  LLM_MODEL     chat model name           (default: Qwen3-8B-Q8_0)
//  EMBED_SERVER  embeddings endpoint       (default: http://localhost:11435/v1/embeddings)
//  EMBED_MODEL   embedding model name      (default: embeddinggemma-300m-qat-Q8_0)
//
// # Prerequisites
//
//	$ make compose-up
//	$ make kronk-up

// Example 21 — Step 1 — Adaptive Retrieval
package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/jmoiron/sqlx"
)

const (
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5432"
	dbName     = "postgres"
)

var (
	llmURL     = "http://localhost:11435/v1/chat/completions"
	llmModel   = "Qwen3-8B-Q8_0"
	embedURL   = "http://localhost:11435/v1/embeddings"
	embedModel = "embeddinggemma-300m-qat-Q8_0"
)

func init() {
	if v := os.Getenv("LLM_SERVER"); v != "" {
		llmURL = v
	}

	if v := os.Getenv("LLM_MODEL"); v != "" {
		llmModel = v
	}

	if v := os.Getenv("EMBED_SERVER"); v != "" {
		embedURL = v
	}

	if v := os.Getenv("EMBED_MODEL"); v != "" {
		embedModel = v
	}
}

// =============================================================================

const systemPrompt = `You are an assistant for the Ultimate Go Notebook. You have access to three tools:

- tool_search_book: search the Ultimate Go Notebook. The tool runs an internal retrieval gate that rejects general-knowledge queries before any vector search, so it is safe to call optimistically for ANY informational question.
- tool_get_reading_progress: get the user's current reading progress.
- tool_query_db: query the notebook database for highlights, bookmarks, and chapters. Provide a natural-language question; the tool generates SQL, executes it, and returns the rows. SELECT-only.

CURRENT USER

- user_id: user_gopher

WORKFLOW

- For EVERY user question that is not a pure greeting, you MUST call tool_search_book BEFORE producing any visible text. Do not attempt to classify the question yourself, do not decide in advance whether it is in scope, and do not predict what the tool will return. The tool's internal gate is the only source of truth. Call it with a short topical query (e.g. for "What is 2+2?" call tool_search_book with query "2+2"; for "What is the capital of France?" call tool_search_book with query "capital of France"; for "What does the book say about goroutines?" call tool_search_book with query "goroutines").
- For questions about reading progress, also call tool_get_reading_progress with user_id set to the CURRENT USER id above.
- For questions about highlights, bookmarks, or chapter stats, also call tool_query_db.
- For pure greetings or chit-chat ("hi", "thanks"), reply briefly without calling any tool.

AFTER tool_search_book RETURNS

- If "context" is present, write your answer using ONLY the text inside "context". Do not add code, examples, or commentary that are not literally present there.
- If the response indicates the question is general knowledge, or that no relevant passages were found, respond with exactly: "That topic is out of scope for the Ultimate Go Notebook." and nothing else.

ADDITIONAL RULES

- Never invent code samples, type names, function names, SQL rows, or quotes. Only use what the tool returned.
- Never end with offers like "Would you like me to explain..." or follow-up questions. Just answer (or refuse) and stop.
- After every tool call you receive JSON with "status" and "data". If status == "FAILED", reply with exactly: "The tool failed, please try again." and stop.
`

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	chatClient := client.NewLLM(llmURL, llmModel)
	embedClient := client.NewLLM(embedURL, embedModel)

	// -------------------------------------------------------------------------
	// Connect to PostgreSQL and seed sample data.
	// region Connect to PostgreSQL and seed sample data.

	db, err := sqldb.Open(sqldb.Config{
		User:         dbUser,
		Password:     dbPassword,
		Host:         dbHost,
		Name:         dbName,
		Schema:       "public",
		MaxIdleConns: 2,
		MaxOpenConns: 5,
		DisableTLS:   true,
	})
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer db.Close()

	if err := sqldb.StatusCheck(ctx, db); err != nil {
		return fmt.Errorf("status check: %w", err)
	}

	if err := initSQLDB(ctx, db); err != nil {
		return fmt.Errorf("init sql db: %w", err)
	}

	if err := initDocuments(ctx, db, embedClient); err != nil {
		return fmt.Errorf("init documents: %w", err)
	}

	// endregion

	// -------------------------------------------------------------------------
	// Open a second, read-only connection pool for the tools. Every connection
	// in this pool starts with default_transaction_read_only=on, so SQL the
	// LLM generates cannot accidentally write to the database.
	// region Open a second, read-only connection pool for the tools. Every connection

	readOnlyDB, err := sqldb.Open(sqldb.Config{
		User:         dbUser,
		Password:     dbPassword,
		Host:         dbHost,
		Name:         dbName,
		Schema:       "public",
		MaxIdleConns: 2,
		MaxOpenConns: 5,
		DisableTLS:   true,
		ReadOnly:     true,
	})
	if err != nil {
		return fmt.Errorf("open read-only db: %w", err)
	}
	defer readOnlyDB.Close()

	// endregion

	// -------------------------------------------------------------------------
	// Demo: run four hardcoded prompts through the agent.
	// region Demo: run four hardcoded prompts through the agent.

	prompts := []string{
		"What does the book say about Rust's borrow checker?",
		"What does the Go Notebook say about goroutines?",
		"How does Python handle decorators?",
		"How do pointer receivers work in Go?",
	}

	promptIdx := 0
	getUserMessage := func() (string, bool) {
		if promptIdx >= len(prompts) {
			return "", false
		}
		msg := prompts[promptIdx]
		promptIdx++
		return msg, true
	}

	agent, err := NewAgent(getUserMessage, chatClient, embedClient, readOnlyDB)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	// endregion

	return agent.Run(ctx)
}

// =============================================================================
// SQL DB setup

func initSQLDB(ctx context.Context, db *sqlx.DB) error {
	if err := dbExecute(ctx, db, schemaSQL); err != nil {
		return fmt.Errorf("schema: %w", err)
	}

	if err := dbExecute(ctx, db, insertSQL); err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	return nil
}

func dbExecute(ctx context.Context, db *sqlx.DB, query string) error {
	tx, err := db.Begin()
	if err != nil {
		return err
	}

	defer func() {
		if errTx := tx.Rollback(); errTx != nil {
			if errors.Is(errTx, sql.ErrTxDone) {
				return
			}
			err = fmt.Errorf("rollback: %w", errTx)
		}
	}()

	if _, err := tx.Exec(query); err != nil {
		return fmt.Errorf("exec: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}

	return nil
}

// =============================================================================
// Documents seeding

func initDocuments(ctx context.Context, db *sqlx.DB, embedLLM *client.LLM) error {
	docs := []struct{ name, text string }{
		{"Goroutine", "A goroutine is a lightweight concurrent function managed by the Go runtime."},
		{"Channel", "Channels let goroutines communicate and synchronize without shared-memory bookkeeping."},
		{"Pointer Receiver", "A method with a pointer receiver can modify the original value because it receives the address of that value."},
		{"Addressable Value", `In Go, bill.changeEmail("bill@hotmail.com") compiles because bill is addressable, so the compiler can automatically take its address to call a pointer receiver method.`},
		{"Interface", "Interfaces in Go describe behavior through method sets rather than inheritance."},
	}

	firstEmbed, err := embedLLM.EmbedText(ctx, docs[0].text)
	if err != nil {
		return fmt.Errorf("embed first doc: %w", err)
	}
	dimensions := len(firstEmbed)

	if err := sqldb.ExecContext(ctx, db, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return fmt.Errorf("create extension vector: %w", err)
	}

	if err := sqldb.ExecContext(ctx, db, `DROP TABLE IF EXISTS documents`); err != nil {
		return fmt.Errorf("drop documents: %w", err)
	}

	query := fmt.Sprintf(`CREATE TABLE documents (
		id BIGINT PRIMARY KEY,
		text TEXT NOT NULL,
		embedding VECTOR(%d) NOT NULL
	)`, dimensions)
	if err := sqldb.ExecContext(ctx, db, query); err != nil {
		return fmt.Errorf("create documents: %w", err)
	}

	const insertQ = `INSERT INTO documents (id, text, embedding) VALUES ($1, $2, $3::vector)`
	if _, err := db.ExecContext(ctx, insertQ, 0, docs[0].text, vector.FormatPGVector(firstEmbed)); err != nil {
		return fmt.Errorf("insert doc 0: %w", err)
	}

	for i := 1; i < len(docs); i++ {
		emb, err := embedLLM.EmbedText(ctx, docs[i].text)
		if err != nil {
			return fmt.Errorf("embed doc %d: %w", i, err)
		}
		if _, err := db.ExecContext(ctx, insertQ, i, docs[i].text, vector.FormatPGVector(emb)); err != nil {
			return fmt.Errorf("insert doc %d: %w", i, err)
		}
	}

	return nil
}

// =============================================================================
// Search helpers (used transitively by the SearchBook tool)

type searchResult struct {
	ID         int
	Text       string
	Distance   float64
	Similarity float64
}

func searchDocuments(ctx context.Context, db *sqlx.DB, llm *client.LLM, query string, topN int) ([]searchResult, error) {
	embedding, err := llm.EmbedText(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	const stmt = `
SELECT
	id,
	text,
	embedding <=> $1::vector AS distance,
	1 - (embedding <=> $1::vector) AS similarity
FROM
	documents
ORDER BY
	embedding <=> $1::vector
LIMIT $2
`

	rows, err := db.QueryContext(ctx, stmt, vector.FormatPGVector(embedding), topN)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	var results []searchResult
	for rows.Next() {
		var r searchResult
		if err := rows.Scan(&r.ID, &r.Text, &r.Distance, &r.Similarity); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}
		results = append(results, r)
	}

	return results, rows.Err()
}
