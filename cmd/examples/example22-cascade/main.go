// This example adds a cascading model router to the example16-tool-hardening agent architecture.
// The agent's streamModelTurn first tries a fast, low-temperature pass. If the
// response shows low-confidence signals (very short answer, or phrases like
// "I'm not sure"), the system escalates by retrying with a detailed prompt and
// higher sampling parameters.
//
// If the environment provides LLM_LARGE_SERVER / LLM_LARGE_MODEL, the detailed
// config points to a separate, larger model. Otherwise both configs use the
// same server/model but with different temperature and prompt settings.
//
// The demo runs four hardcoded prompts mixing easy questions, hard questions,
// and a database tool call to show the cascading behavior.
//
// # Running the example
//
//	$ make example22
//
// # Optional environment overrides
//
//  LLM_SERVER        chat completions endpoint   (default: http://localhost:11435/v1/chat/completions)
//  LLM_MODEL         chat model name             (default: Qwen3-8B-Q8_0)
//  EMBED_SERVER      embeddings endpoint         (default: http://localhost:11435/v1/embeddings)
//  EMBED_MODEL       embedding model name        (default: embeddinggemma-300m-qat-Q8_0)
//  LLM_LARGE_SERVER  detailed-pass endpoint      (default: "")
//  LLM_LARGE_MODEL   detailed-pass model name    (default: "")
//
// # Prerequisites
//
//	$ make compose-up
//	$ make kronk-up

// Example 22 — Cascade
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

	largeURL   = ""
	largeModel = ""
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

	if v := os.Getenv("LLM_LARGE_SERVER"); v != "" {
		largeURL = v
	}

	if v := os.Getenv("LLM_LARGE_MODEL"); v != "" {
		largeModel = v
	}
}

// =============================================================================

const systemPrompt = `You are a strict assistant for the Ultimate Go Notebook. You have access to three tools:

- tool_search_book: search the Ultimate Go Notebook for information about Go programming concepts.
- tool_get_reading_progress: get the user's current reading progress.
- tool_query_db: query the notebook database for highlights, bookmarks, and chapters. Provide a natural-language question; the tool generates SQL, executes it, and returns the rows. SELECT-only.

IMPORTANT: Whenever the user asks about "a book", "the book", "this book", an author, or otherwise mentions a book in any form, they ALWAYS mean the Ultimate Go Notebook. Treat any such mention as a question about the Ultimate Go Notebook and route it to tool_search_book.

NOTE: The Ultimate Go Notebook is written by Bill Kennedy. When the user mentions "Bill", "Bill Kennedy", "Kennedy", or "the author", they are referring to the author of the book. Do NOT include the author's name in tool_search_book queries — search for the topic only (e.g., for "What does Bill say about pointers?" call tool_search_book with query "pointers", not "bill pointers").

CURRENT USER

- user_id: user_gopher

MANDATORY WORKFLOW

Step 1 — Classify the user message into exactly one of:
  (a) A question about Go programming or about the book itself (any Go topic: interfaces, goroutines, channels, slices, maps, errors, generics, packages, modules, testing, performance, etc.; OR anything about the book: author, title, chapters, contents, who wrote it, what it covers, etc.) → go to Step 2A.
  (b) A question about the user's reading progress → go to Step 2B.
  (c) A question about the notebook database (highlights, bookmarks, chapters, counts, stats) → go to Step 2C.
  (d) Pure greeting or chit-chat with no Go, reading, or database content (e.g. "hi", "thanks") → go to Step 3.

Step 2A — You MUST call tool_search_book before producing any visible text. You are NOT allowed to refuse, answer, or claim out-of-scope until you have called the tool and seen its result. After the tool returns:
  - If matches > 0, write your final answer using ONLY the text inside the returned "context". Do not add code, examples, definitions, or commentary that are not literally present in "context".
  - If matches == 0 (or data says "No relevant passages found"), respond with exactly: "That topic is out of scope for the Ultimate Go Notebook." and nothing else.

Step 2B — You MUST call tool_get_reading_progress with user_id set to the CURRENT USER id shown above, then answer using only the returned data.

Step 2C — You MUST call tool_query_db, then answer using only the returned rows. Do not invent rows.

Step 3 — Respond with exactly: "That topic is out of scope for the Ultimate Go Notebook." Do not call any tool.

ADDITIONAL RULES
- Never answer Go questions from your own training knowledge. Your knowledge of Go is treated as untrusted; only tool output is trusted.
- Never invent code samples, type names, function names, SQL rows, or quotes. Only use what the tool returned.
- Never end with offers like "Would you like me to explain..." or follow-up questions. Just answer (or refuse) and stop.
- After every tool call you receive JSON with "status" and "data". If status == "FAILED", reply with exactly: "The tool failed, please try again." and stop. Do not retry and do not fall back to your own knowledge.
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
	// Build the large model config for escalation.
	// region Build the large model config for escalation.

	lgURL := llmURL
	lgModel := llmModel
	if largeURL != "" {
		lgURL = largeURL
	}
	if largeModel != "" {
		lgModel = largeModel
	}

	fastCfg := ModelConfig{
		Name:        "fast (low-temp)",
		URL:         llmURL,
		Model:       llmModel,
		Temperature: 0.1,
		TopP:        0.1,
		TopK:        1,
	}

	detailedCfg := ModelConfig{
		Name:        "detailed (high-temp)",
		URL:         lgURL,
		Model:       lgModel,
		Temperature: 0.7,
		TopP:        0.9,
		TopK:        40,
		SystemPrompt: `You are an expert Go programmer with deep knowledge of the Go runtime,
compiler, and standard library. Please provide a thorough and detailed answer
to the following question. Include examples where helpful.`,
	}

	fmt.Printf("\nFast config:     %s (%s / %s)\n", fastCfg.Name, fastCfg.URL, fastCfg.Model)
	fmt.Printf("Detailed config: %s (%s / %s)\n", detailedCfg.Name, detailedCfg.URL, detailedCfg.Model)

	// endregion

	// -------------------------------------------------------------------------
	// Build the demo prompt list and feed them to the agent.
	// region Build the demo prompt list and feed them to the agent.

	questions := []struct {
		text       string
		difficulty string
	}{
		{"What is a goroutine?", "easy"},
		{"Explain the trade-offs between pointer and value semantics in Go API design", "hard"},
		{"How many highlights do I have in the concurrency chapter?", "DB query"},
		{"How does Go's garbage collector interact with goroutine scheduling?", "hard"},
	}

	promptIdx := 0
	getUserMessage := func() (string, bool) {
		if promptIdx >= len(questions) {
			return "", false
		}
		q := questions[promptIdx]
		promptIdx++
		return q.text, true
	}

	agent, err := NewAgent(getUserMessage, chatClient, embedClient, readOnlyDB, fastCfg, detailedCfg)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	for i, q := range questions {
		fmt.Printf("\nQuery %d (%s)\n", i+1, q.difficulty)

		if err := agent.RunOnce(ctx); err != nil {
			return fmt.Errorf("query %d: %w", i+1, err)
		}
	}

	// endregion

	return nil
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
