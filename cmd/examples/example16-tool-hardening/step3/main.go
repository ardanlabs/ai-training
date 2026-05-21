// This step takes step2 and layers a per-tool timeout on top of panic
// recovery. safeCallTool now opens a context.WithTimeout(toolTimeout) so a
// slow or hung tool cannot wedge the agent loop. The agent now has both
// panic recovery AND time-bounded execution for every tool call.
// tool_crash (carried over from step1) is still caught by recover();
// a hypothetical hung tool would now be cut off by the timeout.
//
// # Running the example
//
//	$ make example16
//
// # Optional environment overrides
//
//  LLM_SERVER     chat completions endpoint (default: http://localhost:11435/v1/chat/completions)
//  LLM_MODEL      chat model name (default: Qwen3-8B-Q8_0)
//  EMBED_SERVER   embeddings endpoint (default: http://localhost:11435/v1/embeddings)
//  EMBED_MODEL    embedding model name (default: embeddinggemma-300m-qat-Q8_0)
//
// # Prerequisites
//
//  - make compose-up
//  - make kronk-up
//
// # What this step adds over step2
//
//	The toolTimeout constant and a context.WithTimeout wrapper inside
//	safeCallTool, bounding the execution time of every tool dispatch.

// Example 16 — Step 3 — Add Tool Timeout
package main

import (
	"bufio"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
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

	// Step 03
	toolTimeout = 30 * time.Second
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

const systemPrompt = `You are a strict assistant for the Ultimate Go Notebook. You have access to four tools:

- tool_search_book: search the Ultimate Go Notebook for information about Go programming concepts.
- tool_get_reading_progress: get the user's current reading progress.
- tool_query_db: query the notebook database for highlights, bookmarks, and chapters. Provide a natural-language question; the tool generates SQL, executes it, and returns the rows. SELECT-only.
- tool_crash: deliberately panic. Call this ONLY when the user explicitly asks to crash a tool or demonstrate tool failure.

IMPORTANT: Whenever the user asks about "a book", "the book", "this book", an author, or otherwise mentions a book in any form, they ALWAYS mean the Ultimate Go Notebook. Treat any such mention as a question about the Ultimate Go Notebook and route it to tool_search_book.

NOTE: The Ultimate Go Notebook is written by Bill Kennedy. When the user mentions "Bill", "Bill Kennedy", "Kennedy", or "the author", they are referring to the author of the book. Do NOT include the author's name in tool_search_book queries — search for the topic only (e.g., for "What does Bill say about pointers?" call tool_search_book with query "pointers", not "bill pointers").

CURRENT USER

- user_id: user_gopher

MANDATORY WORKFLOW

Step 1 — Classify the user message into exactly one of:
  (a) A question about Go programming or about the book itself (any Go topic: interfaces, goroutines, channels, slices, maps, errors, generics, packages, modules, testing, performance, etc.; OR anything about the book: author, title, chapters, contents, who wrote it, what it covers, etc.) → go to Step 2A.
  (b) A question about the user's reading progress → go to Step 2B.
  (c) A question about the notebook database (highlights, bookmarks, chapters, counts, stats) → go to Step 2C.
  (d) An explicit request to crash a tool or demonstrate tool failure (e.g. "crash a tool", "show me a tool panic") → go to Step 2D.
  (e) Pure greeting or chit-chat with no Go, reading, database, or crash content (e.g. "hi", "thanks") → go to Step 3.

Step 2A — You MUST call tool_search_book before producing any visible text. You are NOT allowed to refuse, answer, or claim out-of-scope until you have called the tool and seen its result. After the tool returns:
  - If matches > 0, write your final answer using ONLY the text inside the returned "context". Do not add code, examples, definitions, or commentary that are not literally present in "context".
  - If matches == 0 (or data says "No relevant passages found"), respond with exactly: "That topic is out of scope for the Ultimate Go Notebook." and nothing else.

Step 2B — You MUST call tool_get_reading_progress with user_id set to the CURRENT USER id shown above, then answer using only the returned data.

Step 2C — You MUST call tool_query_db, then answer using only the returned rows. Do not invent rows.

Step 2D — You MUST call tool_crash. The tool will panic; the agent's behavior on that panic is the point of the demonstration.

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

	chatLLM := client.NewLLM(llmURL, llmModel)
	embedLLM := client.NewLLM(embedURL, embedModel)

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

	if err := initDocuments(ctx, db, embedLLM); err != nil {
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
	// Start the agent.
	// region Start the agent.

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}

		message := scanner.Text()
		switch strings.ToLower(strings.TrimSpace(message)) {
		case "quit", "/quit", "/exit", "/bye":
			return "", false
		default:
			return message, true
		}
	}

	agent, err := NewAgent(getUserMessage, chatLLM, embedLLM, readOnlyDB)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	// endregion

	return agent.Run(ctx)
}

// =============================================================================
// Database setup helpers

func initSQLDB(ctx context.Context, db *sqlx.DB) error {
	if err := dbExecute(ctx, db, schemaSQL); err != nil {
		return fmt.Errorf("schema: %w", err)
	}

	if err := dbExecute(ctx, db, insertSQL); err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	return nil
}

func initDocuments(ctx context.Context, db *sqlx.DB, embedLLM *client.LLM) error {
	data, err := os.ReadFile("zarf/data/book.chunks")
	if err != nil {
		return fmt.Errorf("read chunks file: %w", err)
	}

	re := regexp.MustCompile(`<CHUNK>([\s\S]*?)</CHUNK>`)
	matches := re.FindAllStringSubmatch(string(data), -1)

	chunks := make([]string, len(matches))
	for i, m := range matches {
		chunks[i] = strings.TrimSpace(m[1])
	}

	if len(chunks) == 0 {
		return fmt.Errorf("no chunks found in file")
	}

	firstEmbed, err := embedLLM.EmbedText(ctx, chunks[0])
	if err != nil {
		return fmt.Errorf("embed first chunk: %w", err)
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
		name TEXT NOT NULL,
		text TEXT NOT NULL,
		embedding VECTOR(%d) NOT NULL
	)`, dimensions)
	if err := sqldb.ExecContext(ctx, db, query); err != nil {
		return fmt.Errorf("create documents: %w", err)
	}

	const insertQ = `INSERT INTO documents (id, name, text, embedding) VALUES ($1, $2, $3, $4::vector)`
	if _, err := db.ExecContext(ctx, insertQ, 0, fmt.Sprintf("Chunk %d", 0), chunks[0], vector.FormatPGVector(firstEmbed)); err != nil {
		return fmt.Errorf("insert doc 0: %w", err)
	}

	for i := 1; i < len(chunks); i++ {
		emb, err := embedLLM.EmbedText(ctx, chunks[i])
		if err != nil {
			return fmt.Errorf("embed chunk %d: %w", i, err)
		}
		if _, err := db.ExecContext(ctx, insertQ, i, fmt.Sprintf("Chunk %d", i), chunks[i], vector.FormatPGVector(emb)); err != nil {
			return fmt.Errorf("insert doc %d: %w", i, err)
		}
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
