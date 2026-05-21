// This step takes step2 and factors the loop itself behind an Agent
// type. The for loop, conversation buffer, model call, and tool dispatch
// that lived in run() in steps 1 and 2 are now methods on Agent in
// agent.go; run() shrinks to NewAgent + agent.Run(ctx). The tool
// implementations and the {status, data} envelope from step2 are
// unchanged. This is the shape every following example (14, 15, 16, 17)
// carries forward.
//
// The two tools registered here are:
//
//   - tool_search_book: pgvector similarity search over the ingested book
//     (replaces the always-on RAG retrieval from example12 — the model now
//     decides when to retrieve).
//   - tool_get_reading_progress: stub returning fixed reading-progress data.
//
// Streaming, reasoning rendering, and usage panels arrive in example14.
//
// # Running the example
//
//	$ make example13-step3
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

// Example 13 — Step 3 — Agent Type
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"

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

const systemPrompt = `You are a strict assistant for the Ultimate Go Notebook. You have access to two tools:

- tool_search_book: search the Ultimate Go Notebook for information about Go programming concepts.
- tool_get_reading_progress: get the user's current reading progress (chapter, page, percent complete).

IMPORTANT: Whenever the user asks about "a book", "the book", "this book", an author, or otherwise mentions a book in any form, they ALWAYS mean the Ultimate Go Notebook. Treat any such mention as a question about the Ultimate Go Notebook and route it to tool_search_book.

NOTE: The Ultimate Go Notebook is written by Bill Kennedy. When the user mentions "Bill", "Bill Kennedy", "Kennedy", or "the author", they are referring to the author of the book. Do NOT include the author's name in tool_search_book queries — search for the topic only (e.g., for "What does Bill say about pointers?" call tool_search_book with query "pointers", not "bill pointers").

CURRENT USER

- user_id: user_gopher

MANDATORY WORKFLOW

Step 1 — Classify the user message into exactly one of:
  (a) A question about Go programming or about the book itself (any Go topic: interfaces, goroutines, channels, slices, maps, errors, generics, packages, modules, testing, performance, etc.; OR anything about the book: author, title, chapters, contents, who wrote it, what it covers, etc.) → go to Step 2A.
  (b) A question about the user's reading progress → go to Step 2B.
  (c) Pure greeting or chit-chat with no Go or reading content (e.g. "hi", "thanks") → go to Step 3.

Step 2A — You MUST call tool_search_book before producing any visible text. You are NOT allowed to refuse, answer, or claim out-of-scope until you have called the tool and seen its result. After the tool returns:
  - If matches > 0, write your final answer using ONLY the text inside the returned "context". Do not add code, examples, definitions, or commentary that are not literally present in "context".
  - If matches == 0 (or data says "No relevant passages found"), respond with exactly: "That topic is out of scope for the Ultimate Go Notebook." and nothing else.

Step 2B — You MUST call tool_get_reading_progress with user_id set to the CURRENT USER id shown above, then answer using only the returned data.

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
	ctx := context.Background()

	embedClient := client.NewLLM(embedURL, embedModel)

	// -------------------------------------------------------------------------
	// Connect to PostgreSQL.
	// region Connect to PostgreSQL.

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

	if err := initDocuments(ctx, db, embedClient); err != nil {
		return fmt.Errorf("init documents: %w", err)
	}

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
		}
		return message, true
	}

	agent, err := NewAgent(getUserMessage, embedClient, db)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	// endregion

	return agent.Run(ctx)
}

// =============================================================================
// Documents ingestion

func initDocuments(ctx context.Context, db *sqlx.DB, embedLLM *client.LLM) error {
	var count int
	if err := db.GetContext(ctx, &count, "SELECT count(*) FROM documents"); err == nil && count > 0 {
		fmt.Printf("documents table already seeded (%d rows), skipping ingestion\n", count)
		return nil
	}

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
