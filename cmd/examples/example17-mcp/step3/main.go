// This step wires the hardened streaming agent from example16 on top of
// the MCP server (step 1) and client discovery (step 2). Every tool call
// the model issues hops through the MCP server instead of executing
// locally, and safeMCPCall wraps each hop with panic recovery and a
// per-call timeout.
//
// What this step adds over step 2: the Agent type, the agent.go REPL,
// safeMCPCall + mcpClientCall plumbing on the client side, and the same
// hardened streaming loop the in-process tools used in example16.
//
// # Running the example
//
//	$ make example17-step3
//
// # Optional environment overrides
//
//  LLM_SERVER     chat completions endpoint (default: http://localhost:11435/v1/chat/completions)
//  LLM_MODEL      chat model name (default: Qwen3-8B-Q8_0)
//  EMBED_SERVER   embeddings endpoint (default: http://localhost:11435/v1/embeddings)
//  EMBED_MODEL    embedding model name (default: embeddinggemma-300m-qat-Q8_0)
//  MCP_HOST       MCP server host (default: localhost)
//  MCP_PORT       MCP server port (default: 8092)
//
// # Prerequisites
//
//  - make compose-up
//  - make kronk-up
//
// # Extra reading
//
//	https://github.com/modelcontextprotocol/go-sdk
//	https://github.com/modelcontextprotocol/go-sdk/blob/main/design/design.md

// Example 17 — Step 3 — Agent REPL Through MCP
package main

import (
	"bufio" // Step 03
	"context"
	"database/sql"
	"encoding/json" // Step 02
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
	"github.com/modelcontextprotocol/go-sdk/mcp" // Step 02
)

const (
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5432"
	dbName     = "postgres"

	toolTimeout = 30 * time.Second // Step 03
)

var (
	mcpHost = "localhost"
	mcpPort = "8092"

	llmURL     = "http://localhost:11435/v1/chat/completions"
	llmModel   = "Qwen3-8B-Q8_0"
	embedURL   = "http://localhost:11435/v1/embeddings"
	embedModel = "embeddinggemma-300m-qat-Q8_0"
)

// mcpDeps is set in run() before startMCPServer is launched. The MCP tool
// handlers in functions.go read from it because mcp.AddTool requires
// fixed-signature handlers.
var mcpDeps struct {
	db       *sqlx.DB
	chatLLM  *client.LLM
	embedLLM *client.LLM
}

func init() {
	if v := os.Getenv("MCP_HOST"); v != "" {
		mcpHost = v
	}

	if v := os.Getenv("MCP_PORT"); v != "" {
		mcpPort = v
	}

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

// Step 03
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

	setupDB, err := sqldb.Open(sqldb.Config{
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

	if err := sqldb.StatusCheck(ctx, setupDB); err != nil {
		setupDB.Close()
		return fmt.Errorf("status check: %w", err)
	}

	if err := initSQLDB(ctx, setupDB); err != nil {
		setupDB.Close()
		return fmt.Errorf("init sql db: %w", err)
	}

	if err := initDocuments(ctx, setupDB, embedClient); err != nil {
		setupDB.Close()
		return fmt.Errorf("init documents: %w", err)
	}

	setupDB.Close()

	// Reopen as read-only for runtime. tool_query_db will issue LLM-generated
	// SQL through this handle; default_transaction_read_only=on rejects writes
	// at the database layer. tool_search_book only reads pgvector data, so
	// using the same read-only handle is fine.
	db, err := sqldb.Open(sqldb.Config{
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
	defer db.Close()

	// endregion

	// -------------------------------------------------------------------------
	// Wire MCP handler dependencies and start the server.
	// region Wire MCP handler dependencies and start the server.

	mcpDeps.db = db
	mcpDeps.chatLLM = chatClient
	mcpDeps.embedLLM = embedClient

	go startMCPServer(mcpHost, mcpPort)

	// Give the server a moment to start.
	time.Sleep(500 * time.Millisecond)

	fmt.Printf("\nMCP server is running on %s:%s\n", mcpHost, mcpPort)
	fmt.Println("Registered tools:")
	fmt.Println("  - tool_search_book")
	fmt.Println("  - tool_query_db")
	fmt.Println("  - tool_get_reading_progress")

	// Step 02 -----------------------------------------------------------------
	// Discover tools via MCP and convert to the OpenAI tools-array shape.

	mcpTools, err := mcpListTools(ctx, mcpHost, mcpPort)
	if err != nil {
		return fmt.Errorf("list MCP tools: %w", err)
	}

	fmt.Printf("\nDiscovered %d MCP tools:\n", len(mcpTools))
	for _, t := range mcpTools {
		fmt.Printf("  - %s: %s\n", t.Name, t.Description)
	}

	toolDocuments, err := mcpToolsToClientD(mcpTools)
	if err != nil {
		return fmt.Errorf("convert MCP tools: %w", err)
	}

	fmt.Println("\nConverted to OpenAI tools-array shape:")
	for _, doc := range toolDocuments {
		out, _ := json.MarshalIndent(doc, "  ", "  ")
		fmt.Printf("  %s\n", string(out))
	}

	// Step 03 -----------------------------------------------------------------
	// Start the interactive agent.

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

	agent := NewAgent(getUserMessage, toolDocuments)

	if err := agent.Run(ctx); err != nil {
		return err
	}

	// endregion

	return nil
}

// Step 02 =====================================================================
// MCP discovery → OpenAI tools-array conversion.

// Step 02
// mcpToolsToClientD converts MCP tool descriptors into the OpenAI
// chat-completions tools-array shape so we don't have to duplicate the
// schemas client-side.
func mcpToolsToClientD(tools []*mcp.Tool) ([]client.D, error) {
	docs := make([]client.D, 0, len(tools))

	for _, t := range tools {
		params, err := normalizeInputSchema(t.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("tool %q: %w", t.Name, err)
		}

		docs = append(docs, client.D{
			"type": "function",
			"function": client.D{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  params,
			},
		})
	}

	return docs, nil
}

// Step 02
// normalizeInputSchema marshals/unmarshals the MCP schema back through JSON
// so the resulting value is a plain map[string]any the chat server accepts.
func normalizeInputSchema(schema any) (map[string]any, error) {
	if schema == nil {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}, nil
	}

	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("marshal input schema: %w", err)
	}

	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("unmarshal input schema: %w", err)
	}

	return out, nil
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
