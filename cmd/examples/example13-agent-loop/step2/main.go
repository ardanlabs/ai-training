// This step takes step1 and factors the tools behind a uniform shape.
// The fntools.Tool interface gives every tool the same Call method, and
// every Call returns the {status, data} envelope produced by
// fntools.SuccessResponse / fntools.ErrorResponse — so the agent loop
// can react to failures without parsing free-form text. The dispatch
// switch from step1 collapses into a single map[string]Tool lookup;
// adding a new tool is now a registration call instead of another case
// line. The agent loop itself is still written by hand inside run() —
// step3 wraps it in an Agent type.
//
// # Running the example
//
//	$ make example13-step2
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

// Example 13 — Step 2 — Tools Behind an Interface
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	fntools "github.com/ardanlabs/ai-training/foundation/tools"
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

// The strict classification rules below are a small-model accommodation,
// not part of the agent-loop lesson. Larger models can drive this with a
// much shorter prompt.
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
	cln := client.New(client.NoopLogger)

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
	// Tool registration — each Register* call inserts the tool into the
	// dispatch map and returns its JSON description for the model.
	// region Tool registration — each Register* call inserts the tool into the

	tools := make(map[string]fntools.Tool)
	toolDocs := []client.D{
		RegisterSearchBook(tools, embedClient, db),
		RegisterReadingProgress(tools),
	}

	// endregion

	// -------------------------------------------------------------------------
	// Conversation seed.
	// region Conversation seed.

	conversation := []client.D{
		{"role": "system", "content": systemPrompt},
	}

	fmt.Printf("\nChat with GoNotebook AI [%s] (use 'ctrl-c' to quit)\n", llmModel)
	fmt.Println("Try: 'How do interfaces work in Go?' or 'Where am I in the Ultimate Go Notebook?'")

	scanner := bufio.NewScanner(os.Stdin)

	// endregion

	// -------------------------------------------------------------------------
	// Outer REPL: read one user line, then run the agent loop until the
	// model produces a final answer with no tool calls.
	// region Outer REPL: read one user line, then run the agent loop until the

	for {
		fmt.Print("\n==============================\n")
		fmt.Print("-----\n")
		fmt.Print("> Prompt: ")

		if !scanner.Scan() {
			return nil
		}
		userInput := scanner.Text()
		switch strings.ToLower(strings.TrimSpace(userInput)) {
		case "", "quit", "/quit", "/exit", "/bye":
			return nil
		}

		fmt.Print("-----\n")

		conversation = append(conversation, client.D{
			"role":    "user",
			"content": userInput,
		})

		// ---------------------------------------------------------------------
		// The agent loop. Each iteration is one model turn.
		// region The agent loop. Each iteration is one model turn.

		for {
			choice, err := callModel(ctx, cln, conversation, toolDocs)
			if err != nil {
				return err
			}

			if len(choice.Message.ToolCalls) == 0 {
				fmt.Printf("> Response: %s\n", choice.Message.Content)
				fmt.Print("-----\n")

				conversation = append(conversation, client.D{
					"role":    "assistant",
					"content": choice.Message.Content,
				})
				break
			}

			conversation = append(conversation, assistantToolCallsMsg(choice.Message.ToolCalls))

			// Dispatch each tool call through the map lookup. The Tool
			// interface guarantees every Call returns a {status, data}
			// envelope ready to append to the conversation.
			for _, tc := range choice.Message.ToolCalls {
				fmt.Printf("\u001b[92m%s(%v)\u001b[0m:\n", tc.Function.Name, tc.Function.Arguments)

				var resp client.D
				if tool, ok := tools[tc.Function.Name]; ok {
					resp = tool.Call(ctx, tc)
				} else {
					resp = fntools.ErrorResponse(tc.ID, fmt.Errorf("unknown tool: %s", tc.Function.Name))
				}

				fmt.Printf("\u001b[90m  → %s\u001b[0m\n", fntools.Preview(resp))
				conversation = append(conversation, resp)
			}
		}

		// endregion
	}

	// endregion
}

// =============================================================================
// Support functions

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

// callModel sends the conversation to the chat completions endpoint and
// returns the first choice. Tool calls live on choice.Message.ToolCalls;
// the final answer lives on choice.Message.Content.
func callModel(ctx context.Context, cln *client.Client, conversation []client.D, toolDocs []client.D) (agentChoice, error) {
	d := client.D{
		"model":          llmModel,
		"messages":       conversation,
		"temperature":    0.1,
		"top_p":          0.1,
		"top_k":          1,
		"tools":          toolDocs,
		"tool_selection": "auto",
	}

	callCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	var chat agentChat
	if err := cln.Do(callCtx, http.MethodPost, llmURL, d, &chat); err != nil {
		return agentChoice{}, fmt.Errorf("call model: %w", err)
	}

	if len(chat.Choices) == 0 {
		return agentChoice{}, fmt.Errorf("no response from model")
	}

	return chat.Choices[0], nil
}

// assistantToolCallsMsg builds the assistant turn that echoes the model's
// own tool_calls back into the conversation. The arguments field on each
// tool call has to be a JSON-encoded string, not a map, per the OpenAI
// chat completions wire format.
func assistantToolCallsMsg(toolCalls []client.ToolCall) client.D {
	docs := make([]client.D, 0, len(toolCalls))
	for _, tc := range toolCalls {
		argsJSON, _ := json.Marshal(tc.Function.Arguments)
		docs = append(docs, client.D{
			"id":   tc.ID,
			"type": "function",
			"function": client.D{
				"name":      tc.Function.Name,
				"arguments": string(argsJSON),
			},
		})
	}

	return client.D{
		"role":       "assistant",
		"tool_calls": docs,
	}
}

// =============================================================================
// Response types for the chat completions endpoint.

type agentMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	ToolCalls []client.ToolCall `json:"tool_calls,omitempty"`
}

type agentChoice struct {
	Index        int          `json:"index"`
	Message      agentMessage `json:"message"`
	FinishReason string       `json:"finish_reason"`
}

type agentChat struct {
	ID      string        `json:"id"`
	Choices []agentChoice `json:"choices"`
}
