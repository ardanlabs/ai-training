// This example takes step1 and sends the model a single user question along
// with a `tools` array containing a `tool_search_book` function definition.
// When the model decides it wants to call the tool, the response comes back
// with finish_reason "tool_calls" and a structured arguments payload. This
// step prints the tool name and arguments WITHOUT executing the tool — step3
// closes the loop by running it and feeding the result back.
//
// # Running the example
//
//	$ make example12-step2
//
// # Optional environment overrides
//
//	LLM_SERVER     chat completions endpoint (default: http://localhost:11435/v1/chat/completions)
//	LLM_MODEL      chat model name (default: Qwen3-8B-Q8_0)
//	EMBED_SERVER   embeddings endpoint (default: http://localhost:11435/v1/embeddings)
//	EMBED_MODEL    embedding model name (default: embeddinggemma-300m-qat-Q8_0)
//
// # Prerequisites
//
//   - make compose-up
//   - make kronk-up
//
// # What this step adds over step1
//
// Tool definition (`tool_search_book`) plus phase 1 of the two-phase tool
// calling protocol: send tools, observe the tool_calls response, print the
// name and arguments. No execution yet.

// Example 12 — Step 2 — Tool Definition + Signal Phase
package main

import (
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
	"github.com/ardanlabs/ai-training/foundation/rag"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/jmoiron/sqlx"
	"golang.org/x/sync/errgroup"
)

const (
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5432"
	dbName     = "postgres"
)

// Step 02
var (
	llmURL     = "http://localhost:11435/v1/chat/completions"
	llmModel   = "Qwen3-8B-Q8_0"
	embedURL   = "http://localhost:11435/v1/embeddings"
	embedModel = "embeddinggemma-300m-qat-Q8_0"
)

func init() {
	// Step 02
	if v := os.Getenv("LLM_SERVER"); v != "" {
		llmURL = v
	}

	// Step 02
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

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

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

	// endregion

	// -------------------------------------------------------------------------
	// 1) Parse book chunks.
	// region 1) Parse book chunks.

	fmt.Print("\n============================================================\n")
	fmt.Print("1) Parse Book Chunks\n")
	fmt.Print("============================================================\n")

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

	fmt.Printf("\nTotal chunks: %d\n", len(chunks))

	firstEmbedding, err := embedClient.EmbedText(ctx, chunks[0])
	if err != nil {
		return fmt.Errorf("embed first chunk: %w", err)
	}

	dimensions := len(firstEmbedding)
	fmt.Printf("\nEmbedding dimensions: %d\n", dimensions)

	// endregion

	// -------------------------------------------------------------------------
	// 2) Parallel embedding with errgroup (carried over from example11).
	// region 2) Parallel embedding with errgroup (carried over from example11).

	fmt.Print("\n============================================================\n")
	fmt.Print("2) Parallel Embedding (errgroup, concurrency=5)\n")
	fmt.Print("============================================================\n")

	const concurrency = 5

	parStart := time.Now()

	parDocs := make([]rag.Document, len(chunks))
	parDocs[0] = rag.Document{ID: 0, Name: fmt.Sprintf("Chunk %d", 0), Text: chunks[0], Embedding: firstEmbedding}

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(concurrency)

	for i := 1; i < len(chunks); i++ {
		g.Go(func() error {
			embedding, err := embedClient.EmbedText(gCtx, chunks[i])
			if err != nil {
				return fmt.Errorf("parallel embed chunk %d: %w", i, err)
			}
			parDocs[i] = rag.Document{ID: i, Name: fmt.Sprintf("Chunk %d", i), Text: chunks[i], Embedding: embedding}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	parDuration := time.Since(parStart)
	fmt.Printf("\nParallel (%d workers): %d chunks in %s\n", concurrency, len(chunks), parDuration)

	// endregion

	// -------------------------------------------------------------------------
	// 3) Store the parallel results in pgvector.
	// region 3) Store the parallel results in pgvector.

	fmt.Print("\n============================================================\n")
	fmt.Print("3) Store in pgvector\n")
	fmt.Print("============================================================\n")

	if err := initDB(ctx, db, dimensions); err != nil {
		return err
	}

	if err := insertDocuments(ctx, db, parDocs); err != nil {
		return err
	}

	fmt.Printf("\n%d documents stored.\n", len(parDocs))

	// endregion

	// -------------------------------------------------------------------------
	// Step 02
	// 4) Tool calling demo — phase 1 (signal only, no execution).
	// region Step 02

	fmt.Print("\n============================================================\n")
	fmt.Print("4) Tool Calling Demo — Phase 1 (Signal Only)\n")
	fmt.Print("============================================================\n")

	cln := client.New(client.NoopLogger)

	const userPrompt = "What does the Ultimate Go Notebook say about interfaces?"

	fmt.Print("\n==============================\n")
	fmt.Print("-----\n")
	fmt.Printf("> Prompt: %s\n", userPrompt)
	fmt.Print("-----\n")

	messages := []client.D{
		{"role": "system", "content": "You are a helpful assistant for the Ultimate Go Notebook with access to a tool_search_book function. Whenever the user asks about \"a book\", \"the book\", an author, or otherwise mentions a book in any form, they ALWAYS mean the Ultimate Go Notebook."},
		{"role": "user", "content": userPrompt},
	}

	d := client.D{
		"model":          llmModel,
		"messages":       messages,
		"temperature":    0.1,
		"top_p":          0.1,
		"top_k":          1,
		"tools":          []client.D{searchBookToolDef()},
		"tool_selection": "auto",
	}

	var chat toolChat
	if err := cln.Do(ctx, http.MethodPost, llmURL, d, &chat); err != nil {
		return fmt.Errorf("phase 1 chat completions: %w", err)
	}

	if len(chat.Choices) == 0 {
		return fmt.Errorf("no response from model")
	}

	choice := chat.Choices[0]

	if choice.FinishReason == "tool_calls" && len(choice.Message.ToolCalls) > 0 {
		tc := choice.Message.ToolCalls[0]
		argsJSON, _ := json.MarshalIndent(tc.Function.Arguments, "", "  ")

		fmt.Printf("\n  Model requested tool call (NOT executed in this step):\n")
		fmt.Printf("    Tool:      %s\n", tc.Function.Name)
		fmt.Printf("    Arguments: %s\n", argsJSON)
		fmt.Printf("    Call ID:   %s\n", tc.ID)
		fmt.Print("-----\n")
	} else {
		fmt.Printf("\n  Model responded directly (no tool call requested):\n  %s\n", choice.Message.Content)
		fmt.Print("-----\n")
	}

	// endregion

	return nil
}

// =============================================================================

func initDB(ctx context.Context, db *sqlx.DB, dimensions int) error {
	if err := sqldb.ExecContext(ctx, db, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return fmt.Errorf("create extension vector: %w", err)
	}

	if err := sqldb.ExecContext(ctx, db, `DROP TABLE IF EXISTS documents`); err != nil {
		return fmt.Errorf("drop table: %w", err)
	}

	query := fmt.Sprintf(`
CREATE TABLE documents (
	id        BIGINT PRIMARY KEY,
	name      TEXT NOT NULL,
	text      TEXT NOT NULL,
	embedding VECTOR(%d) NOT NULL
)`, dimensions)

	if err := sqldb.ExecContext(ctx, db, query); err != nil {
		return fmt.Errorf("create table: %w", err)
	}

	fmt.Print("\nTable 'documents' created.\n")

	return nil
}

func insertDocuments(ctx context.Context, db *sqlx.DB, documents []rag.Document) error {
	const query = `
INSERT INTO documents (id, name, text, embedding)
VALUES ($1, $2, $3, $4::vector)
`

	for _, doc := range documents {
		if _, err := db.ExecContext(ctx, query, doc.ID, doc.Name, doc.Text, vector.FormatPGVector(doc.Embedding)); err != nil {
			return fmt.Errorf("insert document %d: %w", doc.ID, err)
		}
	}

	return nil
}

// =============================================================================
// Step 02
// Tool definition & response types.

// Step 02
func searchBookToolDef() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        "tool_search_book",
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

// Step 02
type toolFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// Step 02
func (f *toolFunction) UnmarshalJSON(b []byte) error {
	var tmp struct {
		Name         string `json:"name"`
		RawArguments string `json:"arguments"`
	}

	if err := json.Unmarshal(b, &tmp); err != nil {
		return err
	}

	arguments := make(map[string]any)
	if err := json.Unmarshal([]byte(tmp.RawArguments), &arguments); err != nil {
		return err
	}

	*f = toolFunction{
		Name:      tmp.Name,
		Arguments: arguments,
	}

	return nil
}

// Step 02
type toolCallMsg struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

// Step 02
type toolChatMessage struct {
	Role      string        `json:"role"`
	Content   string        `json:"content"`
	ToolCalls []toolCallMsg `json:"tool_calls,omitempty"`
}

// Step 02
type toolChatChoice struct {
	Index        int             `json:"index"`
	Message      toolChatMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// Step 02
type toolChat struct {
	ID      string           `json:"id"`
	Choices []toolChatChoice `json:"choices"`
}
