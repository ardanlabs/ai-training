// This example takes the step1 semantic-cache demo and appends a threshold
// sensitivity sweep. After the agent run populates the cache, a fixed set of
// probe queries is embedded and looked up against the cache at multiple
// similarity thresholds. The resulting matrix makes the threshold trade-off
// visible: too high and a paraphrase misses (false negative); too low and an
// unrelated query falsely hits (false positive). The exact threshold to pick
// is a per-workload decision — this step exists to show there is no free
// lunch in choosing it.
//
// # Running the example
//
//	$ make example20-step2
//
// # What this step adds over step1
//
// A "Threshold Sensitivity" section after the cache-stats panel. No changes
// to the agent or cache code — the probes simply embed and re-query the
// already-populated query_cache table.
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

// Example 20 — Step 2 — Semantic Cache + Threshold Sensitivity
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
	// Open a second, read-only connection pool for the SQL tool. Every
	// connection in this pool starts with default_transaction_read_only=on, so
	// SQL the LLM generates cannot accidentally write to the database. The
	// writable db handle above is still used by SearchBook for its semantic
	// cache writes.
	// region Open a second, read-only connection pool for the SQL tool. Every

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
	// Demo: run three hardcoded questions through the agent.
	// region Demo: run three hardcoded questions through the agent.

	questions := []string{
		"What is a goroutine in Go?",
		"Can you explain goroutines in Go?",
		"How many highlights do I have in the concurrency chapter?",
	}

	questionIdx := 0
	getUserMessage := func() (string, bool) {
		if questionIdx >= len(questions) {
			return "", false
		}
		msg := questions[questionIdx]
		fmt.Println(msg)
		questionIdx++
		return msg, true
	}

	agent, err := NewAgent(getUserMessage, chatClient, embedClient, db, readOnlyDB)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	if err := agent.Run(ctx); err != nil {
		return fmt.Errorf("agent run: %w", err)
	}

	// endregion

	// -------------------------------------------------------------------------
	// Print cache statistics.
	// region Print cache statistics.

	fmt.Println("\n============================================================")
	fmt.Println("Cache Statistics")
	fmt.Println("============================================================")
	fmt.Printf("\n  Cache hits:            %d\n", agent.cacheHits)
	fmt.Printf("  Cache misses:          %d\n", agent.cacheMisses)

	total := agent.cacheHits + agent.cacheMisses
	if total > 0 {
		fmt.Printf("  Hit rate:              %.1f%%\n", float64(agent.cacheHits)/float64(total)*100)
	}
	if agent.cacheHits > 0 {
		fmt.Printf("  Avg hit similarity:    %.4f\n", agent.totalHitSimilarity/float64(agent.cacheHits))
	}
	fmt.Printf("  Vector searches saved: %d\n", agent.cacheHits)

	// endregion

	// -------------------------------------------------------------------------
	// Step 02
	// Threshold sensitivity sweep.
	//
	// The agent run above populated query_cache. Here we probe that cache with
	// a fixed set of queries chosen to exercise the failure modes of a single
	// cosine-similarity threshold: an exact-match, a paraphrase, a related
	// Go topic that should NOT hit, and a totally unrelated query. For each
	// probe we print the similarity to the nearest cached entry and whether
	// each candidate threshold would call it a HIT.
	// region Step 02

	if err := runThresholdSweep(ctx, db, embedClient); err != nil {
		return fmt.Errorf("threshold sweep: %w", err)
	}

	// endregion

	return nil
}

// =============================================================================
// Threshold sensitivity sweep
//
// Step 02

var sweepThresholds = []float64{0.50, 0.85, 0.92, 0.99}

var sweepProbes = []struct {
	label string
	query string
}{
	{"exact match", "What is a goroutine in Go?"},
	{"paraphrase", "Tell me about goroutines in Go"},
	{"related topic", "What's a channel in Go?"},
	{"unrelated", "What's the capital of France?"},
}

func runThresholdSweep(ctx context.Context, db *sqlx.DB, embedLLM *client.LLM) error {
	fmt.Println("\n============================================================")
	fmt.Println("Threshold Sensitivity")
	fmt.Println("============================================================")

	fmt.Printf("\n  %-14s  %-42s  %-8s", "Kind", "Probe", "Sim")
	for _, th := range sweepThresholds {
		fmt.Printf("  %-6s", fmt.Sprintf(">=%.2f", th))
	}
	fmt.Println()

	fmt.Printf("  %-14s  %-42s  %-8s", "----", "-----", "----")
	for range sweepThresholds {
		fmt.Printf("  %-6s", "------")
	}
	fmt.Println()

	for _, p := range sweepProbes {
		emb, err := embedLLM.EmbedText(ctx, p.query)
		if err != nil {
			return fmt.Errorf("embed probe %q: %w", p.query, err)
		}

		cached, err := searchCache(ctx, db, emb)
		if err != nil {
			return fmt.Errorf("search cache for %q: %w", p.query, err)
		}

		sim := 0.0
		if cached != nil {
			sim = cached.Similarity
		}

		fmt.Printf("  %-14s  %-42s  %-8.4f", p.label, fmt.Sprintf("%q", p.query), sim)
		for _, th := range sweepThresholds {
			mark := "  -"
			if sim >= th {
				mark = "  ✓"
			}
			fmt.Printf("  %-6s", mark)
		}
		fmt.Println()
	}

	fmt.Println("\nNo threshold is correct in isolation — it trades false positives")
	fmt.Println("(unrelated probe accepted) against false negatives (paraphrase")
	fmt.Println("rejected). Pick per workload, then monitor both error modes in")
	fmt.Printf("production. A single hardcoded constant like %.2f is a starting\n", cacheThreshold)
	fmt.Println("point, not an answer.")

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
// Documents and cache table seeding

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

	// Create query_cache table for semantic caching.
	if err := sqldb.ExecContext(ctx, db, `DROP TABLE IF EXISTS query_cache`); err != nil {
		return fmt.Errorf("drop query_cache: %w", err)
	}

	cacheQuery := fmt.Sprintf(`CREATE TABLE query_cache (
		id              SERIAL PRIMARY KEY,
		query_text      TEXT NOT NULL,
		query_embedding VECTOR(%d) NOT NULL,
		response        TEXT NOT NULL
	)`, dimensions)
	if err := sqldb.ExecContext(ctx, db, cacheQuery); err != nil {
		return fmt.Errorf("create query_cache: %w", err)
	}

	return nil
}

// =============================================================================
// Search and cache helpers (used transitively by the SearchBook tool)

type searchResult struct {
	ID         int
	Text       string
	Distance   float64
	Similarity float64
}

type cacheResult struct {
	Response   string
	Similarity float64
}

func searchDocumentsWithEmbedding(ctx context.Context, db *sqlx.DB, embedding []float64, topN int) ([]searchResult, error) {
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

func searchCache(ctx context.Context, db *sqlx.DB, embedding []float64) (*cacheResult, error) {
	const stmt = `
SELECT
	response,
	1 - (query_embedding <=> $1::vector) AS similarity
FROM
	query_cache
ORDER BY
	query_embedding <=> $1::vector
LIMIT 1
`

	rows, err := db.QueryContext(ctx, stmt, vector.FormatPGVector(embedding))
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	if !rows.Next() {
		return nil, nil
	}

	var r cacheResult
	if err := rows.Scan(&r.Response, &r.Similarity); err != nil {
		return nil, fmt.Errorf("scan: %w", err)
	}

	return &r, nil
}

func insertCache(ctx context.Context, db *sqlx.DB, query string, embedding []float64, response string) error {
	const stmt = `
INSERT INTO query_cache (query_text, query_embedding, response)
VALUES ($1, $2::vector, $3)
`

	_, err := db.ExecContext(ctx, stmt, query, vector.FormatPGVector(embedding), response)
	if err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	return nil
}
