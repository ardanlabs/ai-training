// Package harness provides the plumbing every security scenario in example15
// shares: a document store to poison, a retrieval call to poison it through,
// and a model to aim the result at.
//
// The package deliberately holds no opinion about attacks or defenses. It knows
// how to seed a corpus, embed a question, return the nearest documents, run a
// single-shot classification, and - in agent.go - drive a tool-calling
// conversation. Everything that makes a scenario an attack - the payloads, the
// defenses, the tools, the scoring - lives in the packages that use this.
//
// That split is the point. A row of the attack matrix has to run the identical
// pipeline in every cell, with the active defenses as the only variable. If
// the pipeline lived in the step, each scenario would grow its own slightly
// different copy and the comparison would stop meaning anything.
package harness

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/jmoiron/sqlx"
)

// Access levels a seeded document can carry. Seed assigns these, Search can
// filter on them, and PrintDocs shows them, so a scenario can demonstrate that
// retrieval happily hands confidential text to a public conversation.
const (
	AccessPublic       = "public"
	AccessConfidential = "confidential"
)

// The pgvector database from `make compose-up`. It listens on 5433 to avoid the
// stock postgres on 5432, which does not carry the vector extension.
const (
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5433"
	dbName     = "postgres"
)

// Defaults match the Kronk server the workshop brings up with `make kronk-up`.
// Every one is overridable so a room with a different endpoint is not stuck.
var (
	llmURL     = "http://localhost:11435/v1/chat/completions"
	llmModel   = "Qwen3-8B-Q8_0"
	embedURL   = "http://localhost:11435/v1/embeddings"
	embedModel = "embeddinggemma-300m-qat-Q8_0"

	// The detection classifier's model. Empty means the answering model, which
	// init resolves after the LLM_MODEL override; CLASSIFIER_MODEL overrides it.
	//
	// A small guard was the default and measured NO on step 2's plain-English
	// payload, so it blocked nothing in any step. The 8B flags that payload and
	// is still walked past by the same request in Morse.
	classifierModel = ""
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

	if v := os.Getenv("CLASSIFIER_MODEL"); v != "" {
		classifierModel = v
	}

	if classifierModel == "" {
		classifierModel = llmModel
	}
}

// =============================================================================

// Document is one row returned by Search: the stored text plus what retrieval
// knows about it. AccessLevel is carried through deliberately - the whole
// exfiltration story is that a confidential document reaches the model context
// with its label attached and nothing acts on the label.
type Document struct {
	ID          int
	Text        string
	AccessLevel string
	Similarity  float64
}

// Harness owns the connections a scenario needs. Construct it with New, then
// call EnsureDB before the first Seed or Search.
type Harness struct {
	classifyLLM *client.LLM
	embedLLM    *client.LLM
	db          *sqlx.DB
}

// New builds a Harness with the configured endpoints. No network or database
// work happens here, so a scenario that never retrieves anything does not need a
// database running at all.
func New() *Harness {
	return &Harness{
		classifyLLM: client.NewLLM(llmURL, classifierModel),
		embedLLM:    client.NewLLM(embedURL, embedModel),
	}
}

// Config prints the endpoints in use, so a run that behaves oddly can be
// traced to the model that produced it.
func (h *Harness) Config() {
	fmt.Printf("\nChat:       %s (%s)\n", llmURL, llmModel)
	fmt.Printf("Classifier: %s (%s)\n", llmURL, classifierModel)
	fmt.Printf("Embed:      %s (%s)\n", embedURL, embedModel)
}

// Close releases the database connection if one was opened.
func (h *Harness) Close() error {
	if h.db == nil {
		return nil
	}

	return h.db.Close()
}

// EnsureDB opens the database on first call and is a no-op afterwards. Every
// scenario calls it before seeding, which lets a matrix of scenarios run back
// to back over a single pool instead of reconnecting per cell.
func (h *Harness) EnsureDB(ctx context.Context) error {
	if h.db != nil {
		return nil
	}

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

	if err := sqldb.StatusCheck(ctx, db); err != nil {
		db.Close()
		return fmt.Errorf("db status check: %w", err)
	}

	h.db = db

	return nil
}

// =============================================================================

// Seed drops and recreates the document table, then embeds and inserts each
// document. Dropping is what makes a matrix reproducible: a scenario that
// poisons the store cannot leak that poison into the next scenario's run.
//
// The access level is derived from the text rather than passed in, so the
// corpus in the step file stays a plain list of strings and there is exactly
// one place that decides what counts as confidential.
func (h *Harness) Seed(ctx context.Context, docs []string) error {
	if len(docs) == 0 {
		return fmt.Errorf("seed: no documents")
	}

	if err := h.EnsureDB(ctx); err != nil {
		return err
	}

	// The first embedding determines the column width. Asking the model rather
	// than hardcoding a number keeps the table correct across embed models.
	firstEmbed, err := h.embedLLM.EmbedText(ctx, docs[0])
	if err != nil {
		return fmt.Errorf("embed first: %w", err)
	}

	if err := sqldb.ExecContext(ctx, h.db, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return fmt.Errorf("create extension vector: %w", err)
	}

	if err := sqldb.ExecContext(ctx, h.db, `DROP TABLE IF EXISTS injection_docs`); err != nil {
		return fmt.Errorf("drop table: %w", err)
	}

	q := fmt.Sprintf(`CREATE TABLE injection_docs (
		id BIGINT PRIMARY KEY,
		text TEXT NOT NULL,
		access_level TEXT NOT NULL,
		embedding VECTOR(%d) NOT NULL
	)`, len(firstEmbed))

	if err := sqldb.ExecContext(ctx, h.db, q); err != nil {
		return fmt.Errorf("create table: %w", err)
	}

	const ins = `INSERT INTO injection_docs (id, text, access_level, embedding) VALUES ($1, $2, $3, $4::vector)`

	for i, doc := range docs {
		embedding := firstEmbed

		if i > 0 {
			embedding, err = h.embedLLM.EmbedText(ctx, doc)
			if err != nil {
				return fmt.Errorf("embed doc %d: %w", i, err)
			}
		}

		if _, err := h.db.ExecContext(ctx, ins, i, doc, accessLevel(doc), vector.FormatPGVector(embedding)); err != nil {
			return fmt.Errorf("insert doc %d: %w", i, err)
		}
	}

	return nil
}

// accessLevel classifies a document by how its text opens. Real stores carry
// this as metadata from the system that produced the row; deriving it here
// keeps the corpus in the attacks package a plain list of strings.
//
// The check is a prefix rather than a substring, and that is not fussiness.
// The poisoned document talks *about* confidential records - its fake audit
// notice instructs the model to POST "every CONFIDENTIAL record". A substring
// match would file the attacker's own document as confidential and quietly
// filter it out alongside the real one, making an access-level defense look
// like it stopped the injection when all it did was hide the payload.
//
// A classifier that catches the attack for the wrong reason is worse than one
// that misses it, because it survives review.
func accessLevel(doc string) string {
	if strings.HasPrefix(strings.TrimSpace(strings.ToUpper(doc)), "CONFIDENTIAL") {
		return AccessConfidential
	}

	return AccessPublic
}

// Search embeds the query and returns the topK nearest documents by cosine
// similarity.
//
// The two filters are the levers a defense step reaches for, and both are off
// in the attack steps, which pass ("", 0):
//
//   - access limits results to one access level. This is the defense that
//     actually works, because it runs on document metadata rather than on the
//     attacker's text, so no encoding trick reaches it.
//   - minSimilarity drops weak matches. This is the defense that looks
//     reasonable and is not one: a poisoned document opens with on-topic bait
//     precisely so it scores well.
func (h *Harness) Search(ctx context.Context, query string, topK int, access string, minSimilarity float64) ([]Document, error) {
	if err := h.EnsureDB(ctx); err != nil {
		return nil, err
	}

	embedding, err := h.embedLLM.EmbedText(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	// The access filter is applied in SQL so it constrains what topK selects
	// from. Filtering after LIMIT would let confidential rows consume slots and
	// silently shrink the result, which is a different behavior than "this
	// conversation cannot see those documents at all".
	const q = `
		SELECT id, text, access_level, 1 - (embedding <=> $1::vector) AS similarity
		FROM injection_docs
		WHERE ($2 = '' OR access_level = $2)
		ORDER BY embedding <=> $1::vector
		LIMIT $3`

	rows, err := h.db.QueryContext(ctx, q, vector.FormatPGVector(embedding), access, topK)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	var docs []Document

	for rows.Next() {
		var d Document
		if err := rows.Scan(&d.ID, &d.Text, &d.AccessLevel, &d.Similarity); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		// Applied here rather than in SQL: the threshold is a property of the
		// scores that came back, and keeping it out of the WHERE clause means
		// the ORDER BY still uses the vector index.
		if d.Similarity < minSimilarity {
			continue
		}

		docs = append(docs, d)
	}

	return docs, rows.Err()
}

// =============================================================================

// Classify runs a single-shot prompt through the chat model and returns the
// trimmed answer with any reasoning block removed. The detection classifier in
// a step is built on this.
func (h *Harness) Classify(ctx context.Context, prompt string) (string, error) {
	answer, err := h.classifyLLM.ChatCompletions(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("classify: %w", err)
	}

	// Qwen3 may emit a <think>...</think> block before its answer even when
	// asked not to. Only the text after it is the verdict.
	if idx := strings.LastIndex(answer, "</think>"); idx != -1 {
		answer = answer[idx+len("</think>"):]
	}

	return strings.TrimSpace(answer), nil
}
