// Everything that talks to the database. One table, no edges: id, chapter,
// text, embedding. Chapter is one per chunk so a column answers it, and page
// order is arithmetic: the chunk after 93 is 94. Neither needs a graph. The
// relationships that do are many-to-many and discovered, like shared subject.

package main

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/jmoiron/sqlx"
)

// The database `make compose-up` starts. This step uses none of its graph
// features, but shares the instance with the later steps so they reuse the
// embeddings this step pays for.
const (
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5433"
	dbName     = "postgres"
)

// document is one chunk on its way into the prompt, plus its similarity rank.
type document struct {
	ID         int
	Chapter    string
	Text       string
	Similarity float64
	Rank       int
}

// neighbor is a chunk immediately before or after a hit in the book, with the
// rank similarity gave it. A rank outside topK means the model never sees it.
type neighbor struct {
	HitID int
	ID    int
	Rank  int
	After bool
}

// searchResult is everything one question produced.
type searchResult struct {
	Docs      []document
	Neighbors []neighbor
	Corpus    int
}

// =============================================================================

// openDB connects to postgres and makes sure pgvector is available.
func openDB(ctx context.Context) (*sqlx.DB, error) {
	db, err := sqldb.Open(sqldb.Config{
		User:         dbUser,
		Password:     dbPassword,
		Host:         dbHost,
		Name:         dbName,
		MaxIdleConns: 2,
		MaxOpenConns: 5,
		DisableTLS:   true,
	})
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	if err := sqldb.StatusCheck(ctx, db); err != nil {
		db.Close()
		return nil, fmt.Errorf("status check: %w", err)
	}

	if _, err := db.ExecContext(ctx, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		db.Close()
		return nil, fmt.Errorf("create extension vector: %w", err)
	}

	fmt.Printf("- connected to postgres on %s\n", dbHost)

	return db, nil
}

// =============================================================================

// loadData ingests the chunks file on the first run and does nothing after
// that, since embedding the book is a model call per chunk. Later steps share
// the table, so whichever step runs first pays for the embeddings.
func loadData(ctx context.Context, db *sqlx.DB, krnEmbed *kronk.Kronk, chunksFile string) error {
	create := fmt.Sprintf(`
	CREATE TABLE IF NOT EXISTS chunks (
		id        INT PRIMARY KEY,
		text      TEXT NOT NULL,
		embedding VECTOR(%d) NOT NULL
	)`, dimensions)

	if _, err := db.ExecContext(ctx, create); err != nil {
		return fmt.Errorf("create table: %w", err)
	}

	// Step2 and step3 keep the chapter in the graph and create this table without
	// the column, so add it in case one of them ingested the book first.
	const chapter = `ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chapter TEXT NOT NULL DEFAULT ''`
	if _, err := db.ExecContext(ctx, chapter); err != nil {
		return fmt.Errorf("add chapter column: %w", err)
	}

	chunks, err := parseChunks(chunksFile)
	if err != nil {
		return fmt.Errorf("parse chunks: %w", err)
	}

	fmt.Printf("- parsed %d chunks\n", len(chunks))

	var rowCount int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks`).Scan(&rowCount); err != nil {
		return fmt.Errorf("count chunks: %w", err)
	}

	if rowCount > 0 {
		fmt.Printf("- table exists with %d rows\n", rowCount)

		// Chapters are cheap, so backfill instead of dropping the table.
		return backfillChapters(ctx, db, chunks)
	}

	// -------------------------------------------------------------------------

	fmt.Print("\n")
	fmt.Print("\033[s")

	t := time.Now()

	for i, c := range chunks {
		fmt.Print("\033[u\033[K")
		fmt.Printf("Vectorizing: %d of %d", i+1, len(chunks))

		embedding, err := embed(ctx, krnEmbed, c.text)
		if err != nil {
			return fmt.Errorf("embed chunk %d: %w", c.id, err)
		}

		const insert = `
		INSERT INTO chunks (id, chapter, text, embedding)
		VALUES ($1, $2, $3, $4::vector)`

		if _, err := db.ExecContext(ctx, insert, c.id, c.chapter, c.text, vector.FormatPGVector(embedding)); err != nil {
			return fmt.Errorf("insert chunk %d: %w", c.id, err)
		}
	}

	fmt.Printf("\nVectorized in %v\n", time.Since(t))

	// -------------------------------------------------------------------------

	const index = `
	CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
	USING hnsw (embedding vector_cosine_ops)`

	if _, err := db.ExecContext(ctx, index); err != nil {
		return fmt.Errorf("create hnsw index: %w", err)
	}

	return nil
}

// backfillChapters fills the chapter column for rows another step inserted.
func backfillChapters(ctx context.Context, db *sqlx.DB, chunks []chunk) error {
	var missing int
	const count = `SELECT COUNT(*) FROM chunks WHERE chapter = ''`
	if err := db.QueryRowContext(ctx, count).Scan(&missing); err != nil {
		return fmt.Errorf("count missing chapters: %w", err)
	}

	if missing == 0 {
		return nil
	}

	for _, c := range chunks {
		const update = `UPDATE chunks SET chapter = $1 WHERE id = $2 AND chapter = ''`
		if _, err := db.ExecContext(ctx, update, c.chapter, c.id); err != nil {
			return fmt.Errorf("update chapter %d: %w", c.id, err)
		}
	}

	fmt.Printf("- filled in the chapter for %d rows\n", missing)

	return nil
}

// =============================================================================

// search is the whole retrieval strategy of an ordinary RAG application: embed
// the question, take the topK nearest chunks, stop. The neighbor lookup that
// follows never reaches the prompt; it measures what the strategy left behind.
func search(ctx context.Context, db *sqlx.DB, krnEmbed *kronk.Kronk, question string, topK int) (searchResult, error) {
	embedding, err := embed(ctx, krnEmbed, question)
	if err != nil {
		return searchResult{}, fmt.Errorf("embed question: %w", err)
	}

	corpus, err := corpusSize(ctx, db)
	if err != nil {
		return searchResult{}, fmt.Errorf("corpus size: %w", err)
	}

	docs, err := topKSearch(ctx, db, embedding, topK)
	if err != nil {
		return searchResult{}, fmt.Errorf("vector search: %w", err)
	}

	if len(docs) == 0 {
		return searchResult{Corpus: corpus}, nil
	}

	// -------------------------------------------------------------------------
	// Page order is arithmetic here, so ask where similarity ranked the chunk
	// before and after every hit.

	want := make([]int, 0, len(docs)*2)
	for _, d := range docs {
		want = append(want, d.ID-1, d.ID+1)
	}

	ranks, err := ranksOf(ctx, db, embedding, want)
	if err != nil {
		return searchResult{}, fmt.Errorf("neighbor ranks: %w", err)
	}

	neighbors := make([]neighbor, 0, len(docs)*2)
	for _, d := range docs {
		for _, id := range []int{d.ID - 1, d.ID + 1} {
			rank, ok := ranks[id]
			if !ok {
				// The first and last chunk of the book have one neighbor.
				continue
			}

			neighbors = append(neighbors, neighbor{
				HitID: d.ID,
				ID:    id,
				Rank:  rank,
				After: id > d.ID,
			})
		}
	}

	return searchResult{Docs: docs, Neighbors: neighbors, Corpus: corpus}, nil
}

// corpusSize reports the number of chunks a rank has to be read against.
func corpusSize(ctx context.Context, db *sqlx.DB) (int, error) {
	var n int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks`).Scan(&n); err != nil {
		return 0, fmt.Errorf("count: %w", err)
	}

	return n, nil
}

// topKSearch returns the topK chunks closest to the query, by cosine distance.
func topKSearch(ctx context.Context, db *sqlx.DB, embedding []float64, topK int) ([]document, error) {
	const q = `
	SELECT
		id,
		chapter,
		text,
		1 - (embedding <=> $1::vector) AS similarity,
		ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
	FROM chunks
	ORDER BY embedding <=> $1::vector
	LIMIT $2`

	rows, err := db.QueryContext(ctx, q, vector.FormatPGVector(embedding), topK)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	var docs []document

	for rows.Next() {
		var d document
		if err := rows.Scan(&d.ID, &d.Chapter, &d.Text, &d.Similarity, &d.Rank); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		docs = append(docs, d)
	}

	return docs, rows.Err()
}

// ranksOf reports where the given chunks placed in the similarity ordering. It
// ranks the whole table, so it cannot use the index. Fine at 180 chunks, not
// something to copy into a hot path.
func ranksOf(ctx context.Context, db *sqlx.DB, embedding []float64, ids []int) (map[int]int, error) {
	const base = `
	WITH scored AS (
		SELECT
			id,
			ROW_NUMBER() OVER (ORDER BY embedding <=> ?::vector) AS rank
		FROM chunks
	)
	SELECT id, rank FROM scored WHERE id IN (?)`

	q, args, err := sqlx.In(base, vector.FormatPGVector(embedding), ids)
	if err != nil {
		return nil, fmt.Errorf("build in clause: %w", err)
	}

	rows, err := db.QueryContext(ctx, db.Rebind(q), args...)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	out := make(map[int]int, len(ids))

	for rows.Next() {
		var id, rank int
		if err := rows.Scan(&id, &rank); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		out[id] = rank
	}

	return out, rows.Err()
}

// =============================================================================

// chunk is one parsed <CHUNK> block and the chapter it belongs to.
type chunk struct {
	id      int
	chapter string
	text    string
}

var (
	chunkRE = regexp.MustCompile(`<CHUNK>[\w\W]*?</CHUNK>`)
	tocRE   = regexp.MustCompile(`Chapter (\d+): ([^.]+?)\.{2,}`)

	// The three signals that say which chapter a chunk belongs to, strongest
	// first. A "Chapter N:" heading is exact but rare; half the chunks open with
	// a section number instead, and "6.1 Scheduler Semantics" says chapter 6.
	sectionTopRE = regexp.MustCompile(`^(\d{1,2})\.\d{1,2} [A-Z]`)
	chapterRE    = regexp.MustCompile(`Chapter (\d+):`)
	sectionRE    = regexp.MustCompile(`\b(\d{1,2})\.\d{1,2} [A-Z]`)
)

// chapterOf returns the chapter number a chunk announces, or "" if it announces
// none and so continues the chunk before it. The capital letter in the section
// patterns keeps "1.5 million" and "Listing 9.51 import (" from reading as
// headings, which would file a third of the book under the wrong chapter.
func chapterOf(text string) string {
	if m := sectionTopRE.FindStringSubmatch(text); m != nil {
		return m[1]
	}

	// Later matches win: a chunk crossing a boundary belongs to its tail chapter.
	if m := chapterRE.FindAllStringSubmatch(text, -1); m != nil {
		return m[len(m)-1][1]
	}

	if m := sectionRE.FindStringSubmatch(text); m != nil {
		return m[1]
	}

	return ""
}

// parseChunks reads the chunks file and works out which chapter each chunk
// belongs to. Chapter titles come from the table of contents because in the
// body a heading runs into the first sentence with no punctuation to stop at,
// while in the contents a run of dots follows it.
func parseChunks(chunksFile string) ([]chunk, error) {
	data, err := os.ReadFile(chunksFile)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	blocks := chunkRE.FindAllString(string(data), -1)
	if len(blocks) == 0 {
		return nil, fmt.Errorf("no chunks found in %s", chunksFile)
	}

	titles := make(map[string]string)
	for _, m := range tocRE.FindAllStringSubmatch(string(data), -1) {
		titles[m[1]] = fmt.Sprintf("Chapter %s: %s", m[1], strings.TrimSpace(m[2]))
	}

	chunks := make([]chunk, 0, len(blocks))
	current := "Front Matter"

	for i, block := range blocks {
		text := strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(block, "<CHUNK>"), "</CHUNK>"))

		// The table of contents names every chapter in the book, so letting it set
		// the current chapter would file it under the last one.
		if !strings.Contains(text, "Table of Contents") {
			if n := chapterOf(text); n != "" {
				// The table of contents only lists chapters 1 to 10.
				current = cmp.Or(titles[n], "Chapter "+n)
			}
		}

		chunks = append(chunks, chunk{
			id:      i,
			chapter: current,
			text:    text,
		})
	}

	return chunks, nil
}

// =============================================================================

// embed turns text into a vector with the embedding model.
func embed(ctx context.Context, krn *kronk.Kronk, text string) ([]float64, error) {
	d := model.D{
		"input":              text,
		"truncate":           true,
		"truncate_direction": "right",
	}

	resp, err := krn.Embeddings(ctx, d)
	if err != nil {
		return nil, fmt.Errorf("embeddings: %w", err)
	}

	if len(resp.Data) == 0 || len(resp.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("empty vector")
	}

	// The model returns float32; the pgvector helper in this repo wants float64.
	out := make([]float64, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		out[i] = float64(v)
	}

	return out, nil
}
