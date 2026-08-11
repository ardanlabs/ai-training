// Everything that talks to the database. Chunk text and its embedding live in
// a pgvector table, the structure of the book beside it as an Apache AGE
// graph. The graph stores keys and the table stores content, so traversals
// return ids and text is fetched with a parameterized query. That keeps book
// text out of hand-built Cypher and the graph small.
//
// Embedding and extraction each take a minute or two, so both record progress
// per chunk. A ctrl-c costs one chunk, not the run.
//
// Connecting, running Cypher and reading agtype live in foundation/age.

package main

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/age"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/jmoiron/sqlx"
)

// Connection settings for the Apache AGE database started by
// `make compose-up`. It listens on 5433 to avoid the stock postgres.
const (
	dbUser     = "postgres"
	dbPassword = "postgres"
	dbHost     = "localhost:5433"
	dbName     = "postgres"
	graphName  = "book"
)

// document is one chunk on its way into the prompt.
type document struct {
	ID      int
	Chapter string
	Text    string

	// Similarity is the cosine score step4 ranked on. Here it only picks the
	// shortlist, and is kept so the table can print it beside the reranker's.
	Similarity float64
	VectorRank int

	// Hop is how retrieval reached this chunk, Relevance what the reranker made
	// of it, Score that relevance after the hop prior is applied.
	Hop       string
	Relevance float64
	Score     float64
}

// =============================================================================

// openGraph connects to the AGE database, ensures the graph exists, and creates
// the tables this application owns.
func openGraph(ctx context.Context) (*age.Graph, error) {
	g, err := age.Open(ctx, age.Config{
		User:      dbUser,
		Password:  dbPassword,
		Host:      dbHost,
		Name:      dbName,
		GraphName: graphName,

		// Retrieval issues several independent queries per question while ingest
		// holds one connection for a long time. Sized so they do not wait on
		// each other.
		MaxIdleConns: 4,
		MaxOpenConns: 8,
		DisableTLS:   true,
	})
	if err != nil {
		return nil, err
	}

	if err := createTables(ctx, g.DB()); err != nil {
		g.Close()
		return nil, fmt.Errorf("create tables: %w", err)
	}

	fmt.Printf("- connected to AGE on %s, graph %q\n", dbHost, graphName)

	return g, nil
}

// createTables builds the two tables. extracted is the per-chunk resume log.
func createTables(ctx context.Context, db *sqlx.DB) error {
	stmts := []string{
		fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS chunks (
			id        INT PRIMARY KEY,
			chapter   TEXT NOT NULL,
			text      TEXT NOT NULL,
			embedding VECTOR(%d) NOT NULL
		)`, dimensions),

		// Step2 and step3 create this table without a chapter column. If one of
		// them ingested first the CREATE above is a no-op and reads would fail.
		`ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chapter TEXT NOT NULL DEFAULT ''`,

		`CREATE TABLE IF NOT EXISTS extracted (
			chunk_id   INT PRIMARY KEY,
			entities   INT NOT NULL,
			extracted_at TIMESTAMPTZ NOT NULL DEFAULT now()
		)`,

		`CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
		 USING hnsw (embedding vector_cosine_ops)`,
	}

	for _, q := range stmts {
		if _, err := db.ExecContext(ctx, q); err != nil {
			return err
		}
	}

	return nil
}

// =============================================================================

// loadData brings the database up to date. Every stage skips work already done,
// so an ingested database costs three lookup queries and nothing else.
func loadData(ctx context.Context, g *age.Graph, krnEmbed *kronk.Kronk, krnChat *kronk.Kronk, chunksFile string) error {
	chunks, err := parseChunks(chunksFile)
	if err != nil {
		return fmt.Errorf("parse chunks: %w", err)
	}

	if err := embedChunks(ctx, g.DB(), krnEmbed, chunks); err != nil {
		return fmt.Errorf("embed chunks: %w", err)
	}

	if err := buildStructure(ctx, g, chunks); err != nil {
		return fmt.Errorf("build structure: %w", err)
	}

	if err := buildEntities(ctx, g, krnChat, chunks); err != nil {
		return fmt.Errorf("build entities: %w", err)
	}

	return nil
}

// embedChunks vectorizes and stores every chunk that is not already stored.
func embedChunks(ctx context.Context, db *sqlx.DB, krnEmbed *kronk.Kronk, chunks []chunk) error {
	done, err := storedIDs(ctx, db, `SELECT id FROM chunks`)
	if err != nil {
		return fmt.Errorf("stored ids: %w", err)
	}

	todo := make([]chunk, 0, len(chunks))
	for _, c := range chunks {
		if !done[c.id] {
			todo = append(todo, c)
		}
	}

	if len(todo) == 0 {
		fmt.Printf("- %d chunks already embedded\n", len(done))
		return nil
	}

	fmt.Printf("- embedding %d of %d chunks\n", len(todo), len(chunks))

	// Prepared once because the insert runs once per chunk.
	const insert = `
	INSERT INTO chunks (id, chapter, text, embedding)
	VALUES ($1, $2, $3, $4::vector)
	ON CONFLICT (id) DO NOTHING`

	stmt, err := db.PreparexContext(ctx, insert)
	if err != nil {
		return fmt.Errorf("prepare insert: %w", err)
	}
	defer stmt.Close()

	fmt.Print("\033[s")
	t := time.Now()

	for i, c := range todo {
		fmt.Print("\033[u\033[K")
		fmt.Printf("Vectorizing: %d of %d", i+1, len(todo))

		embedding, err := embed(ctx, krnEmbed, c.text)
		if err != nil {
			return fmt.Errorf("embed chunk %d: %w", c.id, err)
		}

		if _, err := stmt.ExecContext(ctx, c.id, c.chapter, c.text, vector.FormatPGVector(embedding)); err != nil {
			return fmt.Errorf("insert chunk %d: %w", c.id, err)
		}
	}

	fmt.Printf("\nVectorized in %v\n", time.Since(t))

	return nil
}

// buildStructure writes the edges plain chunking throws away: which chunk
// follows which, and which chapter each came from. No model is involved.
// All MERGE, so an interrupted ingest resumes without duplicating.
func buildStructure(ctx context.Context, g *age.Graph, chunks []chunk) error {
	count, err := g.Count(ctx, `MATCH (c:Chunk) RETURN count(c)`)
	if err != nil {
		return fmt.Errorf("count chunk nodes: %w", err)
	}

	if count == len(chunks) {
		fmt.Printf("- structural graph has %d chunk nodes\n", count)
		return nil
	}

	fmt.Print("Building structural graph...")
	t := time.Now()

	for _, c := range chunks {
		q := fmt.Sprintf(`MERGE (:Chunk {id: %d})`, c.id)
		if err := g.Exec(ctx, q); err != nil {
			return fmt.Errorf("merge chunk node %d: %w", c.id, err)
		}
	}

	// -------------------------------------------------------------------------
	// A chunk boundary is an artifact of ingestion. NEXT lets retrieval undo it.

	for i := 1; i < len(chunks); i++ {
		q := fmt.Sprintf(`
			MATCH (a:Chunk {id: %d}), (b:Chunk {id: %d})
			MERGE (a)-[:NEXT]->(b)`,
			chunks[i-1].id, chunks[i].id)

		if err := g.Exec(ctx, q); err != nil {
			return fmt.Errorf("merge next edge %d: %w", chunks[i].id, err)
		}
	}

	// -------------------------------------------------------------------------

	for _, c := range chunks {
		q := fmt.Sprintf(`MERGE (:Section {name: %s})`, age.Quote(c.chapter))
		if err := g.Exec(ctx, q); err != nil {
			return fmt.Errorf("merge section: %w", err)
		}

		q = fmt.Sprintf(`
			MATCH (s:Section {name: %s}), (c:Chunk {id: %d})
			MERGE (s)-[:HAS_CHUNK]->(c)`,
			age.Quote(c.chapter), c.id)

		if err := g.Exec(ctx, q); err != nil {
			return fmt.Errorf("merge has_chunk edge %d: %w", c.id, err)
		}
	}

	fmt.Printf(" done in %v\n", time.Since(t))

	return nil
}

// =============================================================================

// chunksByID fetches text and chapter for the ids a traversal handed back.
func chunksByID(ctx context.Context, db *sqlx.DB, ids []int) ([]document, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	q, args, err := sqlx.In(`SELECT id, chapter, text FROM chunks WHERE id IN (?)`, ids)
	if err != nil {
		return nil, fmt.Errorf("build in clause: %w", err)
	}

	rows, err := db.QueryContext(ctx, db.Rebind(q), args...)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	var docs []document

	for rows.Next() {
		var d document
		if err := rows.Scan(&d.ID, &d.Chapter, &d.Text); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		docs = append(docs, d)
	}

	return docs, rows.Err()
}

// storedIDs collects a query's id column into the set of work already done.
func storedIDs(ctx context.Context, db *sqlx.DB, query string) (map[int]bool, error) {
	rows, err := db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	out := make(map[int]bool)

	for rows.Next() {
		var id int
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		out[id] = true
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

	// The model returns float32 and the pgvector helper wants float64.
	out := make([]float64, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		out[i] = float64(v)
	}

	return out, nil
}
