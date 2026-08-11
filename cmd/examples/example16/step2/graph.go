// Everything that talks to the database. Chunk text and its embedding live in
// a pgvector table, the structure of the book beside it as an Apache AGE
// graph. The graph stores keys and the table stores content, so traversals
// return ids and text is fetched with a parameterized query. That keeps book
// text out of hand-built Cypher and the graph small.
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

// document is one chunk on its way into the prompt. Hop records how retrieval
// reached it, by similarity or by walking an edge out of something that was.
type document struct {
	ID         int
	Chapter    string
	Text       string
	Similarity float64
	Hop        string
}

// =============================================================================

// openGraph connects to the Apache AGE database and makes sure the graph exists.
func openGraph(ctx context.Context) (*age.Graph, error) {
	g, err := age.Open(ctx, age.Config{
		User:         dbUser,
		Password:     dbPassword,
		Host:         dbHost,
		Name:         dbName,
		GraphName:    graphName,
		MaxIdleConns: 2,
		MaxOpenConns: 5,
		DisableTLS:   true,
	})
	if err != nil {
		return nil, err
	}

	fmt.Printf("- connected to AGE on %s, graph %q\n", dbHost, graphName)

	return g, nil
}

// =============================================================================

// loadData ingests the chunks file on the first run only. Embedding the book is
// a model call per chunk, so the check is what keeps a restart instant.
func loadData(ctx context.Context, g *age.Graph, krnEmbed *kronk.Kronk, chunksFile string) error {
	db := g.DB()

	const create = `
	CREATE TABLE IF NOT EXISTS chunks (
		id        INT PRIMARY KEY,
		text      TEXT NOT NULL,
		embedding VECTOR(%d) NOT NULL
	)`

	if _, err := db.ExecContext(ctx, fmt.Sprintf(create, dimensions)); err != nil {
		return fmt.Errorf("create table: %w", err)
	}

	var rowCount int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks`).Scan(&rowCount); err != nil {
		return fmt.Errorf("count chunks: %w", err)
	}

	if rowCount > 0 {
		fmt.Printf("- table exists with %d rows\n", rowCount)
		return nil
	}

	// -------------------------------------------------------------------------

	chunks, err := parseChunks(chunksFile)
	if err != nil {
		return fmt.Errorf("parse chunks: %w", err)
	}

	fmt.Printf("- parsed %d chunks\n", len(chunks))

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

		const insert = `INSERT INTO chunks (id, text, embedding) VALUES ($1, $2, $3::vector)`
		if _, err := db.ExecContext(ctx, insert, c.id, c.text, vector.FormatPGVector(embedding)); err != nil {
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

	// -------------------------------------------------------------------------

	if err := buildStructure(ctx, g, chunks); err != nil {
		return fmt.Errorf("build structure: %w", err)
	}

	return nil
}

// buildStructure writes the edges plain chunking throws away: which chunk
// follows which, and which chapter each came from. No model is involved.
func buildStructure(ctx context.Context, g *age.Graph, chunks []chunk) error {
	fmt.Print("Building structural graph...")

	t := time.Now()

	for _, c := range chunks {
		q := fmt.Sprintf(`CREATE (:Chunk {id: %d})`, c.id)
		if err := g.Exec(ctx, q); err != nil {
			return fmt.Errorf("create chunk node %d: %w", c.id, err)
		}
	}

	// -------------------------------------------------------------------------
	// A chunk boundary is an artifact of ingestion. NEXT lets retrieval undo it.

	for i := 1; i < len(chunks); i++ {
		q := fmt.Sprintf(`
			MATCH (a:Chunk {id: %d}), (b:Chunk {id: %d})
			CREATE (a)-[:NEXT]->(b)`,
			chunks[i-1].id, chunks[i].id)

		if err := g.Exec(ctx, q); err != nil {
			return fmt.Errorf("create next edge %d: %w", chunks[i].id, err)
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
			CREATE (s)-[:HAS_CHUNK]->(c)`,
			age.Quote(c.chapter), c.id)

		if err := g.Exec(ctx, q); err != nil {
			return fmt.Errorf("create has_chunk edge %d: %w", c.id, err)
		}
	}

	fmt.Printf(" done in %v\n", time.Since(t))

	return nil
}

// =============================================================================

// search runs retrieval in two stages. An ordinary vector search, then a walk
// along NEXT out of the hits to pull back what chunking cut in half.
func search(ctx context.Context, g *age.Graph, krnEmbed *kronk.Kronk, question string, topK int) ([]document, error) {
	db := g.DB()

	embedding, err := embed(ctx, krnEmbed, question)
	if err != nil {
		return nil, fmt.Errorf("embed question: %w", err)
	}

	docs, err := vectorSearch(ctx, db, embedding, topK)
	if err != nil {
		return nil, fmt.Errorf("vector search: %w", err)
	}

	if len(docs) == 0 {
		return nil, nil
	}

	// -------------------------------------------------------------------------

	seen := make(map[int]bool, len(docs))
	ids := make([]int, 0, len(docs))
	for _, d := range docs {
		seen[d.ID] = true
		ids = append(ids, d.ID)
	}

	neighbors, err := neighborIDs(ctx, g, ids)
	if err != nil {
		return nil, fmt.Errorf("neighbor ids: %w", err)
	}

	var add []int
	for _, id := range neighbors {
		if seen[id] {
			continue
		}
		seen[id] = true
		add = append(add, id)
	}

	if len(add) > 0 {
		rows, err := chunksByID(ctx, db, add)
		if err != nil {
			return nil, fmt.Errorf("fetch neighbors: %w", err)
		}

		for _, d := range rows {
			d.Hop = "NEXT"
			docs = append(docs, d)
		}
	}

	// -------------------------------------------------------------------------
	// The chapter is not a column on chunks. It comes back out of the graph.

	all := make([]int, 0, len(docs))
	for _, d := range docs {
		all = append(all, d.ID)
	}

	chapters, err := chapterNames(ctx, g, all)
	if err != nil {
		return nil, fmt.Errorf("chapter names: %w", err)
	}

	for i := range docs {
		docs[i].Chapter = chapters[docs[i].ID]
	}

	return docs, nil
}

// vectorSearch returns the topK chunks closest to the query by cosine distance.
func vectorSearch(ctx context.Context, db *sqlx.DB, embedding []float64, topK int) ([]document, error) {
	const q = `
	SELECT
		id,
		text,
		1 - (embedding <=> $1::vector) AS similarity
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
		if err := rows.Scan(&d.ID, &d.Text, &d.Similarity); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		d.Hop = "vector"
		docs = append(docs, d)
	}

	return docs, rows.Err()
}

// neighborIDs walks NEXT in both directions. Undirected, because the chunk
// before a hit completes it just as often as the chunk after it.
func neighborIDs(ctx context.Context, g *age.Graph, ids []int) ([]int, error) {
	q := fmt.Sprintf(`
		MATCH (c:Chunk)-[:NEXT]-(n:Chunk)
		WHERE c.id IN [%s]
		RETURN n.id`, age.Ints(ids))

	rows, err := g.Query(ctx, q, "id agtype")
	if err != nil {
		return nil, err
	}

	out := make([]int, 0, len(rows))
	for _, r := range rows {
		id, err := age.ParseID(r[0])
		if err != nil {
			return nil, err
		}
		out = append(out, id)
	}

	return out, nil
}

// chapterNames maps chunk ids to their chapter by walking HAS_CHUNK back.
func chapterNames(ctx context.Context, g *age.Graph, ids []int) (map[int]string, error) {
	q := fmt.Sprintf(`
		MATCH (s:Section)-[:HAS_CHUNK]->(c:Chunk)
		WHERE c.id IN [%s]
		RETURN c.id, s.name`, age.Ints(ids))

	rows, err := g.Query(ctx, q, "id agtype", "name agtype")
	if err != nil {
		return nil, err
	}

	out := make(map[int]string, len(rows))
	for _, r := range rows {
		id, err := age.ParseID(r[0])
		if err != nil {
			return nil, err
		}
		out[id] = age.Scalar(r[1])
	}

	return out, nil
}

func chunksByID(ctx context.Context, db *sqlx.DB, ids []int) ([]document, error) {
	q, args, err := sqlx.In(`SELECT id, text FROM chunks WHERE id IN (?)`, ids)
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
		if err := rows.Scan(&d.ID, &d.Text); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		docs = append(docs, d)
	}

	return docs, rows.Err()
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

	book := string(data)

	blocks := chunkRE.FindAllString(book, -1)
	if len(blocks) == 0 {
		return nil, fmt.Errorf("no chunks found in %s", chunksFile)
	}

	titles := make(map[string]string)
	for _, m := range tocRE.FindAllStringSubmatch(book, -1) {
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
	embedding := resp.Data[0].Embedding

	out := make([]float64, len(embedding))
	for i, v := range embedding {
		out[i] = float64(v)
	}

	return out, nil
}
