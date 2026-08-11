// This file holds the retrieval half of this step: step4's pipeline with its
// middle job replaced.
//
//   - Reach: unchanged. Vector hits, their page neighbours, and the chunks that
//     share a subject with them. Hub subjects are not hops, they are noise.
//   - Score: the reranker reads the question against each candidate and returns
//     a relevance score. Step4 approximated this with the cosine distance
//     between two independently computed vectors.
//   - Fit: unchanged. The best candidates are packed until the token budget is
//     spent, because a document count says nothing about how much was used.
//
// Splitting recall from precision is what makes this affordable. The vector index
// and the graph are cheap and generous, so they over-reach on purpose; the
// reranker costs a forward pass per candidate, so it only ever sees the few dozen
// chunks the first stage found. Running it over the whole book per question would
// be correct and unaffordable.

package main

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/age"
	"github.com/ardanlabs/ai-training/foundation/tiktoken"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/jmoiron/sqlx"
)

const (
	// topK is how many chunks the vector search seeds the graph walk with.
	topK = 8

	// entityCap is how many chunks any one subject may carry in. Without it the
	// most talked about subject in the hit set decides the whole context.
	entityCap = 2

	// hubFraction marks a subject as a hub. A subject mentioned by more than this
	// share of the book relates everything to everything, so it says nothing
	// about a particular question. "type" and "value" are hubs here.
	hubFraction = 0.05

	// contextBudget is how many tokens of chunk text may enter the prompt, well
	// short of the 32768 window. That is the point: with the reranker deciding the
	// order, the top few chunks should be carrying the answer, and a budget this
	// tight puts that claim to the test. Roughly ten chunks of this book.
	contextBudget = 8000

	// candidatePool caps how many candidates are fetched and scored, the same pool
	// step4 uses. Here every one is also a forward pass through the reranker, so
	// this constant is the per-question latency too. The cut is made in cosine
	// order, the one place the old proxy still decides something: the reranker
	// reorders this pool, it cannot reach past it.
	candidatePool = 12

	// Hop discounts, kept from step4 but much lighter. Against a cosine score they
	// did most of the work of separating a hit from a subject hop. The reranker
	// judges relevance directly, so all that is left for them is a prior: between
	// two chunks it rates the same, prefer the shorter path.
	weightVector = 1.00
	weightNext   = 0.97
	weightEntity = 0.95
)

// =============================================================================

// retriever holds what retrieval needs across questions. Hubs come out of the
// finished graph and the encoding table is several megabytes to load, so both
// are built once.
type retriever struct {
	g         *age.Graph
	db        *sqlx.DB
	krnEmbed  *kronk.Kronk
	krnRerank *kronk.Kronk
	tkn       *tiktoken.Tiktoken
	hubs      map[string]bool
}

// newRetriever prepares retrieval against an ingested graph.
func newRetriever(ctx context.Context, g *age.Graph, krnEmbed *kronk.Kronk, krnRerank *kronk.Kronk) (*retriever, error) {
	tkn, err := tiktoken.NewTiktoken()
	if err != nil {
		return nil, fmt.Errorf("new tiktoken: %w", err)
	}

	hubs, err := hubEntities(ctx, g)
	if err != nil {
		return nil, fmt.Errorf("hub entities: %w", err)
	}

	fmt.Printf("- retriever ready, %d hub subjects will not be hopped through\n", len(hubs))

	ret := retriever{
		g:         g,
		db:        g.DB(),
		krnEmbed:  krnEmbed,
		krnRerank: krnRerank,
		tkn:       tkn,
		hubs:      hubs,
	}

	return &ret, nil
}

// retrieve turns a question into the chunks that answer it. Reach, score, fit.
func (r *retriever) retrieve(ctx context.Context, question string) ([]document, error) {
	embedding, err := embed(ctx, r.krnEmbed, question)
	if err != nil {
		return nil, fmt.Errorf("embed question: %w", err)
	}

	// -------------------------------------------------------------------------
	// Reach. hops records why each candidate is a candidate, which is both the
	// discount it gets below and the explanation printed in the table.

	hits, err := vectorIDs(ctx, r.db, embedding, topK)
	if err != nil {
		return nil, fmt.Errorf("vector search: %w", err)
	}

	if len(hits) == 0 {
		return nil, nil
	}

	hops := make(map[int]string, len(hits))
	weights := make(map[int]float64, len(hits))

	for _, id := range hits {
		hops[id] = "vector"
		weights[id] = weightVector
	}

	// -------------------------------------------------------------------------
	// A chunk boundary is an artifact of ingestion, so a hit's neighbours often
	// hold the rest of what it started saying.

	next, err := nextHop(ctx, r.g, hits)
	if err != nil {
		return nil, fmt.Errorf("next hop: %w", err)
	}

	for _, id := range slices.Sorted(maps.Keys(next)) {
		if _, ok := hops[id]; ok {
			continue
		}

		hops[id] = fmt.Sprintf("NEXT of %d", next[id])
		weights[id] = weightNext
	}

	// -------------------------------------------------------------------------
	// This hop leaves page order and follows what the chunks are about, so a
	// chunk can arrive without sharing any of the question's wording.

	viaEntity, err := entityHop(ctx, r.g, hits, entityCap, r.hubs)
	if err != nil {
		return nil, fmt.Errorf("entity hop: %w", err)
	}

	for _, id := range slices.Sorted(maps.Keys(viaEntity)) {
		if _, ok := hops[id]; ok {
			continue
		}

		hops[id] = "via " + viaEntity[id]
		weights[id] = weightEntity
	}

	// -------------------------------------------------------------------------
	// Score. The database returns the candidates in cosine order, already cut to
	// the pool the reranker can afford. That similarity is no longer the ranking,
	// only the shortlist and the number the table prints beside the reranker's.

	ids := slices.Sorted(maps.Keys(hops))

	docs, err := scoreChunks(ctx, r.db, embedding, ids, candidatePool)
	if err != nil {
		return nil, fmt.Errorf("score chunks: %w", err)
	}

	for i := range docs {
		docs[i].Hop = hops[docs[i].ID]
	}

	// The rank the old pipeline would have used, recorded before the reranker
	// touches the order.
	for i := range docs {
		docs[i].VectorRank = i + 1
	}

	// -------------------------------------------------------------------------
	// The cross-encoder reads the question against each candidate and returns a
	// relevance score in [0,1]. The hop weight is applied on top as a prior.

	if err := r.rerank(ctx, question, docs); err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}

	for i := range docs {
		docs[i].Score = docs[i].Relevance * weights[docs[i].ID]
	}

	slices.SortFunc(docs, func(a, b document) int {
		// Id breaks ties so two runs of one question build the same context.
		return cmp.Or(-cmp.Compare(a.Score, b.Score), cmp.Compare(a.ID, b.ID))
	})

	return r.fit(docs), nil
}

// rerank scores every candidate against the question in one call. The model
// returns positions into the slice it was given, not ids, so the order the
// documents went in is what maps a score back to a chunk.
func (r *retriever) rerank(ctx context.Context, question string, docs []document) error {
	if len(docs) == 0 {
		return nil
	}

	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.Text
	}

	d := model.D{
		"query":     question,
		"documents": texts,
	}

	start := time.Now()

	resp, err := r.krnRerank.Rerank(ctx, d)
	if err != nil {
		return fmt.Errorf("rerank call: %w", err)
	}

	for _, result := range resp.Data {
		if result.Index < 0 || result.Index >= len(docs) {
			return fmt.Errorf("rerank index %d out of range", result.Index)
		}

		docs[result.Index].Relevance = float64(result.RelevanceScore)
	}

	fmt.Printf("  reranked %d candidates in %s, %d prompt tokens\n",
		len(docs), time.Since(start).Round(time.Millisecond), resp.Usage.PromptTokens)

	return nil
}

// fit takes chunks in score order until the token budget is spent. An oversized
// chunk is skipped rather than ending the loop: the next one may still fit.
func (r *retriever) fit(docs []document) []document {
	out := make([]document, 0, len(docs))
	var used int

	for _, doc := range docs {
		tokens := r.tkn.TokenCount(doc.Text)

		if used+tokens > contextBudget {
			continue
		}

		used += tokens
		out = append(out, doc)
	}

	fmt.Printf("  %d of %d candidates, %d of %d context tokens\n",
		len(out), len(docs), used, contextBudget)

	return out
}

// =============================================================================

// hubEntities returns the subjects that are mentioned by too much of the book
// to be evidence of anything. They stay in the graph; retrieval will not hop
// through them.
func hubEntities(ctx context.Context, g *age.Graph) (map[string]bool, error) {
	chunks, err := g.Count(ctx, `MATCH (c:Chunk) RETURN count(c)`)
	if err != nil {
		return nil, fmt.Errorf("count chunks: %w", err)
	}

	if chunks == 0 {
		return map[string]bool{}, nil
	}

	limit := int(float64(chunks) * hubFraction)

	q := fmt.Sprintf(`
		MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
		WITH e, count(c) AS mentions
		WHERE mentions > %d
		RETURN e.name`, limit)

	rows, err := g.Query(ctx, q, "name agtype")
	if err != nil {
		return nil, err
	}

	hubs := make(map[string]bool, len(rows))
	for _, row := range rows {
		hubs[age.Scalar(row[0])] = true
	}

	return hubs, nil
}

// nextHop walks NEXT in both directions and reports which hit pulled each
// neighbour in. Undirected, because the chunk before a hit completes it as
// often as the chunk after it.
func nextHop(ctx context.Context, g *age.Graph, ids []int) (map[int]int, error) {
	q := fmt.Sprintf(`
		MATCH (c:Chunk)-[:NEXT]-(n:Chunk)
		WHERE c.id IN [%s]
		RETURN n.id, c.id`, age.Ints(ids))

	// The second column cannot be called "from": these names are spelled into
	// the AS list of a plain SELECT, where a reserved keyword is a syntax error.
	rows, err := g.Query(ctx, q, "id agtype", "src agtype")
	if err != nil {
		return nil, err
	}

	out := make(map[int]int, len(rows))

	for _, row := range rows {
		id, err := age.ParseID(row[0])
		if err != nil {
			return nil, err
		}

		from, err := age.ParseID(row[1])
		if err != nil {
			return nil, err
		}

		if _, ok := out[id]; ok {
			continue
		}

		out[id] = from
	}

	return out, nil
}

// scoreChunks fetches the candidates and how close each one is to the question,
// using the distance operator the vector search used so a hop and a hit are
// measured on the same scale.
//
// The shortlist is cut here rather than in Go: the graph can nominate far more
// candidates than the reranker will be paid to read, and every row past the limit
// is chunk text carried out of the database to be thrown away. The id breaks ties
// so two runs of one question shortlist the same chunks.
func scoreChunks(ctx context.Context, db *sqlx.DB, embedding []float64, ids []int, limit int) ([]document, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	const query = `
	SELECT
		id,
		chapter,
		text,
		1 - (embedding <=> ?::vector) AS similarity
	FROM chunks
	WHERE id IN (?)
	ORDER BY similarity DESC, id ASC
	LIMIT ?`

	q, args, err := sqlx.In(query, vector.FormatPGVector(embedding), ids, limit)
	if err != nil {
		return nil, fmt.Errorf("build in clause: %w", err)
	}

	rows, err := db.QueryContext(ctx, db.Rebind(q), args...)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	docs := make([]document, 0, min(len(ids), limit))

	for rows.Next() {
		var d document
		if err := rows.Scan(&d.ID, &d.Chapter, &d.Text, &d.Similarity); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		docs = append(docs, d)
	}

	return docs, rows.Err()
}

// vectorIDs returns the ids of the topK chunks closest to the question. Only
// ids, because scoreChunks fetches the text for hits and hops in one query.
func vectorIDs(ctx context.Context, db *sqlx.DB, embedding []float64, topK int) ([]int, error) {
	const q = `
	SELECT id
	FROM chunks
	ORDER BY embedding <=> $1::vector
	LIMIT $2`

	rows, err := db.QueryContext(ctx, q, vector.FormatPGVector(embedding), topK)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	out := make([]int, 0, topK)

	for rows.Next() {
		var id int
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		out = append(out, id)
	}

	return out, rows.Err()
}

// =============================================================================

// stats reports what ingest built, so the graph can be inspected without loading
// a model.
type stats struct {
	chunkNodes  int
	sectionNode int
	entityNodes int
	nextEdges   int
	mentions    int
	related     int
	extracted   int
	hubs        []string
	topEntities []entityCount
}

type entityCount struct {
	name     string
	kind     string
	mentions int
}

// graphStats counts the graph and lists the subjects the book talks about most.
func graphStats(ctx context.Context, g *age.Graph, top int) (stats, error) {
	var s stats

	counts := []struct {
		cypher string
		dest   *int
	}{
		{`MATCH (c:Chunk) RETURN count(c)`, &s.chunkNodes},
		{`MATCH (s:Section) RETURN count(s)`, &s.sectionNode},
		{`MATCH (e:Entity) RETURN count(e)`, &s.entityNodes},
		{`MATCH ()-[r:NEXT]->() RETURN count(r)`, &s.nextEdges},
		{`MATCH ()-[r:MENTIONS]->() RETURN count(r)`, &s.mentions},
		{`MATCH ()-[r:RELATED]->() RETURN count(r)`, &s.related},
	}

	for _, c := range counts {
		count, err := g.Count(ctx, c.cypher)
		if err != nil {
			return stats{}, fmt.Errorf("count: %w", err)
		}

		*c.dest = count
	}

	// -------------------------------------------------------------------------

	if err := g.DB().QueryRowContext(ctx, `SELECT count(*) FROM extracted`).Scan(&s.extracted); err != nil {
		return stats{}, fmt.Errorf("count extracted: %w", err)
	}

	// -------------------------------------------------------------------------

	q := fmt.Sprintf(`
		MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
		WITH e, count(c) AS mentions
		RETURN e.name, e.kind, mentions
		ORDER BY mentions DESC
		LIMIT %d`, top)

	rows, err := g.Query(ctx, q, "name agtype", "kind agtype", "mentions agtype")
	if err != nil {
		return stats{}, fmt.Errorf("top entities: %w", err)
	}

	for _, row := range rows {
		mentions, err := age.ParseID(row[2])
		if err != nil {
			return stats{}, err
		}

		s.topEntities = append(s.topEntities, entityCount{
			name:     age.Scalar(row[0]),
			kind:     age.Scalar(row[1]),
			mentions: mentions,
		})
	}

	// -------------------------------------------------------------------------
	// The same set retrieval refuses to hop through, so the report shows which of
	// the popular subjects above are being ignored.

	hubs, err := hubEntities(ctx, g)
	if err != nil {
		return stats{}, fmt.Errorf("hub entities: %w", err)
	}

	s.hubs = slices.Sorted(maps.Keys(hubs))

	return s, nil
}

// print writes the report. Extracted against chunks is the one number that says
// whether ingest finished, extraction being the part that gets interrupted.
func (s stats) print() {
	fmt.Printf("\nGraph %q on %s\n", graphName, dbHost)
	fmt.Println(strings.Repeat("-", 60))

	fmt.Printf("  Chunk nodes    : %d\n", s.chunkNodes)
	fmt.Printf("  Section nodes  : %d\n", s.sectionNode)
	fmt.Printf("  Entity nodes   : %d\n", s.entityNodes)
	fmt.Printf("  NEXT edges     : %d\n", s.nextEdges)
	fmt.Printf("  MENTIONS edges : %d\n", s.mentions)
	fmt.Printf("  RELATED edges  : %d\n", s.related)
	fmt.Printf("  Chunks extracted: %d of %d\n", s.extracted, s.chunkNodes)

	fmt.Printf("\nMost mentioned subjects (* = hub, not hopped through)\n")
	fmt.Println(strings.Repeat("-", 60))

	hubs := make(map[string]bool, len(s.hubs))
	for _, name := range s.hubs {
		hubs[name] = true
	}

	for _, e := range s.topEntities {
		mark := " "
		if hubs[e.name] {
			mark = "*"
		}

		fmt.Printf("  %s %-32s %-10s %d\n", mark, truncate(e.name, 32), e.kind, e.mentions)
	}

	fmt.Printf("\n%d of %d subjects are hubs\n", len(s.hubs), s.entityNodes)
}
