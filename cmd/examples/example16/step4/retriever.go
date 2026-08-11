// The retrieval half of this step. Step2 and step3 handed the model everything
// two hops reached and left it there. Two hops off ten vector hits is most of a
// chapter, and nothing in the pipeline said which of those chunks was worth a
// token. So retrieval here is three separate jobs:
//
//   - Reach: vector hits, their page neighbours, and the chunks that share a
//     subject with them. Hub subjects are noise, not hops.
//   - Score: every candidate is measured against the question, then discounted
//     for how far from a hit it came in.
//   - Fit: the best candidates are packed until the token budget is spent,
//     because a document count says nothing about how much context was used.

package main

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/age"
	"github.com/ardanlabs/ai-training/foundation/tiktoken"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/ardanlabs/kronk/sdk/kronk"
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
	// short of the 32768 window. That is the point: a chunk earns its place by
	// being one of the best few, not by there still being room. Roughly ten
	// chunks of this book.
	contextBudget = 8000

	// candidatePool caps how many candidates are fetched and scored, set to what
	// contextBudget can actually hold: carrying chunk text out of the database
	// with no room for it in the prompt buys nothing. The cut is made in cosine
	// order, before the hop discounts below, so a chunk those discounts would
	// have promoted still has to survive on cosine alone. Step5 keeps this pool
	// and spends it on the reranker.
	candidatePool = 12

	// Hop discounts. A chunk that matched the question is worth more than a
	// chunk that merely sits beside one, which is worth more than a chunk that
	// arrived because it shares a subject. Similarity alone does not know that.
	weightVector = 1.00
	weightNext   = 0.85
	weightEntity = 0.70
)

// =============================================================================

// retriever holds what retrieval needs across questions. Hubs come out of the
// finished graph and the encoding table is several megabytes to load, so both
// are built once.
type retriever struct {
	g        *age.Graph
	db       *sqlx.DB
	krnEmbed *kronk.Kronk
	tkn      *tiktoken.Tiktoken
	hubs     map[string]bool
}

// newRetriever prepares retrieval against an ingested graph.
func newRetriever(ctx context.Context, g *age.Graph, krnEmbed *kronk.Kronk) (*retriever, error) {
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
		g:        g,
		db:       g.DB(),
		krnEmbed: krnEmbed,
		tkn:      tkn,
		hubs:     hubs,
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
	// Score. The database returns candidates in cosine order, already cut to the
	// pool. Each is then discounted by how it arrived.

	ids := slices.Sorted(maps.Keys(hops))

	docs, err := scoreChunks(ctx, r.db, embedding, ids, candidatePool)
	if err != nil {
		return nil, fmt.Errorf("score chunks: %w", err)
	}

	for i := range docs {
		docs[i].Hop = hops[docs[i].ID]
		docs[i].Score = docs[i].Similarity * weights[docs[i].ID]
	}

	slices.SortFunc(docs, func(a, b document) int {
		// Id breaks ties so two runs of one question build the same context.
		return cmp.Or(-cmp.Compare(a.Score, b.Score), cmp.Compare(a.ID, b.ID))
	})

	return r.fit(docs), nil
}

// fit takes chunks in score order until the token budget is spent. A low
// scoring chunk is skipped rather than ending the loop: the next one may be
// small enough to still earn its place.
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
// The shortlist is cut here rather than in Go: the graph nominates far more
// candidates than the prompt has room for, and every row past the limit is chunk
// text carried out of the database to be thrown away. Id breaks ties so two runs
// of one question shortlist the same chunks.
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
// a model to ask it a question.
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
	// The same set retrieval refuses to hop through, so the report can mark which
	// of the popular subjects above are being ignored.

	hubs, err := hubEntities(ctx, g)
	if err != nil {
		return stats{}, fmt.Errorf("hub entities: %w", err)
	}

	s.hubs = slices.Sorted(maps.Keys(hubs))

	return s, nil
}

// print writes the report. Extracted against chunks is the one number that says
// whether ingest finished, since extraction is the part that gets interrupted.
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
