// This example extends step2 with a more rigorous perf walkthrough that
// isolates the two embedding-throughput levers — concurrency and batch
// size — instead of conflating them.
//
// Four configurations are timed:
//
//   - Sequential        (concurrency=1, batchSize=1)  — baseline
//   - Batched only      (concurrency=1, batchSize=32) — server-side batch decode
//   - Parallel only     (concurrency=5, batchSize=1)  — multiple in-flight requests
//   - Batched+Parallel  (concurrency=5, batchSize=32) — both together
//
// Each configuration runs one warmup pass (discarded — first runs are
// usually faster than later runs because of server-side prefix cache, and
// we want to acknowledge that) followed by 3 timed passes. The reported
// duration is the MEDIAN of the timed passes, not the mean, because it's
// less sensitive to one slow outlier.
//
// After timing, a correctness check compares 5 random chunk embeddings
// from the Batched+Parallel run against the Sequential run using cosine
// similarity. If max delta exceeds 0.01 the run prints a warning. This
// catches embedding servers that produce subtly different vectors under
// batching or concurrency.
//
// Finally the Batched+Parallel results are stored in pgvector and the
// canonical end-to-end RAG query is run, so the perf demo still produces
// a usable corpus.
//
// # What this perf example does NOT improve
//
// Ingestion happens once per corpus. Query-time embedding is ONE call per
// question and is unaffected by these changes. If your interactive
// latency is bad, none of these knobs help — look at end-to-end latency
// of the query path instead.
//
// # Running the example
//
//	$ make example11-step3
//
// # Optional environment overrides
//
//	LLM_SERVER    chat completions endpoint (default: http://localhost:11435/v1/chat/completions)
//	LLM_MODEL     chat model name           (default: Qwen3-8B-Q8_0)
//	EMBED_SERVER  embeddings endpoint       (default: http://localhost:11435/v1/embeddings)
//	EMBED_MODEL   embedding model name      (default: embeddinggemma-300m-qat-Q8_0)
//
// # Prerequisites
//
//	$ make compose-up
//	$ make kronk-up
//
// # What this step adds over step2
//
//   - Four isolated configurations instead of two conflated ones.
//   - Warmup + 3 timed reps with median wall-clock reporting.
//   - Correctness verification against the sequential baseline.
//   - Honest framing of what is and isn't optimized.

// Example 11 — Step 3 — Isolated Perf Comparison + Correctness Check
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"regexp"
	"sort"
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

// embedConfig defines one row of the perf comparison table.
type embedConfig struct {
	name        string
	concurrency int
	batchSize   int
}

// runStats captures the result of one config's warmup + timed reps.
type runStats struct {
	cfg            embedConfig
	timedDurations []time.Duration
	median         time.Duration
	httpCallsPer   int // HTTP calls per pass (constant across reps)
	lastEmbeddings [][]float64
}

const (
	warmupReps = 1
	timedReps  = 3
)

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	fmt.Printf("\nChat Server:\n%s\n", llmURL)
	fmt.Printf("\nChat Model:\n%s\n", llmModel)
	fmt.Printf("\nEmbedding Server:\n%s\n", embedURL)
	fmt.Printf("\nEmbedding Model:\n%s\n", embedModel)

	chatLLM := client.NewLLM(llmURL, llmModel)
	embedLLM := client.NewLLM(embedURL, embedModel)

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

	fmt.Print("\nPostgreSQL connected.\n")

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

	// endregion

	// -------------------------------------------------------------------------
	// 2) Run four isolated configurations.
	//
	// Each config: 1 warmup pass (timing discarded) + 3 timed passes.
	// We report the MEDIAN of the timed passes. The warmup is important
	// because the embedding server's prefix cache makes the first pass
	// after a cold start unrepresentatively fast; subsequent passes
	// reflect steady-state better.
	// region 2) Run four isolated configurations.

	fmt.Print("\n============================================================\n")
	fmt.Printf("2) Perf Comparison (warmup=%d, timed reps=%d, report = median)\n", warmupReps, timedReps)
	fmt.Print("============================================================\n")

	configs := []embedConfig{
		{name: "Sequential       ", concurrency: 1, batchSize: 1},
		{name: "Batched only     ", concurrency: 1, batchSize: 32},
		{name: "Parallel only    ", concurrency: 5, batchSize: 1},
		{name: "Batched+Parallel ", concurrency: 5, batchSize: 32},
	}

	stats := make([]runStats, 0, len(configs))
	for _, cfg := range configs {
		s, err := timeConfig(ctx, embedLLM, chunks, cfg)
		if err != nil {
			return fmt.Errorf("config %q: %w", cfg.name, err)
		}
		stats = append(stats, s)
	}

	printPerfTable(stats)

	// endregion

	// -------------------------------------------------------------------------
	// 3) Correctness check.
	//
	// Compare 5 random chunks: their Batched+Parallel embedding vs their
	// Sequential embedding, by cosine similarity. If the embeddings come
	// from a server that's stable under batching and concurrency these
	// numbers should be effectively 1.0. If they diverge it usually means
	// the server normalizes per-request differently or the batched path
	// uses different padding/quantization than the single-input path.
	// region 3) Correctness check.

	fmt.Print("\n============================================================\n")
	fmt.Print("3) Correctness Check (Sequential vs Batched+Parallel)\n")
	fmt.Print("============================================================\n")

	seqStats := findStats(stats, "Sequential       ")
	bpStats := findStats(stats, "Batched+Parallel ")
	if seqStats == nil || bpStats == nil {
		return fmt.Errorf("missing stats for correctness check")
	}

	maxDelta := correctnessCheck(seqStats.lastEmbeddings, bpStats.lastEmbeddings, 5)

	switch {
	case maxDelta > 0.01:
		fmt.Printf("\nWARNING: max cosine-distance delta = %.6f (>0.01). Batched/parallel embeddings differ meaningfully from the sequential baseline.\n", maxDelta)

	default:
		fmt.Printf("\nOK: max cosine-distance delta = %.6f across 5 sampled chunks.\n", maxDelta)
	}

	// endregion

	// -------------------------------------------------------------------------
	// 4) Store the Batched+Parallel results in pgvector.
	// region 4) Store the Batched+Parallel results in pgvector.

	fmt.Print("\n============================================================\n")
	fmt.Print("4) Store in pgvector\n")
	fmt.Print("============================================================\n")

	dimensions := len(bpStats.lastEmbeddings[0])

	if err := initDB(ctx, db, dimensions); err != nil {
		return err
	}

	docs := make([]rag.Document, len(chunks))
	for i := range chunks {
		docs[i] = rag.Document{
			ID:        i,
			Name:      fmt.Sprintf("Chunk %d", i),
			Text:      chunks[i],
			Embedding: bpStats.lastEmbeddings[i],
		}
	}

	if err := insertDocuments(ctx, db, docs); err != nil {
		return err
	}

	fmt.Printf("\n%d documents stored.\n", len(docs))

	// endregion

	// -------------------------------------------------------------------------
	// 5) End-to-end RAG query.
	//
	// Reminder: this single query path is what users actually pay for at
	// interactive time. None of the throughput improvements above made it
	// faster — query-time embedding is still ONE call.
	// region 5) End-to-end RAG query.

	fmt.Print("\n============================================================\n")
	fmt.Print("5) End-to-End RAG Query (query-time path — unchanged by the perf work above)\n")
	fmt.Print("============================================================\n")

	const question = `According to the Ultimate Go Notebook, what is Bill Kennedy's "never, ever, never" rule about switching between data semantics in a call chain, and what does he say is never safe?`

	results, err := rag.SearchDocuments(ctx, db, embedLLM, question, 5)
	if err != nil {
		return fmt.Errorf("search documents: %w", err)
	}

	printResults(results)

	contextText := rag.BuildContext(results)

	const prompt = `/no_think

Use the following pieces of information to answer the user's question.
If you don't know the answer, say that you don't know.

Context: %s

Question: %s

Answer the question and provide additional helpful information, but be concise.

Responses should be properly formatted to be easily read.
`

	finalPrompt := fmt.Sprintf(prompt, contextText, question)

	ch, err := chatLLM.ChatCompletionsSSE(ctx, finalPrompt)
	if err != nil {
		return fmt.Errorf("chat completions sse: %w", err)
	}

	fmt.Print("\n==============================\n")
	fmt.Print("-----\n")
	fmt.Printf("> Prompt: %s\n", question)
	fmt.Print("-----\n")
	fmt.Print("> Response: ")

	for resp := range ch {
		if len(resp.Choices) == 0 {
			continue
		}

		fmt.Print(resp.Choices[0].Delta.Content)
	}

	fmt.Print("\n-----\n")

	// endregion

	return nil
}

// =============================================================================

// timeConfig runs warmup + timed reps for one config and returns the
// median wall-clock time plus the last rep's embeddings.
func timeConfig(ctx context.Context, embedLLM *client.LLM, chunks []string, cfg embedConfig) (runStats, error) {
	fmt.Printf("\n[%s] warmup...", strings.TrimSpace(cfg.name))

	httpCalls, _, err := embedAll(ctx, embedLLM, chunks, cfg)
	if err != nil {
		return runStats{}, fmt.Errorf("warmup: %w", err)
	}

	durations := make([]time.Duration, 0, timedReps)
	var lastEmbeddings [][]float64

	for r := 0; r < timedReps; r++ {
		start := time.Now()

		_, embs, err := embedAll(ctx, embedLLM, chunks, cfg)
		if err != nil {
			return runStats{}, fmt.Errorf("rep %d: %w", r, err)
		}

		d := time.Since(start)
		durations = append(durations, d)
		lastEmbeddings = embs

		fmt.Printf(" rep%d=%s", r+1, d.Round(time.Millisecond))
	}
	fmt.Print("\n")

	return runStats{
		cfg:            cfg,
		timedDurations: durations,
		median:         medianDuration(durations),
		httpCallsPer:   httpCalls,
		lastEmbeddings: lastEmbeddings,
	}, nil
}

// embedAll dispatches embedding for all chunks according to cfg. Returns
// the number of HTTP calls performed and the resulting embeddings in
// chunk order.
func embedAll(ctx context.Context, embedLLM *client.LLM, chunks []string, cfg embedConfig) (int, [][]float64, error) {
	embeddings := make([][]float64, len(chunks))

	switch {
	case cfg.batchSize <= 1 && cfg.concurrency <= 1:
		// Sequential, one chunk per call.
		for i, c := range chunks {
			emb, err := embedLLM.EmbedText(ctx, c)
			if err != nil {
				return 0, nil, fmt.Errorf("seq embed %d: %w", i, err)
			}
			embeddings[i] = emb
		}
		return len(chunks), embeddings, nil

	case cfg.batchSize > 1 && cfg.concurrency <= 1:
		// Sequential batches.
		calls := 0
		for start := 0; start < len(chunks); start += cfg.batchSize {
			end := min(start+cfg.batchSize, len(chunks))
			batch := chunks[start:end]
			embs, err := embedLLM.EmbedTexts(ctx, batch)
			if err != nil {
				return 0, nil, fmt.Errorf("seq batch [%d:%d]: %w", start, end, err)
			}
			for i, emb := range embs {
				embeddings[start+i] = emb
			}
			calls++
		}
		return calls, embeddings, nil

	case cfg.batchSize <= 1 && cfg.concurrency > 1:
		// Parallel, one chunk per call, errgroup with semaphore.
		g, gCtx := errgroup.WithContext(ctx)
		g.SetLimit(cfg.concurrency)
		for i, c := range chunks {
			g.Go(func() error {
				emb, err := embedLLM.EmbedText(gCtx, c)
				if err != nil {
					return fmt.Errorf("par embed %d: %w", i, err)
				}
				embeddings[i] = emb
				return nil
			})
		}
		if err := g.Wait(); err != nil {
			return 0, nil, err
		}
		return len(chunks), embeddings, nil

	default:
		// Parallel batches.
		calls := 0
		g, gCtx := errgroup.WithContext(ctx)
		g.SetLimit(cfg.concurrency)
		for start := 0; start < len(chunks); start += cfg.batchSize {
			end := min(start+cfg.batchSize, len(chunks))
			batch := chunks[start:end]
			calls++
			g.Go(func() error {
				embs, err := embedLLM.EmbedTexts(gCtx, batch)
				if err != nil {
					return fmt.Errorf("par batch [%d:%d]: %w", start, end, err)
				}
				for i, emb := range embs {
					embeddings[start+i] = emb
				}
				return nil
			})
		}
		if err := g.Wait(); err != nil {
			return 0, nil, err
		}
		return calls, embeddings, nil
	}
}

func medianDuration(ds []time.Duration) time.Duration {
	sorted := append([]time.Duration(nil), ds...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	return sorted[len(sorted)/2]
}

func printPerfTable(stats []runStats) {
	if len(stats) == 0 {
		return
	}

	baseline := stats[0].median

	fmt.Print("\n")
	fmt.Printf("  %-22s  %-10s  %-10s  %-12s  %-10s\n", "config", "concur", "batch", "median", "speedup")
	fmt.Printf("  %-22s  %-10s  %-10s  %-12s  %-10s\n", "------", "------", "-----", "------", "-------")
	for _, s := range stats {
		speedup := float64(baseline) / float64(s.median)
		fmt.Printf("  %-22s  %-10d  %-10d  %-12s  %-10s\n",
			strings.TrimSpace(s.cfg.name), s.cfg.concurrency, s.cfg.batchSize,
			s.median.Round(time.Millisecond), fmt.Sprintf("%.2fx", speedup))
	}
	fmt.Print("\n")
	fmt.Printf("  Baseline = %q. HTTP calls per pass: seq=%d, batched=%d.\n",
		strings.TrimSpace(stats[0].cfg.name), stats[0].httpCallsPer, stats[len(stats)-1].httpCallsPer)
}

func findStats(stats []runStats, name string) *runStats {
	for i := range stats {
		if stats[i].cfg.name == name {
			return &stats[i]
		}
	}
	return nil
}

// correctnessCheck samples n chunk indices and returns the maximum
// 1 - cosine_similarity between the two sets of embeddings. A value of 0
// means perfectly identical directions; values up to ~1e-6 are normal
// floating-point noise.
func correctnessCheck(a, b [][]float64, n int) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	if n > len(a) {
		n = len(a)
	}

	// Use a small RNG with a fixed seed so the sample is reproducible.
	rng := rand.New(rand.NewPCG(1, 2))

	var maxDelta float64
	for k := 0; k < n; k++ {
		i := rng.IntN(len(a))
		sim := cosineSimilarity(a[i], b[i])
		delta := 1 - sim
		fmt.Printf("  chunk %3d: cosine_similarity=%.8f delta=%.8f\n", i, sim, delta)
		if delta > maxDelta {
			maxDelta = delta
		}
	}
	return maxDelta
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
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

func printResults(results []rag.SearchResult) {
	fmt.Print("\nTop matching chunks:\n\n")

	for i, r := range results {
		preview := r.Text
		if len(preview) > 120 {
			preview = preview[:120] + "..."
		}

		fmt.Printf("%d. %s (similarity: %.2f%%) %s\n\n", i+1, r.Name, r.Similarity*100, preview)
	}
}
