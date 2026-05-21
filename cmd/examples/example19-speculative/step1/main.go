// This example loads the target model without speculative decoding and runs
// a single code-generation prompt against it. The output records latency,
// tokens-per-second, and prompt/output token counts so step2 has a baseline
// to compare against.
//
// # Running the example
//
//	$ make example19-step1
//
// # Optional environment overrides
//
//	KRONK_MODEL_URL  target model (default: unsloth/Qwen3-8B-Q8_0)

// Example 19 — Step 1 — Baseline (No Speculative)
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
)

var (
	targetModelURL = "unsloth/Qwen3-8B-Q8_0"
)

func init() {
	if v := os.Getenv("KRONK_MODEL_URL"); v != "" {
		targetModelURL = v
	}
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	targetMP, err := installTarget()
	if err != nil {
		return fmt.Errorf("install target: %w", err)
	}

	if err := kronk.Init(); err != nil {
		return fmt.Errorf("kronk init: %w", err)
	}

	// A code-generation prompt plays to the draft model's strengths:
	// Go syntax is highly predictable token-by-token (keywords, braces,
	// receiver names, error-return idioms), so the small 0.6B draft model
	// agrees with the 8B target model much more often than on free-form
	// prose — which is exactly when speculative decoding shines.
	const prompt = "Write a complete, production-ready Go implementation of a thread-safe LRU cache " +
		"with Get, Put, and Delete methods. Include full doc comments on every exported identifier, " +
		"proper error handling, and a small example in a comment block. Do not omit anything."

	// -------------------------------------------------------------------------
	// 1) Baseline (no speculative).
	// region 1) Baseline (no speculative).

	fmt.Println("\n============================================================")
	fmt.Println("1) Baseline — No Speculative Decoding")
	fmt.Println("============================================================")

	baseline, err := benchmark(targetMP, nil, prompt)
	if err != nil {
		return fmt.Errorf("baseline benchmark: %w", err)
	}

	fmt.Printf("\n  Latency: %s\n", baseline.latency.Truncate(time.Millisecond))
	fmt.Printf("  TPS:     %.2f tokens/sec\n", baseline.tps)
	fmt.Printf("  Tokens:  %d prompt, %d output (%d answer + %d reasoning)\n",
		baseline.promptTokens, baseline.outputTokens, baseline.completionTokens, baseline.reasoningTokens)
	fmt.Printf("  Response:  %.150s...\n", baseline.answer)

	// endregion

	return nil
}

func installTarget() (models.Path, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Minute)
	defer cancel()

	l, err := libs.New(
		libs.WithVersion(defaults.LibVersion("")),
	)
	if err != nil {
		return models.Path{}, err
	}

	if _, err := l.Download(ctx, kronk.FmtLogger); err != nil {
		return models.Path{}, fmt.Errorf("install llama.cpp: %w", err)
	}

	mdls, err := models.New()
	if err != nil {
		return models.Path{}, fmt.Errorf("create models api: %w", err)
	}

	targetMP, err := mdls.Download(ctx, kronk.FmtLogger, targetModelURL)
	if err != nil {
		return models.Path{}, fmt.Errorf("download target model: %w", err)
	}

	return targetMP, nil
}

type benchResult struct {
	promptTokens     int
	completionTokens int // answer tokens only
	reasoningTokens  int // <think> tokens (0 when thinking is disabled)
	outputTokens     int // completion + reasoning
	latency          time.Duration
	tps              float64
	answer           string

	// Speculative-decoding metrics (zero in baseline run).
	draftTokens         int
	draftAcceptedTokens int
	draftAcceptanceRate float64
}

func benchmark(targetMP models.Path, draft *model.DraftModelConfig, prompt string) (benchResult, error) {
	opts := []model.Option{
		model.WithContextWindow(32 * 1024),
		model.WithModelFiles(targetMP.ModelFiles),
		model.WithCacheTypeK(model.GGMLTypeQ8_0),
		model.WithCacheTypeV(model.GGMLTypeQ8_0),
		model.WithNSeqMax(1),
	}
	if draft != nil {
		opts = append(opts, model.WithDraftModel(draft))
	}

	krn, err := kronk.New(opts...)
	if err != nil {
		return benchResult{}, fmt.Errorf("kronk new: %w", err)
	}

	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		krn.Unload(ctx)
	}()

	mode := "baseline"
	if draft != nil {
		mode = "speculative"
	}
	fmt.Printf("  Model loaded (mode=%s)\n", mode)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	messages := []model.D{
		model.TextMessage(model.RoleUser, prompt),
	}

	// Deterministic sampling (seed + temperature=0) ensures both the
	// baseline and speculative runs follow the same token path, so any
	// throughput difference is attributable to speculative decoding and
	// not to RNG-driven divergence in output length.
	//
	// enable_thinking=false disables Qwen3's <think>...</think> reasoning
	// phase. Reasoning content is free-form and unpredictable, which kills
	// draft-model acceptance. Skipping it focuses both runs on the
	// structured code output where the draft model agrees with the target.
	d := model.D{
		"messages":        messages,
		"max_tokens":      512,
		"temperature":     0.0,
		"seed":            42,
		"enable_thinking": false,
	}

	start := time.Now()

	ch, err := krn.ChatStreaming(ctx, d)
	if err != nil {
		return benchResult{}, fmt.Errorf("chat request: %w", err)
	}

	var content strings.Builder
	var lastResp model.ChatResponse

	for resp := range ch {
		lastResp = resp
		if len(resp.Choices) > 0 && resp.Choices[0].Delta != nil {
			content.WriteString(resp.Choices[0].Delta.Content)
		}
	}

	elapsed := time.Since(start)

	res := benchResult{
		latency: elapsed,
		answer:  content.String(),
	}
	if lastResp.Usage != nil {
		u := lastResp.Usage
		res.promptTokens = u.PromptTokens
		res.completionTokens = u.CompletionTokens
		res.reasoningTokens = u.ReasoningTokens
		res.outputTokens = u.OutputTokens
		res.tps = u.TokensPerSecond
		res.draftTokens = u.DraftTokens
		res.draftAcceptedTokens = u.DraftAcceptedTokens
		res.draftAcceptanceRate = u.DraftAcceptanceRate
	}
	return res, nil
}
