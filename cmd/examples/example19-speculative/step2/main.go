// This example takes the baseline run from step1 and adds a second run with a
// draft model attached. With speculative decoding active, the small draft
// model proposes candidate tokens that the larger target model verifies in a
// single forward pass; when the draft frequently agrees with the target,
// throughput improves with identical output quality.
//
// Requirements:
//   - Draft and target models must share the same tokenizer/vocabulary.
//   - NSeqMax must be 1 (single-slot mode) when speculative decoding is active.
//
// # Running the example
//
//	$ make example19
//
// # Optional environment overrides
//
//	KRONK_MODEL_URL        target model (default: unsloth/Qwen3-8B-Q8_0)
//	KRONK_DRAFT_MODEL_URL  draft model (default: unsloth/Qwen3-0.6B-Q8_0)
//
// # What this step adds over step1
//
//	Second benchmark with the draft model attached, plus a side-by-side
//	comparison of latency, tokens/sec, and draft acceptance rate.

// Example 19 — Step 2 — Speculative + Comparison
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

	// Step 02
	draftModelURL = "unsloth/Qwen3-0.6B-Q8_0"
)

func init() {
	if v := os.Getenv("KRONK_MODEL_URL"); v != "" {
		targetModelURL = v
	}

	// Step 02
	if v := os.Getenv("KRONK_DRAFT_MODEL_URL"); v != "" {
		draftModelURL = v
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

	// Step 02
	draftMP, err := installDraft()
	if err != nil {
		return fmt.Errorf("install draft: %w", err)
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

	// -------------------------------------------------------------------------
	// Step 02
	// 2) Speculative decoding.
	// region Step 02

	fmt.Println("\n============================================================")
	fmt.Println("2) Speculative Decoding — Draft Model Attached")
	fmt.Println("============================================================")

	draftCfg := &model.DraftModelConfig{
		ModelFiles: draftMP.ModelFiles,
		NDraft:     5,
	}

	speculative, err := benchmark(targetMP, draftCfg, prompt)
	if err != nil {
		return fmt.Errorf("speculative benchmark: %w", err)
	}

	fmt.Printf("\n  Latency: %s\n", speculative.latency.Truncate(time.Millisecond))
	fmt.Printf("  TPS:     %.2f tokens/sec\n", speculative.tps)
	fmt.Printf("  Tokens:  %d prompt, %d output (%d answer + %d reasoning)\n",
		speculative.promptTokens, speculative.outputTokens, speculative.completionTokens, speculative.reasoningTokens)
	fmt.Printf("  Draft:   %d drafted, %d accepted (%.1f%% acceptance)\n",
		speculative.draftTokens, speculative.draftAcceptedTokens, speculative.draftAcceptanceRate*100)
	fmt.Printf("  Response:  %.150s...\n", speculative.answer)

	// endregion

	// -------------------------------------------------------------------------
	// Step 02
	// 3) Comparison summary.
	// region Step 02

	fmt.Println("\n============================================================")
	fmt.Println("3) Comparison Summary")
	fmt.Println("============================================================")

	fmt.Printf("\n  %-22s  %-15s  %-15s\n", "Metric", "Baseline", "Speculative")
	fmt.Printf("  %-22s  %-15s  %-15s\n", "----------------------", "---------------", "---------------")
	fmt.Printf("  %-22s  %-15s  %-15s\n", "Latency",
		baseline.latency.Truncate(time.Millisecond),
		speculative.latency.Truncate(time.Millisecond))
	fmt.Printf("  %-22s  %-15.2f  %-15.2f\n", "Tokens/sec", baseline.tps, speculative.tps)
	fmt.Printf("  %-22s  %-15d  %-15d\n", "Prompt Tokens", baseline.promptTokens, speculative.promptTokens)
	fmt.Printf("  %-22s  %-15d  %-15d\n", "Output Tokens", baseline.outputTokens, speculative.outputTokens)
	fmt.Printf("  %-22s  %-15s  %-15s\n", "Draft Acceptance", "n/a",
		fmt.Sprintf("%.1f%% (%d/%d)", speculative.draftAcceptanceRate*100,
			speculative.draftAcceptedTokens, speculative.draftTokens))

	if speculative.tps > 0 && baseline.tps > 0 {
		speedup := speculative.tps / baseline.tps
		fmt.Printf("\n  Speedup: %.2f×\n", speedup)
	}

	// Step 02
	fmt.Println("\nSpeculative decoding uses a smaller draft model to propose")
	fmt.Println("candidate tokens that the target model verifies in parallel.")
	fmt.Println("The application code is identical — this is purely a config change.")

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

// Step 02
func installDraft() (models.Path, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Minute)
	defer cancel()

	mdls, err := models.New()
	if err != nil {
		return models.Path{}, fmt.Errorf("create models api: %w", err)
	}

	draftMP, err := mdls.Download(ctx, kronk.FmtLogger, draftModelURL)
	if err != nil {
		return models.Path{}, fmt.Errorf("download draft model: %w", err)
	}

	return draftMP, nil
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
