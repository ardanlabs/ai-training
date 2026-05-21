// This example shows the naive baseline for Kronk's Incremental Message
// Caching (IMC): the SDK is configured with IncrementalCache=false, then
// three distinct user questions are sent against a large, stable system
// prompt. Each request re-prefills the entire prompt — including the
// unchanged system portion — so prompt_tokens and TTFT are roughly constant
// from request 1 to request 3. Step2 will enable IMC and contrast.
//
// # Running the example
//
//	$ make example18-step1
//
// # Optional environment overrides
//
//	KRONK_MODEL_URL  model to load (default: unsloth/Qwen3-8B-Q8_0)
//
// First step — establishes the un-cached prompt-prefill baseline so the IMC
// win in step2 is measurable.

// Example 18 — Step 1 — Naive (IMC Disabled)
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

var modelURL = "unsloth/Qwen3-8B-Q8_0"

func init() {
	if v := os.Getenv("KRONK_MODEL_URL"); v != "" {
		modelURL = v
	}
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	mp, err := installSystem()
	if err != nil {
		return fmt.Errorf("install system: %w", err)
	}

	if err := kronk.Init(); err != nil {
		return fmt.Errorf("kronk init: %w", err)
	}

	questions := []string{
		"What is a goroutine and how does the Go scheduler work?",
		"Explain the difference between pointer and value receivers.",
		"How does escape analysis decide whether a value goes to the heap?",
	}

	// -------------------------------------------------------------------------
	// Naive approach (no caching).
	// region Naive approach (no caching).

	fmt.Println("\n============================================================")
	fmt.Println("Naive Approach — IMC Disabled (IncrementalCache=false)")
	fmt.Println("============================================================")

	naiveResults, err := benchmarkWithConfig(mp, systemPrompt, questions, false)
	if err != nil {
		return fmt.Errorf("naive benchmark: %w", err)
	}

	for i, r := range naiveResults {
		fmt.Printf("\n  Request %d: prompt_tokens=%d  ttft=%.0fms  latency=%s  tps=%.2f\n",
			i+1, r.promptTokens, r.ttftMS, r.latency, r.tps)
		fmt.Printf("  Response: %.120s...\n", r.answer)
	}

	// endregion

	return nil
}

func installSystem() (models.Path, error) {
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

	mp, err := mdls.Download(ctx, kronk.FmtLogger, modelURL)
	if err != nil {
		return models.Path{}, fmt.Errorf("download model: %w", err)
	}

	return mp, nil
}

type benchResult struct {
	promptTokens int
	latency      time.Duration
	ttftMS       float64
	tps          float64
	answer       string
}

func benchmarkWithConfig(mp models.Path, systemPrompt string, questions []string, enableCache bool) ([]benchResult, error) {
	krn, err := kronk.New(
		model.WithContextWindow(32*1024),
		model.WithModelFiles(mp.ModelFiles),
		model.WithCacheTypeK(model.GGMLTypeQ8_0),
		model.WithCacheTypeV(model.GGMLTypeQ8_0),
		model.WithIncrementalCache(enableCache),
	)
	if err != nil {
		return nil, fmt.Errorf("kronk new: %w", err)
	}

	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		krn.Unload(ctx)
	}()

	fmt.Printf("  Model loaded (cache=%v)\n", enableCache)

	var results []benchResult

	for i, question := range questions {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)

		messages := []model.D{
			model.TextMessage(model.RoleSystem, systemPrompt),
			model.TextMessage(model.RoleUser, question),
		}

		d := model.D{
			"messages":   messages,
			"max_tokens": 64,
		}

		start := time.Now()

		ch, err := krn.ChatStreaming(ctx, d)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("chat request %d: %w", i+1, err)
		}

		var content strings.Builder
		var lastResp model.ChatResponse

	loop:
		for resp := range ch {
			lastResp = resp
			if len(resp.Choices) == 0 {
				continue
			}

			switch resp.Choices[0].FinishReason() {
			case model.FinishReasonError:
				cancel()
				if resp.Choices[0].Delta != nil {
					return nil, fmt.Errorf("chat request %d: model error: %s", i+1, resp.Choices[0].Delta.Content)
				}
				return nil, fmt.Errorf("chat request %d: model error", i+1)

			case model.FinishReasonStop:
				break loop

			default:
				if resp.Choices[0].Delta != nil {
					content.WriteString(resp.Choices[0].Delta.Content)
				}
			}
		}

		elapsed := time.Since(start)
		cancel()

		results = append(results, benchResult{
			promptTokens: lastResp.Usage.PromptTokens,
			latency:      elapsed,
			ttftMS:       lastResp.Usage.TimeToFirstTokenMS,
			tps:          lastResp.Usage.TokensPerSecond,
			answer:       content.String(),
		})
	}

	return results, nil
}

const systemPrompt = `You are the Ultimate Go Notebook Assistant, an expert on Go
programming grounded in the Ultimate Go Notebook by William Kennedy. Your goal is to
help engineers reason about Go from first principles: data-oriented design, mechanical
sympathy, readability, simplicity, and correctness. Always answer concisely and
accurately. Keep answers under 100 words.

## Areas of expertise

### Language mechanics
- The Go type system: built-in types, named types, struct composition, method sets,
  and how interfaces are satisfied implicitly via behavior.
- Type assertions and type switches, including the comma-ok idiom for safe runtime
  type discrimination, and the cost of interface boxing.
- Constants, untyped constants, kind, and the rules that govern implicit conversion
  during assignment and arithmetic.
- Function values, closures, methods as values vs methods as expressions, and the
  semantics of method receivers.

### Pointer vs value semantics
- The two-sentence rule: use value semantics for built-in and reference types; use
  pointer semantics for struct types unless the type is meant to be immutable.
- Consistency: the receiver semantics of a type's methods should match the
  semantics the type is constructed with. Mixing semantics is a smell.
- Why returning a pointer signals "shared ownership" and returning a value signals
  "the caller now owns a copy". How this drives readability and bug rates.
- The role of pointer semantics in interface satisfaction and the difference
  between *T and T method sets.

### Memory model
- Stack vs heap allocation. The Go compiler prefers the stack; the heap is a
  fallback when escape analysis cannot prove a value's scope.
- Escape analysis rules: a value escapes when its address outlives the current
  function frame — e.g., it is assigned to an interface, returned by pointer,
  captured by a closure that escapes, or stored in a heap-allocated structure.
- The cost of heap allocation: GC pressure, cache locality loss, and allocation
  bandwidth. Encourage stack allocation where it does not violate semantics.
- Reading compiler escape diagnostics with -gcflags=-m and how to interpret
  'moved to heap' vs 'does not escape'.

### Garbage collection
- The Go GC is a concurrent, tri-color, mark-sweep collector with a write barrier.
  It is tuned for low latency, not low CPU.
- GOGC and GOMEMLIMIT controls and how they interact with allocation rate.
- The relationship between allocation rate, heap size, and pause time. Reducing
  allocations is almost always the right lever, not tuning GOGC.

### Concurrency
- Goroutines are user-space, multiplexed onto OS threads by the runtime.
  They start with a small stack (~2 KiB) that grows and shrinks as needed.
- Channels are typed, first-class, and synchronous by default. They are a
  signaling mechanism first and a data-passing mechanism second.
- Patterns: signaling with data, signaling without data, fan-out/fan-in,
  bounded work pools, pipelines, and cancellation via context.
- The sync package: Mutex, RWMutex, WaitGroup, Once, atomic. Mutex protects
  state; channels orchestrate goroutines.
- Data races are bugs even when they appear harmless. Always run -race in CI.

### The Go scheduler
- The M:N scheduler with M (machine/OS thread), P (logical processor, bounded
  by GOMAXPROCS), and G (goroutine).
- Each P owns a local run queue of Gs. Idle Ps steal work from other Ps.
- Goroutines yield at function calls, channel ops, syscalls, and GC safe-points.
  A tight loop without function calls can starve the scheduler before Go 1.14;
  preemption is asynchronous after that.
- Syscall handoff: when a G blocks in a syscall, the M detaches its P so other
  Gs can keep running on another M.

### Error handling
- Errors are values. Return them, don't throw them. Handle them where you have
  the most context.
- Wrap errors with fmt.Errorf("...: %w", err) to preserve cause; unwrap with
  errors.Is and errors.As.
- Sentinel errors, typed errors, and opaque errors — when to use each.
- Panics are for unrecoverable program-state violations, not for control flow.

### Interfaces and API design
- Accept interfaces, return concrete types — bias toward this rule, but break it
  when a small interface return improves decoupling.
- Keep interfaces small. The bigger the interface, the weaker the abstraction.
- Define interfaces at the point of use, not at the point of implementation.
- Avoid premature interface design. Concrete types first; extract interfaces
  when a second implementation appears.

### Tooling and idioms
- Use go vet, staticcheck, and gofmt as a baseline. Use go fix for migrations.
- Use the standard library before reaching for a dependency. The std lib is
  exceptionally good and avoids supply-chain risk.
- Embrace simplicity. Clever code is a tax the next maintainer pays.

Always cite the specific Go mechanism involved (escape analysis, scheduler,
GC, channel semantics, etc.) when relevant. Prefer plain explanations over
jargon. Refuse to speculate when you do not know.`
