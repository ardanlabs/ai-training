// This example is the baseline the rest of example16 argues with: plain vector
// RAG with no graph. One pgvector table, embed the question, take the topK
// nearest chunks, into the prompt, done. Embedding and chat models both run
// through the Kronk SDK.
//
// After every search it reports where similarity ranked the chunks adjacent to
// each hit, which of them fell outside topK, and what topK would cost to reach
// the closest one. That measurement is the argument for the next three steps.
//
// # Running the example:
//
//	$ make compose-up
//	$ make example16-step1

package main

import (
	"bufio"
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
	"github.com/jmoiron/sqlx"
)

const (
	modelChatSource  = "unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL"
	modelEmbedSource = "ggml-org/embeddinggemma-300m-qat-Q8_0"
	chunksFile       = "zarf/data/book.chunks"
	dimensions       = 768

	// topK is the only retrieval knob in this step. The report after each search
	// shows what raising it costs.
	topK = 5

	// contextWindow is raised from the default 8192: chunks in this book run past
	// three thousand characters and five of them plus the answer crowd it. Step3
	// and step4 double it again, since every hop adds a chunk.
	contextWindow = 16384

	// maxTokens covers reasoning plus answer, not just the answer. This model
	// spends 1500 tokens thinking before it writes, and kronk stops the moment
	// the two together hit the cap.
	maxTokens = 4096
)

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	infoEmbed, infoChat, err := installSystem()
	if err != nil {
		return fmt.Errorf("unable to install system: %w", err)
	}

	krnEmbed, err := newKronk(infoEmbed)
	if err != nil {
		return fmt.Errorf("unable to create embedding model: %w", err)
	}
	defer func() {
		fmt.Println("\nUnloading embedding model")
		if err := krnEmbed.Unload(context.Background()); err != nil {
			fmt.Printf("failed to unload embedding model: %v", err)
		}
	}()

	krnChat, err := newKronk(infoChat, model.WithContextWindow(contextWindow))
	if err != nil {
		return fmt.Errorf("unable to create chat model: %w", err)
	}
	defer func() {
		fmt.Println("\nUnloading chat model")
		if err := krnChat.Unload(context.Background()); err != nil {
			fmt.Printf("failed to unload chat model: %v", err)
		}
	}()

	// -------------------------------------------------------------------------

	db, err := openDB(context.Background())
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer db.Close()

	if err := loadData(context.Background(), db, krnEmbed, chunksFile); err != nil {
		return fmt.Errorf("error loading data: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Printf("Try something like: %q\n", "How does the scheduler handle blocking system calls?")

	var messages []model.D

	for {
		messages, err = userInput(messages)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return fmt.Errorf("unable to get user input: %w", err)
		}

		// ---------------------------------------------------------------------

		docs, err := func() ([]document, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			return vectorSearch(ctx, krnEmbed, db, messages)
		}()

		if err != nil {
			return fmt.Errorf("unable to get search results: %w", err)
		}

		// ---------------------------------------------------------------------

		err = func() error {
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			d := model.D{
				"messages":    addContextPrompt(docs, messages),
				"max_tokens":  maxTokens,
				"temperature": 0.7,
				"top_p":       0.9,
				"top_k":       40,
			}

			ch, err := krnChat.ChatStreaming(ctx, d)
			if err != nil {
				return fmt.Errorf("chat streaming: %w", err)
			}

			return modelResponse(krnChat, ch)
		}()

		if err != nil {
			return err
		}
	}
}

func installSystem() (models.Path, models.Path, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Minute)
	defer cancel()

	libs, err := libs.New(
		libs.WithVersion(defaults.LibVersion("")),
	)
	if err != nil {
		return models.Path{}, models.Path{}, err
	}

	if _, err := libs.Download(ctx, kronk.FmtLogger); err != nil {
		return models.Path{}, models.Path{}, fmt.Errorf("unable to install llama.cpp: %w", err)
	}

	// -------------------------------------------------------------------------

	mdls, err := models.New()
	if err != nil {
		return models.Path{}, models.Path{}, fmt.Errorf("unable to create models api: %w", err)
	}

	infoEmbed, err := mdls.Download(ctx, kronk.FmtLogger, modelEmbedSource)
	if err != nil {
		return models.Path{}, models.Path{}, fmt.Errorf("unable to install model: %w", err)
	}

	infoChat, err := mdls.Download(ctx, kronk.FmtLogger, modelChatSource)
	if err != nil {
		return models.Path{}, models.Path{}, fmt.Errorf("unable to install model: %w", err)
	}

	return infoEmbed, infoChat, nil
}

func newKronk(mp models.Path, opts ...model.Option) (*kronk.Kronk, error) {
	if err := kronk.Init(); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	krn, err := kronk.New(
		append([]model.Option{model.WithModelFiles(mp.ModelFiles)}, opts...)...,
	)

	if err != nil {
		return nil, fmt.Errorf("unable to create inference model: %w", err)
	}

	return krn, nil
}

func userInput(messages []model.D) ([]model.D, error) {
	fmt.Print("\nUSER> ")

	reader := bufio.NewReader(os.Stdin)

	userInput, err := reader.ReadString('\n')
	if err != nil {
		return messages, fmt.Errorf("unable to read user input: %w", err)
	}

	if userInput == "quit\n" {
		return nil, io.EOF
	}

	messages = append(messages, model.TextMessage("user", userInput))

	return messages, nil
}

// vectorSearch runs the retrieval and then prints what it missed. Only the
// retrieval is part of the application.
func vectorSearch(ctx context.Context, krnEmbed *kronk.Kronk, db *sqlx.DB, messages []model.D) ([]document, error) {
	question := messages[len(messages)-1]["content"].(string)

	res, err := search(ctx, db, krnEmbed, question, topK)
	if err != nil {
		return nil, err
	}

	fmt.Printf("\n--- Vector Search: top %d of %d chunks ---\n\n", topK, res.Corpus)

	for _, doc := range res.Docs {
		fmt.Printf("[rank=%-3d sim=%.4f id=%-3d %s] %s\n",
			doc.Rank, doc.Similarity, doc.ID, doc.Chapter, preview(doc.Text, 70))
	}

	reportMissed(res)

	return res.Docs, nil
}

// reportMissed prints where similarity ranked the chunks adjacent to the hits
// and what topK would have to be to include the best of them. Instrumentation
// only; nothing here changes the answer.
func reportMissed(res searchResult) {
	if len(res.Neighbors) == 0 {
		return
	}

	fmt.Print("\n--- What Similarity Did Not Reach ---\n\n")

	inPrompt := make(map[int]bool, len(res.Docs))
	for _, d := range res.Docs {
		inPrompt[d.ID] = true
	}

	var missed []neighbor

	for _, n := range res.Neighbors {
		label := "precedes"
		if n.After {
			label = "continues"
		}

		switch {
		case inPrompt[n.ID]:
			fmt.Printf("\u001b[90mid=%-3d %s id=%-3d rank=%-3d in the prompt\u001b[0m\n",
				n.ID, label, n.HitID, n.Rank)

		default:
			missed = append(missed, n)

			fmt.Printf("id=%-3d %s id=%-3d rank=%-3d \u001b[91mMISSED\u001b[0m\n",
				n.ID, label, n.HitID, n.Rank)
		}
	}

	if len(missed) == 0 {
		fmt.Print("\nEvery neighbor of every hit is in the prompt.\n")
		return
	}

	// -------------------------------------------------------------------------
	// The best-ranked missing neighbor sets how far topK would have to reach.
	// Every chunk ranked in between is bought along the way.

	best := slices.MinFunc(missed, func(a, b neighbor) int {
		return cmp.Compare(a.Rank, b.Rank)
	})

	fmt.Printf("\n%d of %d neighbors are not in the prompt. The closest is id=%d at rank %d.\n",
		len(missed), len(res.Neighbors), best.ID, best.Rank)

	switch extra := best.Rank - topK - 1; extra {
	case 0:
		fmt.Printf("topK would have to be %d to reach it.\n",
			best.Rank)

	default:
		fmt.Printf("topK would have to be %d to reach it, plus the %d chunks ranked between them.\n",
			best.Rank, extra)
	}

	// -------------------------------------------------------------------------
	// How concentrated the hits are. Passages that resemble the question's
	// wording tend to resemble each other and sit together, so a top-5 is often
	// four chunks from one chapter.

	counts := make(map[string]int, len(res.Docs))
	for _, d := range res.Docs {
		counts[d.Chapter]++
	}

	chapters := make([]string, 0, len(counts))
	for _, chapter := range slices.Sorted(maps.Keys(counts)) {
		chapters = append(chapters, fmt.Sprintf("%s (%d)", chapter, counts[chapter]))
	}

	fmt.Printf("Chapters represented: %s\n", strings.Join(chapters, ", "))
}

// preview flattens a chunk onto one line for the console.
func preview(text string, width int) string {
	text = strings.Join(strings.Fields(text), " ")
	if len(text) <= width {
		return text
	}

	return text[:width] + "..."
}

func addContextPrompt(documents []document, messages []model.D) []model.D {
	const prompt = `
		- Use the following Context to answer the user's question.
		- If you don't know the answer, say that you don't know.
		- Responses should be properly formatted to be easily read.
		- Share code if code is presented in the context.
		- Do not include any additional information not present in the context.

		Context:

		%s

		Question: %s
		`

	var content strings.Builder
	for _, doc := range documents {
		fmt.Fprintf(&content, "%s\n\n", doc.Text)
	}

	lastUserInput := messages[len(messages)-1]["content"].(string)
	finalPrompt := fmt.Sprintf(prompt, content.String(), lastUserInput)

	messages = append(messages, model.TextMessage("user", finalPrompt))

	return messages
}

// modelResponse streams the answer and reports what the turn cost. Nothing is
// added to the conversation history: every question retrieves on its own.
func modelResponse(krn *kronk.Kronk, ch <-chan model.ChatResponse) error {
	fmt.Print("\nMODEL> ")

	var reasoning bool
	var lr model.ChatResponse

	for resp := range ch {
		if resp.Usage != nil {
			lr = resp
		}

		switch resp.Choices[0].FinishReason() {
		case model.FinishReasonError:
			return fmt.Errorf("error from model: %s", resp.Choices[0].Delta.Content)

		case model.FinishReasonStop:
			continue

		default:
			if resp.Choices[0].Delta.Reasoning != "" {
				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
				reasoning = true
				continue
			}

			if reasoning {
				reasoning = false

				fmt.Println()
			}

			fmt.Printf("%s", resp.Choices[0].Delta.Content)
		}
	}

	// -------------------------------------------------------------------------

	if lr.Usage == nil {
		fmt.Println()
		return nil
	}

	contextTokens := lr.Usage.PromptTokens + lr.Usage.CompletionTokens
	contextWindow := krn.ModelConfig().ContextWindow()
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("\n\n\u001b[90mPrompt: %d  Reasoning: %d  Completion: %d  Window: %d (%.0f%% of %.0fK) TPS: %.2f\u001b[0m\n",
		lr.Usage.PromptTokens, lr.Usage.CompletionTokensDetails.ReasoningTokens, lr.Usage.CompletionTokens, contextTokens, percentage, of, lr.Usage.TokensPerSecond)

	// Hitting the cap ends the stream with a finish reason of "stop", exactly as a
	// finished answer does, so truncation is invisible unless the count is checked.
	if lr.Usage.CompletionTokens >= maxTokens {
		fmt.Printf("\u001b[91mTruncated: hit the %d token cap\u001b[0m\n", maxTokens)
	}

	return nil
}
