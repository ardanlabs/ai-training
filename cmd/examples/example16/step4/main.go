// This example makes the previous two graph steps production-shaped. A graph
// reaches much of the corpus in two hops; neither step said which chunks matter.
//
// What this step adds:
//
//   - Ingest resumes per chunk.
//   - Entity names normalize into one node per subject.
//   - Hubs are not hops, so a subject half the book mentions carries nothing in.
//   - Every candidate is scored, then discounted by how it arrived.
//   - Context is filled to a token budget, not a document count.
//   - Answers cite chunk ids and the table prints why each chunk is there.
//
// # Running the example:
//
//	$ make compose-up
//	$ make example16-step4
//
// # Seeing what ingest built, without starting a chat:
//
//	$ make example16-step4-stats

package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
)

const (
	modelChatSource  = "unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL"
	modelEmbedSource = "ggml-org/embeddinggemma-300m-qat-Q8_0"
	chunksFile       = "zarf/data/book.chunks"
	dimensions       = 768

	// contextWindow has to be asked for: topK chunks plus every hop is several
	// times a plain vector prompt and does not fit the default 8192.
	contextWindow = 32768

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
	var showStats bool
	flag.BoolVar(&showStats, "stats", false, "report what ingest built and exit")
	flag.Parse()

	// Stats only reads the database, so no model is loaded to print six counts.
	if showStats {
		return runStats()
	}

	// -------------------------------------------------------------------------

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

	g, err := openGraph(context.Background())
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer g.Close()

	if err := loadData(context.Background(), g, krnEmbed, krnChat, chunksFile); err != nil {
		return fmt.Errorf("error loading data: %w", err)
	}

	ret, err := newRetriever(context.Background(), g, krnEmbed)
	if err != nil {
		return fmt.Errorf("error building retriever: %w", err)
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
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			return graphSearch(ctx, ret, messages)
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
			return fmt.Errorf("unable to perform chat: %w", err)
		}

		printSources(docs)
	}
}

func runStats() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	g, err := openGraph(ctx)
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer g.Close()

	s, err := graphStats(ctx, g, 20)
	if err != nil {
		return fmt.Errorf("error reading stats: %w", err)
	}

	s.print()

	return nil
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

	fmt.Println("- contextWindow:", krn.ModelConfig().ContextWindow())
	fmt.Println("- embeddings   :", krn.ModelInfo().IsEmbedModel)
	fmt.Println("- template     :", krn.ModelInfo().Template.FileName)

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

// graphSearch runs the pipeline and prints the table that explains the context.
// Hop column: "vector" matched the question, "NEXT of n" was split off a chunk
// that did, "via <subject>" arrived through the entity graph.
func graphSearch(ctx context.Context, ret *retriever, messages []model.D) ([]document, error) {
	fmt.Print("\n--- Retrieval ---\n")

	question := messages[len(messages)-1]["content"].(string)

	docs, err := ret.retrieve(ctx, question)
	if err != nil {
		return nil, err
	}

	for _, doc := range docs {
		fmt.Printf("  #%-3d %-20s score=%.4f  %-28s %s\n",
			doc.ID, doc.Hop, doc.Score, truncate(doc.Chapter, 28), preview(doc.Text, 60))
	}

	return docs, nil
}

// printSources repeats the chunk ids so a claim can be traced to its chunk.
func printSources(docs []document) {
	if len(docs) == 0 {
		return
	}

	fmt.Print("\u001b[90mSources: ")

	for i, doc := range docs {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("#%d (%s)", doc.ID, doc.Chapter)
	}

	fmt.Print("\u001b[0m\n")
}

// preview flattens a chunk onto one line for the console.
func preview(text string, width int) string {
	return truncate(strings.Join(strings.Fields(text), " "), width)
}

func truncate(text string, width int) string {
	if len(text) <= width {
		return text
	}

	return text[:width-3] + "..."
}

// addContextPrompt builds the final prompt. Chunks are labelled with id and
// chapter and the model is told to cite them, which makes an answer checkable.
func addContextPrompt(documents []document, messages []model.D) []model.D {
	const prompt = `
		- Use the following Context to answer the user's question.
		- Each context section is labelled with a source id like [#12].
		- Cite the source id inline for each claim you make.
		- If the context does not answer the question, say so plainly.
		- Responses should be properly formatted to be easily read.
		- Share code if code is presented in the context.
		- Do not include any information that is not present in the context.

		Context:

		%s

		Question: %s
		`

	var content strings.Builder
	for _, doc := range documents {
		fmt.Fprintf(&content, "[#%d] (%s)\n%s\n\n", doc.ID, doc.Chapter, doc.Text)
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
