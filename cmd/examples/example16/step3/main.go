// This example adds inferred edges to the graph so retrieval can reach chunks
// that share a subject with the question but none of its wording. Apache AGE
// holds the graph, pgvector the embeddings, the Kronk SDK runs the models.
//
// Step2 hopped by page order. Step3 hops by shared subject:
//
//   - the chat model reads every chunk and reports its subjects
//   - each subject becomes an Entity node the chunk MENTIONS
//   - retrieval adds one entity hop to the NEXT hop out of every vector hit
//   - ingest pays one model call per chunk, a minute or two, run once and stored
//
// # Running the example:
//
//	$ make compose-up
//	$ make example16-step3

package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/age"
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
	topK             = 5

	// entityCap is how many chunks any single subject may contribute. Without
	// it the most mentioned subject in the book decides the whole context.
	entityCap = 2

	// contextWindow has to be asked for. The default 8192 cannot hold topK chunks
	// plus everything two hops add, the first cost of a graph.
	contextWindow = 32768

	// maxTokens covers reasoning plus answer, not just the answer. This model
	// spends 1500 tokens thinking before it writes, and kronk stops the moment
	// the two together hit the cap.
	maxTokens = 4096

	// maxContextDocs is a blunt cap on how many chunks reach the prompt. Chunk
	// lengths vary by 5x, so step4 replaces it with a token budget.
	maxContextDocs = 8
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

	g, err := openGraph(context.Background())
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer g.Close()

	if err := loadData(context.Background(), g, krnEmbed, krnChat, chunksFile); err != nil {
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

			return graphSearch(ctx, krnEmbed, g, messages)
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

				// Without this, a streaming call reports no usage at all and
				// the token accounting below never runs.
				"stream_options": model.D{
					"include_usage": true,
				},
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

// graphSearch performs the retrieval and prints how each chunk was reached. A
// "via <subject>" row came through an Entity node, not through the wording.
func graphSearch(ctx context.Context, krnEmbed *kronk.Kronk, g *age.Graph, messages []model.D) ([]document, error) {
	fmt.Print("\n--- Graph Search ---\n\n")

	question := messages[len(messages)-1]["content"].(string)

	docs, err := search(ctx, g, krnEmbed, question, topK)
	if err != nil {
		return nil, err
	}

	for _, doc := range docs {
		fmt.Printf("[%-18s sim=%.4f id=%-3d %s] %s\n",
			doc.Hop, doc.Similarity, doc.ID, doc.Chapter, preview(doc.Text, 70))
	}

	return docs, nil
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

	// Capped here, not in retrieval, so the search above still shows every hop.
	var content strings.Builder
	for i, doc := range documents {
		if i == maxContextDocs {
			fmt.Printf("\n\u001b[90m%d of %d retrieved chunks dropped to fit the prompt\u001b[0m\n",
				len(documents)-maxContextDocs, len(documents))
			break
		}

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

		// The usage report arrives as its own response carrying no choices, so
		// it has to be picked up before anything indexes into them.
		if len(resp.Choices) == 0 {
			continue
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
