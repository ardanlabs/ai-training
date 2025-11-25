// This example shows you a complete RAG application using DuckDB as an embedding
// DB and an embedding model to generate embeddings, and a chat model for
// answering a question using llamacpp directly via yzma and a native Go application.
//
// # Running the example:
//
//	$ make example13-step3

package main

import (
	"bufio"
	"context"
	"database/sql"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/duck"
	"github.com/ardanlabs/ai-training/cmd/examples/example13/install"
	"github.com/ardanlabs/kronk"
	"github.com/hybridgroup/yzma/pkg/download"
)

const (
	modelChatURL  = "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q8_0.gguf?download=true"
	modelEmbedURL = "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf?download=true"
	libPath       = "zarf/llamacpp"
	modelPath     = "zarf/models"
	dbPath        = "zarf/data/duck-ex13-step3.db" // ":memory:"
	chunksFile    = "zarf/data/book.chunks"
	dimentions    = 768
)

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelEmbedFile, modelChatFile, err := installSystem()
	if err != nil {
		return fmt.Errorf("unable to install system: %w", err)
	}

	embedding := true

	krnEmbed, err := newKronk(modelEmbedFile, 0, embedding)
	if err != nil {
		return fmt.Errorf("unable to create embedding model: %w", err)
	}
	defer krnEmbed.Unload()

	embedding = false

	const nBatch = 32 * 1024
	krnChat, err := newKronk(modelChatFile, nBatch, embedding)
	if err != nil {
		return fmt.Errorf("unable to create chat model: %w", err)
	}
	defer krnChat.Unload()

	// -------------------------------------------------------------------------

	db, err := duck.LoadData(dbPath, krnEmbed, dimentions, chunksFile)
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer db.Close()

	// -------------------------------------------------------------------------

	var messages []kronk.ChatMessage

	for {
		messages, err = userInput(messages)
		if err != nil {
			return fmt.Errorf("unable to get user input: %w", err)
		}

		// ---------------------------------------------------------------------

		docs, err := func() ([]duck.Document, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			docs, err := vectorSearch(ctx, krnEmbed, db, messages)
			if err != nil {
				return nil, fmt.Errorf("unable to get vector search results: %w", err)
			}

			return docs, nil
		}()

		if err != nil {
			return fmt.Errorf("unable to get vector search results: %w", err)
		}

		// ---------------------------------------------------------------------

		messages, err = func() ([]kronk.ChatMessage, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			ch, err := performChat(ctx, krnChat, addContextPrompt(docs, messages))
			if err != nil {
				return nil, fmt.Errorf("unable to perform chat: %w", err)
			}

			messages, err = modelResponse(krnChat, messages, ch)
			if err != nil {
				return nil, fmt.Errorf("unable to get model response: %w", err)
			}

			return messages, nil
		}()

		if err != nil {
			return fmt.Errorf("unable to perform chat: %w", err)
		}
	}
}

func installSystem() (string, string, error) {
	if err := install.LlamaCPP(libPath, download.CPU, true); err != nil {
		return "", "", fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelEmbedFile, err := install.Model(modelEmbedURL, modelPath)
	if err != nil {
		return "", "", fmt.Errorf("unable to install embedding model: %w", err)
	}

	modelChatFile, err := install.Model(modelChatURL, modelPath)
	if err != nil {
		return "", "", fmt.Errorf("unable to install chat model: %w", err)
	}

	return modelEmbedFile, modelChatFile, nil
}

func newKronk(modelFile string, nBatch int, embeddings bool) (*kronk.Kronk, error) {
	if err := kronk.Init(libPath, kronk.LogSilent); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	const concurrency = 1

	krn, err := kronk.New(concurrency, modelFile, "", kronk.ModelConfig{
		NBatch:     nBatch,
		Embeddings: embeddings,
	})
	if err != nil {
		return nil, fmt.Errorf("unable to create inference model: %w", err)
	}

	fmt.Println("- modelFile      :", krn.ModelName())
	fmt.Println("  - contextWindow:", krn.ModelConfig().ContextWindow)
	fmt.Println("  - embeddings   :", krn.ModelConfig().Embeddings)

	return krn, nil
}

func userInput(messages []kronk.ChatMessage) ([]kronk.ChatMessage, error) {
	fmt.Print("\nUSER> ")

	reader := bufio.NewReader(os.Stdin)

	userInput, err := reader.ReadString('\n')
	if err != nil {
		return messages, fmt.Errorf("unable to read user input: %w", err)
	}

	messages = append(messages, kronk.ChatMessage{
		Role:    "user",
		Content: userInput,
	})

	return messages, nil
}

func vectorSearch(ctx context.Context, krnEmbed *kronk.Kronk, db *sql.DB, messages []kronk.ChatMessage) ([]duck.Document, error) {
	fmt.Print("\n--- Vector Search ---\n\n")

	lastUserInput := messages[len(messages)-1].Content

	queryVector, err := krnEmbed.Embed(ctx, lastUserInput)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	if len(queryVector) == 0 {
		return nil, fmt.Errorf("empty query vector")
	}

	docs, err := duck.Search(db, queryVector, 5)
	if err != nil {
		return nil, fmt.Errorf("error searching database: %w", err)
	}

	for _, doc := range docs {
		fmt.Printf("Doc: %f: %s\n", doc.Similarity, strings.ReplaceAll(doc.Text, "\n", " ")[:100])
	}

	return docs, nil
}

func addContextPrompt(documents []duck.Document, messages []kronk.ChatMessage) []kronk.ChatMessage {
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

	var count int
	var content strings.Builder
	for _, doc := range documents {
		content.WriteString(fmt.Sprintf("%s\n%s\n", doc.Text, doc.Text))
		count++
		if count == 2 {
			break
		}
	}

	lastUserInput := messages[len(messages)-1].Content
	finalPrompt := fmt.Sprintf(prompt, content.String(), lastUserInput)

	messages = append(messages, kronk.ChatMessage{
		Role:    "user",
		Content: finalPrompt,
	})

	return messages
}

func performChat(ctx context.Context, krn *kronk.Kronk, messages []kronk.ChatMessage) (<-chan kronk.ChatResponse, error) {
	ch, err := krn.ChatStreaming(ctx, messages, kronk.Params{
		MaxTokens: 2048,
	})
	if err != nil {
		return nil, fmt.Errorf("chat streaming: %w", err)
	}

	return ch, nil
}

func modelResponse(krn *kronk.Kronk, messages []kronk.ChatMessage, ch <-chan kronk.ChatResponse) ([]kronk.ChatMessage, error) {
	fmt.Print("\nMODEL> ")

	var reasoning bool
	var lr kronk.ChatResponse

loop:
	for resp := range ch {
		switch resp.Choice[0].FinishReason {
		case kronk.FinishReasonStop:
			break loop

		case kronk.FinishReasonError:
			return messages, fmt.Errorf("error from model: %s", resp.Choice[0].Delta.Content)
		}

		if resp.Choice[0].Delta.Reasoning != "" {
			fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choice[0].Delta.Reasoning)
			reasoning = true
			continue
		}

		if reasoning {
			reasoning = false
			fmt.Print("\n\n")
		}

		fmt.Printf("%s", resp.Choice[0].Delta.Content)
		lr = resp
	}

	messages = append(messages, kronk.ChatMessage{
		Role:    "assistant",
		Content: lr.Choice[0].Delta.Content,
	})

	// -------------------------------------------------------------------------

	contextTokens := lr.Usage.InputTokens + lr.Usage.CompletionTokens
	contextWindow := krn.ModelConfig().ContextWindow
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("\n\n\u001b[90mInput: %d  Reasoning: %d  Completion: %d  Output: %d  Window: %d (%.0f%% of %.0fK) TPS: %.2f\u001b[0m\n",
		lr.Usage.InputTokens, lr.Usage.ReasoningTokens, lr.Usage.CompletionTokens, lr.Usage.OutputTokens, contextTokens, percentage, of, lr.Usage.TokensPerSecond)

	return messages, nil
}
