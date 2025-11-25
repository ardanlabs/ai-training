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
	modelChatURL  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf?download=true"
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

	contextWindow := 32 * 1024
	embedding := true

	krnEmbed, err := newKronk(modelEmbedFile, contextWindow, embedding)
	if err != nil {
		return fmt.Errorf("unable to create embedding model: %w", err)
	}
	defer krnEmbed.Unload()

	contextWindow = 0
	embedding = false

	krnChat, err := newKronk(modelChatFile, contextWindow, embedding)
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

		rankings, err := rerank(krnChat, docs)
		if err != nil {
			return fmt.Errorf("unable to rerank: %w", err)
		}

		// ---------------------------------------------------------------------

		messages, err = func() ([]kronk.ChatMessage, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			ch, err := performChat(ctx, krnChat, addContextPrompt(rankings, messages))
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

func newKronk(modelFile string, contextWindow int, embeddings bool) (*kronk.Kronk, error) {
	if err := kronk.Init(libPath, kronk.LogSilent); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	const concurrency = 1

	krn, err := kronk.New(concurrency, modelFile, "", kronk.ModelConfig{
		ContextWindow: contextWindow,
		Embeddings:    embeddings,
	})
	if err != nil {
		return nil, fmt.Errorf("unable to create inference model: %w", err)
	}

	fmt.Println("- contextWindow:", krn.ModelConfig().ContextWindow)
	fmt.Println("- embeddings   :", krn.ModelConfig().Embeddings)

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

func rerank(krnEmbed *kronk.Kronk, docs []duck.Document) ([]kronk.Ranking, error) {
	fmt.Print("\n--- Rerank ---\n\n")

	documents := make([]kronk.RankingDocument, len(docs))
	for i, doc := range docs {
		documents[i] = kronk.RankingDocument{Document: doc.Text, Embedding: doc.Embedding}
	}

	rankings, err := krnEmbed.Rerank(documents)
	if err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}

	for _, ranking := range rankings {
		fmt.Printf("Doc: %f: %s\n", ranking.Score, strings.ReplaceAll(ranking.Document, "\n", " ")[:100])
	}

	return rankings, nil
}

func addContextPrompt(rankings []kronk.Ranking, messages []kronk.ChatMessage) []kronk.ChatMessage {
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
	for _, ranking := range rankings[:2] {
		content.WriteString(fmt.Sprintf("%s\n%s\n", ranking.Document, ranking.Document))
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

	var finalResponse strings.Builder
	var contextTokens int
	var inputTokens int
	var outputTokens int

	now := time.Now()

	for msg := range ch {
		if msg.Err != nil {
			return messages, fmt.Errorf("error from model: %w", msg.Err)
		}

		fmt.Print(msg.Response)
		finalResponse.WriteString(msg.Response)

		contextTokens = msg.Tokens.Context
		inputTokens = msg.Tokens.Input
		outputTokens += msg.Tokens.Output
	}

	// ---------------------------------------------------------------------

	elapsedSeconds := time.Since(now).Seconds()
	tokensPerSecond := float64(outputTokens) / elapsedSeconds

	contextWindow := krn.ModelConfig().ContextWindow
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("\n\n\u001b[90mInput: %d  Output: %d  Context: %d (%.0f%% of %.0fK) TPS: %.2f\u001b[0m\n",
		inputTokens, outputTokens, contextTokens, percentage, of, tokensPerSecond)

	messages = append(messages, kronk.ChatMessage{
		Role:    "assistant",
		Content: finalResponse.String(),
	})

	return messages, nil
}
