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
	"fmt"
	"log"
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
	log.Default().SetOutput(os.Stdout)

	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	if err := install.LlamaCPP(libPath, download.CPU, true); err != nil {
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelEmbedFile, err := install.Model(modelEmbedURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install embedding model: %w", err)
	}

	modelChatFile, err := install.Model(modelChatURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install chat model: %w", err)
	}

	// -------------------------------------------------------------------------

	if err := kronk.Init(libPath, kronk.LogSilent); err != nil {
		return fmt.Errorf("unable to init kronk: %w", err)
	}

	const concurrency = 1

	krnEmbed, err := kronk.New(concurrency, modelEmbedFile, "", kronk.ModelConfig{
		Embeddings: true,
	})
	if err != nil {
		return fmt.Errorf("unable to create embedding model: %w", err)
	}
	defer krnEmbed.Unload()

	fmt.Println("- embed contextWindow:", krnEmbed.ModelConfig().ContextWindow)
	fmt.Println("- embed maxTokens    :", krnEmbed.ModelConfig().MaxTokens)
	fmt.Println("- embed embeddings   :", krnEmbed.ModelConfig().Embeddings)

	krnChat, err := kronk.New(concurrency, modelChatFile, "", kronk.ModelConfig{})
	if err != nil {
		return fmt.Errorf("unable to create chat model: %w", err)
	}
	defer krnChat.Unload()

	fmt.Println("- chat contextWindow :", krnChat.ModelConfig().ContextWindow)
	fmt.Println("- chat maxTokens     :", krnChat.ModelConfig().MaxTokens)
	fmt.Println("- chat embeddings    :", krnChat.ModelConfig().Embeddings)

	// -------------------------------------------------------------------------

	db, err := duck.LoadData(dbPath, krnEmbed, dimentions, chunksFile)
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer db.Close()

	// -------------------------------------------------------------------------

	for {
		fmt.Print("\nQuestion> ")

		reader := bufio.NewReader(os.Stdin)

		question, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("unable to read user input", err.Error())
			os.Exit(1)
		}

		// ---------------------------------------------------------------------

		fmt.Print("\n-- Similarity ---\n\n")

		queryVector, err := func() ([]float32, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			queryVector, err := krnEmbed.Embed(ctx, question)
			if err != nil {
				return nil, fmt.Errorf("embed: %w", err)
			}

			return queryVector, nil
		}()
		if err != nil {
			return err
		}

		docs, err := duck.Search(db, queryVector, 5)
		if err != nil {
			return fmt.Errorf("error searching database: %w", err)
		}

		for _, doc := range docs {
			fmt.Printf("Doc: %f: %s\n", doc.Similarity, strings.ReplaceAll(doc.Text, "\n", " ")[:100])
		}

		// ---------------------------------------------------------------------

		fmt.Print("\n-- Rerank ---\n\n")

		documents := make([]kronk.RankingDocument, len(docs))
		for i, doc := range docs {
			documents[i] = kronk.RankingDocument{Document: doc.Text, Embedding: doc.Embedding}
		}

		rankings, err := krnEmbed.Rerank(documents)
		if err != nil {
			return fmt.Errorf("rerank: %w", err)
		}

		for _, ranking := range rankings {
			fmt.Printf("Doc: %f: %s\n", ranking.Score, strings.ReplaceAll(ranking.Document, "\n", " ")[:100])
		}

		// ---------------------------------------------------------------------

		fmt.Print("\n-- Response ---\n\n")

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

		var content string
		for _, ranking := range rankings[:2] {
			content = fmt.Sprintf("%s\n%s\n", content, ranking.Document)
		}

		finalPrompt := fmt.Sprintf(prompt, content, question)

		msgs := []kronk.ChatMessage{
			{Role: "user", Content: finalPrompt},
		}

		err = func() error {
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			ch, err := krnChat.ChatStreaming(ctx, msgs, kronk.Params{
				TopK: 1.0,
				TopP: 0.9,
				Temp: 0.7,
			})
			if err != nil {
				return fmt.Errorf("chat streaming: %w", err)
			}

			var contextTokens int
			var inputTokens int
			var outputTokens int

			for msg := range ch {
				if msg.Err != nil {
					return fmt.Errorf("error from model: %w", msg.Err)
				}

				fmt.Print(msg.Response)

				contextTokens = msg.Tokens.Context
				inputTokens = msg.Tokens.Input
				outputTokens += msg.Tokens.Output
			}

			contextWindow := krnChat.ModelConfig().ContextWindow
			percentage := (float64(contextTokens) / float64(contextWindow)) * 100
			of := float32(contextWindow) / float32(1024)

			fmt.Printf("\n\n\u001b[90mInput: %d  Output: %d  Context: %d (%.0f%% of %.0fK)\u001b[0m",
				inputTokens, outputTokens, contextTokens, percentage, of)

			return nil
		}()
		if err != nil {
			return err
		}

		fmt.Println()
	}
}
