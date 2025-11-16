// This example shows you a complete RAG application used duckDB as an embedding
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
	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
	"github.com/hybridgroup/yzma/pkg/download"
)

var (
	modelChatURL  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true"
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
		log.Fatal(err)
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

	const concurrency = 1

	llmEmbed, err := llamacpp.New(concurrency, libPath, modelEmbedFile, llamacpp.Config{
		ContextWindow: 1024 * 32,
		Embeddings:    true,
	})
	if err != nil {
		return fmt.Errorf("unable to create embedding model: %w", err)
	}
	defer llmEmbed.Unload()

	llmChat, err := llamacpp.New(concurrency, libPath, modelChatFile, llamacpp.Config{
		ContextWindow: 1024 * 32,
	})
	if err != nil {
		return fmt.Errorf("unable to create chat model: %w", err)
	}
	defer llmChat.Unload()

	// -------------------------------------------------------------------------

	db, err := duck.LoadData(dbPath, llmEmbed, dimentions, chunksFile)
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
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()

			queryVector, err := llmEmbed.Embed(ctx, question)
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

		documents := make([]llamacpp.RankingDocument, len(docs))
		for i, doc := range docs {
			documents[i] = llamacpp.RankingDocument{Document: doc.Text, Embedding: doc.Embedding}
		}

		rankings, err := llmEmbed.Rerank(documents)
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

		msgs := []llamacpp.ChatMessage{
			{Role: "user", Content: finalPrompt},
		}

		err = func() error {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()

			ch, err := llmChat.ChatCompletions(ctx, msgs, llamacpp.Params{
				TopK: 1.0,
				TopP: 0.9,
				Temp: 0.7,
			})
			if err != nil {
				return fmt.Errorf("chat completions: %w", err)
			}

			for msg := range ch {
				if msg.Err != nil {
					return fmt.Errorf("error from model: %w", msg.Err)
				}
				fmt.Print(msg.Response)
			}
			return nil
		}()
		if err != nil {
			return err
		}

		fmt.Println()
	}
}
