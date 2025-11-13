// This example shows you how to use duckDB as an embedding DB and an
// inference model to generate embeddings for a set of items all contained
// in a single Go binary.
//
// # Running the example:
//
//	$ make example13-step4
//
// # This requires running the following command:
//
//	$ make yzma-models // This downloads the needed models

package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
)

var (
	modelChatURL  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true"
	modelEmbedURL = "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf?download=true"
	libPath       = os.Getenv("YZMA_LIB")
	modelPath     = "zarf/models"
	dbPath        = "zarf/data/duck.db" // ":memory:"
	dimentions    = 768
)

func main() {
	log.Default().SetOutput(os.Stdout)

	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := llamacpp.InstallLibraries(libPath); err != nil {
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelEmbedFile, err := llamacpp.InstallModel(modelEmbedURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install embedding model: %w", err)
	}

	fmt.Println("- loading Embedding Model", modelEmbedFile)
	llmEmbed, err := llamacpp.New(libPath, modelEmbedFile, llamacpp.Config{
		ContextWindow: 1024 * 32,
		Embeddings:    true,
	})
	if err != nil {
		return fmt.Errorf("unable to create embedding model: %w", err)
	}
	defer llmEmbed.Unload()

	modelChatFile, err := llamacpp.InstallModel(modelChatURL, modelPath)
	if err != nil {
		return fmt.Errorf("unable to install chat model: %w", err)
	}

	fmt.Println("- loading Chat Model", modelChatFile)
	llmChat, err := llamacpp.New(libPath, modelChatFile, llamacpp.Config{
		ContextWindow: 1024 * 32,
	})
	if err != nil {
		return fmt.Errorf("unable to create chat model: %w", err)
	}
	defer llmChat.Unload()

	// -------------------------------------------------------------------------

	db, err := dbConnection(llmEmbed, dimentions)
	if err != nil {
		return fmt.Errorf("error connecting to database: %w", err)
	}
	defer db.Close()

	// -------------------------------------------------------------------------

	question := "Can you provide some examples of how to use interfaces in Go?"

	// -------------------------------------------------------------------------

	fmt.Print("\n-- Similarity ---\n\n")

	queryVector, err := llmEmbed.Embed(question)
	if err != nil {
		return fmt.Errorf("error embedding query: %w", err)
	}

	docs, err := dbSearch(db, queryVector, 5)
	if err != nil {
		return fmt.Errorf("error searching database: %w", err)
	}

	for _, doc := range docs {
		fmt.Printf("Doc: %f: %s\n", doc.Similarity, strings.ReplaceAll(doc.Text, "\n", " ")[:100])
	}

	// -------------------------------------------------------------------------

	fmt.Print("\n-- Rerank ---\n\n")

	documents := make([]llamacpp.RankingDocument, len(docs))
	for i, doc := range docs {
		documents[i] = llamacpp.RankingDocument{Document: doc.Text, Embedding: doc.Embedding}
	}

	rankings, err := llmEmbed.Rerank(documents)
	if err != nil {
		return fmt.Errorf("error reranking documents: %w", err)
	}

	for _, ranking := range rankings {
		fmt.Printf("Doc: %f: %s\n", ranking.Score, strings.ReplaceAll(ranking.Document, "\n", " ")[:100])
	}

	// -------------------------------------------------------------------------

	fmt.Print("\n-- Question ---\n\n")

	fmt.Println(question)

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

	var context string
	for _, ranking := range rankings[:2] {
		context = fmt.Sprintf("%s\n%s\n", context, ranking.Document)
	}

	finalPrompt := fmt.Sprintf(prompt, context, question)

	msgs := []llamacpp.ChatMessage{
		{Role: "user", Content: finalPrompt},
	}

	ch := llmChat.ChatCompletions(msgs, llamacpp.Params{
		TopK: 1.0,
		TopP: 0.9,
		Temp: 0.7,
	})

	for msg := range ch {
		if msg.Err != nil {
			return fmt.Errorf("error from model: %w", msg.Err)
		}
		fmt.Print(msg.Response)
	}

	fmt.Println()

	return nil
}
