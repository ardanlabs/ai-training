// This example shows you how to use MongoDB and Kronk to perform a vector
// search for a user question. The search will return the top 5 chunks from
// the database. Then these chunks are sent to the Llama model to create a
// coherent response. You must run example05 first.
//
// # Running the example:
//
//	$ make example06
//
// # This requires running the following commands:
//
//  $ make compose-up
//  $ make kronk-up
//	$ make example05

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

var (
	urlChat    = "http://localhost:8080/v1/chat/completions"
	urlEmbed   = "http://localhost:8080/v1/embeddings"
	modelChat  = "Qwen3-8B-Q8_0"
	modelEmbed = "embeddinggemma-300m-qat-Q8_0"

	dbName  = "example06"
	colName = "book"
)

func init() {
	if v := os.Getenv("LLM_CHAT_SERVER"); v != "" {
		urlChat = v
	}

	if v := os.Getenv("LLM_EMBED_SERVER"); v != "" {
		urlEmbed = v
	}

	if v := os.Getenv("LLM_CHAT_MODEL"); v != "" {
		modelChat = v
	}

	if v := os.Getenv("LLM_EMBED_MODEL"); v != "" {
		modelEmbed = v
	}
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("\nAsk Bill a question about Go: ")

	question, _ := reader.ReadString('\n')
	if question == "" {
		return nil
	}

	fmt.Print("\n")

	results, err := vectorSearch(ctx, question)
	if err != nil {
		return fmt.Errorf("vectorSearch: %w", err)
	}

	if err := questionResponse(ctx, question, results); err != nil {
		return fmt.Errorf("questionResponse: %w", err)
	}

	return nil
}

func vectorSearch(ctx context.Context, question string) ([]searchResult, error) {
	llm := client.NewLLM(urlEmbed, modelEmbed)

	vector, err := llm.EmbedText(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}

	// -------------------------------------------------------------------------

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return nil, fmt.Errorf("mongodb.Connect: %w", err)
	}

	col := client.Database(dbName).Collection(colName)

	// -------------------------------------------------------------------------

	const limitResults = 2

	results, err := vectorDBSearch(ctx, col, vector, limitResults)
	if err != nil {
		return nil, fmt.Errorf("vectorDBSearch: %w", err)
	}

	return results, nil
}

func questionResponse(ctx context.Context, question string, results []searchResult) error {
	const prompt = `Use only the CONTEXT to answer the user's question.	
	
	If the CONTEXT doesn't provide enough context, say that you don't know.	
	
	Answer the question and provide additional helpful information.
	
	Responses should be properly formatted to be easily read.	
	
	CONTEXT:
	%s	
	
	QUESTION:
	%s
`

	var chunks strings.Builder

	for _, res := range results {
		if res.Score >= .70 {
			chunks.WriteString(res.Text)
			chunks.WriteString(".\n")

			// YOU WILL WANT TO KNOW HOW MANY TOKENS ARE CURRENTLY IN THE CHUNK
			// SO YOU DON'T EXCEED THE CONTEXT WINDOW (MAXIMUM TOKENS ALLOWED BY
			// THE MODEL). OUR CURRENT MODEL SUPPORTS 8192 TOKENS. THERE IS A
			// TIKTOKEN PACKAGE IN FOUNDATION TO HELP YOU WITH THIS.
		}
	}

	content := chunks.String()
	if content == "" {
		fmt.Println("Don't have enough information to provide an answer")
		return nil
	}

	finalPrompt := fmt.Sprintf(prompt, content, question)

	// -------------------------------------------------------------------------

	llm := client.NewLLM(urlChat, modelChat)

	ch, err := llm.ChatCompletionsSSE(ctx, finalPrompt)
	if err != nil {
		return fmt.Errorf("do: %w", err)
	}

	fmt.Print("Model Response:\n\n")

	for resp := range ch {
		fmt.Print(resp.Choices[0].Delta.Content)
	}

	return nil
}

// =============================================================================

type searchResult struct {
	ID        int       `bson:"id"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
	Score     float64   `bson:"score"`
}

func vectorDBSearch(ctx context.Context, col *mongo.Collection, vector []float64, limit int) ([]searchResult, error) {
	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":       "vector_index",
				"exact":       true,
				"path":        "embedding",
				"queryVector": vector,
				"limit":       limit,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"id":        1,
				"text":      1,
				"embedding": 1,
				"score": bson.M{
					"$meta": "vectorSearchScore",
				},
			}},
		},
	}

	cur, err := col.Aggregate(ctx, pipeline)
	if err != nil {
		return nil, fmt.Errorf("aggregate: %w", err)
	}
	defer cur.Close(ctx)

	var results []searchResult
	if err := cur.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("all: %w", err)
	}

	return results, nil
}
