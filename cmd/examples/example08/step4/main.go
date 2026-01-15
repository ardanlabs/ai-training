// This example takes step3 and shows you how to search for an image based on
// its description.
//
// # Running the example:
//
//	$ make example08-step4
//
// # This requires running the following commands:
//
//	$ make kronk-up
//	$ make compose-up

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

var (
	urlChat    = "http://localhost:8080/v1/chat/completions"
	urlEmbed   = "http://localhost:8080/v1/embeddings"
	modelChat  = "Qwen2.5-VL-3B-Instruct-Q8_0"
	modelEmbed = "embeddinggemma-300m-qat-Q8_0"

	imagePath  = "zarf/samples/gallery/roseimg.png"
	dbName     = "example8"
	colName    = "images-4"
	dimensions = 768
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

type document struct {
	FileName    string    `bson:"file_name"`
	Description string    `bson:"description"`
	Embedding   []float64 `bson:"embedding"`
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

	// -------------------------------------------------------------------------

	fmt.Println("\nConnecting to MongoDB")

	dbClient, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("mongodb.Connect: %w", err)
	}

	fmt.Println("Initializing Database")

	col, err := initDB(ctx, dbClient)
	if err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	// -------------------------------------------------------------------------

	findRes := col.FindOne(ctx, bson.D{{Key: "file_name", Value: imagePath}})
	if findRes.Err() == nil {
		fmt.Println("Deleting existing image from database")
		_, err := col.DeleteOne(ctx, bson.D{{Key: "file_name", Value: imagePath}})
		if err != nil {
			return fmt.Errorf("col.DeleteOne: %w", err)
		}
	}

	// -------------------------------------------------------------------------

	fmt.Println("\nGenerating image description:")

	image, mimeType, err := readImage(imagePath)
	if err != nil {
		return fmt.Errorf("readImage: %w", err)
	}

	// -------------------------------------------------------------------------

	const prompt = `
		Describe the image and be concise and accurate keeping the description under 200 words.

		Do not be overly verbose or stylistic.

		Make sure all the elements in the image are enumerated and described.

		At the end of the description, create a list of tags with the names of all the
		elements in the image and do not output anything past this list.

		Encode the list as valid JSON, as in this example:
		["tag1","tag2","tag3",...]

		Make sure the JSON is valid, doesn't have any extra spaces, and is
		properly formatted.`

	llm := client.NewLLM(urlChat, modelChat)

	results, err := llm.ChatCompletions(ctx, prompt, client.WithImage(mimeType, image))
	if err != nil {
		return fmt.Errorf("llm.ChatCompletions: %w", err)
	}

	fmt.Printf("%s\n", results)

	// -------------------------------------------------------------------------

	fmt.Println("\nGenerating embeddings for the image description:")

	embedLLM := client.NewLLM(urlEmbed, modelEmbed)

	vector, err := embedLLM.EmbedText(ctx, results)
	if err != nil {
		return fmt.Errorf("llm.EmbedText: %w", err)
	}

	fmt.Printf("%v...%v\n", vector[0:3], vector[len(vector)-3:])

	// -------------------------------------------------------------------------

	fmt.Println("\nInserting image information into the database:")

	d1 := document{
		FileName:    imagePath,
		Description: results,
		Embedding:   vector,
	}

	res, err := col.InsertOne(ctx, d1)
	if err != nil {
		return fmt.Errorf("col.InsertOne: %w", err)
	}

	fmt.Printf("%s\n", res.InsertedID)

	// We need to give mongodb some time to index the document.
	// There is no way to know when this gets done.
	time.Sleep(time.Second)

	// -------------------------------------------------------------------------

	fmt.Println("\nAsk a single question about images:")

	question := "Do you have any images of roses?"
	fmt.Printf("%s\n", question)

	// -------------------------------------------------------------------------

	fmt.Println("\nPerforming vector search:")

	searchResults, err := vectorSearch(ctx, embedLLM, col, question)
	if err != nil {
		return fmt.Errorf("vectorSearch: %w", err)
	}

	for _, result := range searchResults {
		fmt.Printf("FileName[%s] Score[%.2f]\n", result.FileName, result.Score)
	}

	// -------------------------------------------------------------------------

	fmt.Println("\nProviding response")

	if err := questionResponse(ctx, llm, question, searchResults); err != nil {
		return fmt.Errorf("questionResponse: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Println("\n\nDONE")
	return nil
}

func readImage(fileName string) ([]byte, string, error) {
	data, err := os.ReadFile(fileName)
	if err != nil {
		return nil, "", fmt.Errorf("read file: %w", err)
	}

	switch mimeType := http.DetectContentType(data); mimeType {
	case "image/jpeg", "image/png":
		return data, mimeType, nil
	default:
		return nil, "", fmt.Errorf("unsupported file type: %s: filename: %s", mimeType, fileName)
	}
}

func questionResponse(ctx context.Context, llm *client.LLM, question string, results []searchResult) error {
	type searchResult struct {
		FileName    string `json:"file_name"`
		Description string `json:"image_description"`
	}

	fmt.Println("\nUsing these vectors:")

	var finalResults []searchResult

	for _, result := range results {
		if result.Score >= 0.75 {
			fmt.Printf("FileName[%s] Score[%.2f]\n", result.FileName, result.Score)
			finalResults = append(finalResults, searchResult{
				FileName:    result.FileName,
				Description: result.Description,
			})
		}
	}

	content, err := json.Marshal(finalResults)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	// -------------------------------------------------------------------------
	// Let's ask the LLM to provide a response

	prompt := `
		INSTRUCTIONS:
		
		- Use the following RESULTS to answer the user's question.

		- The data will be a JSON array with the following fields:
		
		[
			{
				"file_name":string,
				"image_description":string
			},
			{
				"file_name":string,
				"image_description":string
			}
		]

		- The response should be in a JSON array with the following fields:
		
		[
			{
				"status": string,
				"filename": string,
				"description": string
			},
			{
				"status": string,
				"filename": string,
				"description": string
			}
		]

		- If there are no RESULTS, provide this response:
		
		[
			{
				"status": "not found"
			}
		]

		- Do not change anything related to the file_name provided.
		- Only provide a brief description of the image.
		- Only provide a valid JSON response.

		RESULTS:
		
		%s
			
		QUESTION:
		
		%s
	`

	finalPrompt := fmt.Sprintf(prompt, string(content), question)

	ch, err := llm.ChatCompletionsSSE(ctx, finalPrompt)
	if err != nil {
		return fmt.Errorf("chat completions: %w", err)
	}

	fmt.Println("\nModel Response:")

	for resp := range ch {
		fmt.Print(resp.Choices[0].Delta.Content)
	}

	return nil
}

// =============================================================================

type searchResult struct {
	FileName    string    `bson:"file_name" json:"file_name"`
	Description string    `bson:"description" json:"image_description"`
	Embedding   []float64 `bson:"embedding" json:"-"`
	Score       float64   `bson:"score" json:"-"`
}

func initDB(ctx context.Context, client *mongo.Client) (*mongo.Collection, error) {
	db := client.Database(dbName)

	col, err := mongodb.CreateCollection(ctx, db, colName)
	if err != nil {
		return nil, fmt.Errorf("createCollection: %w", err)
	}

	const textIndexName = "vector_embedding_index"

	settings := mongodb.VectorIndexSettings{
		NumDimensions: dimensions,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	if err := mongodb.CreateVectorIndex(ctx, col, textIndexName, settings); err != nil {
		return nil, fmt.Errorf("createVectorIndex (text): %w", err)
	}

	return col, nil
}

func vectorSearch(ctx context.Context, llm *client.LLM, col *mongo.Collection, question string) ([]searchResult, error) {
	vector, err := llm.EmbedText(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("embed text: %w", err)
	}

	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":       "vector_embedding_index",
				"exact":       true,
				"path":        "embedding",
				"queryVector": vector,
				"limit":       5,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"file_name":   1,
				"description": 1,
				"embedding":   1,
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
