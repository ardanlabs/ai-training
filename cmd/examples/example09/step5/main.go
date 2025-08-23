// This example takes step4 and shows you how to process a set of images
// from a location on disk and provide search capabilities.
//
// # Running the example:
//
//	$ make example9-step5
//
// # This requires running the following commands:
//
//	$ make ollama-up

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	urlChat     = "http://localhost:11434/v1/chat/completions"
	urlEmbed    = "http://localhost:11434/v1/embeddings"
	modelChat   = "qwen2.5vl:latest"
	modelEmbed  = "bge-m3:latest"
	dbName      = "example9"
	colName     = "images-5"
	dimensions  = 1024
	gallaryPath = "zarf/samples/gallery/"
)

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

	llmChat := client.NewLLM(urlChat, modelChat)
	llmEmbed := client.NewLLM(urlEmbed, modelEmbed)

	// -------------------------------------------------------------------------

	fmt.Println("\nConnecting to MongoDB")

	dbClient, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("connectToMongo: %w", err)
	}

	col, err := initDB(ctx, dbClient)
	if err != nil {
		return fmt.Errorf("db init: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Println("Saving images in DB")

	if err := saveImagesInDB(ctx, llmChat, llmEmbed, col); err != nil {
		return fmt.Errorf("loadImages: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Print("\nAsk questions about images (use 'ctrl-c' to quit)\n\n")

	for {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Question: ")

		question, _ := reader.ReadString('\n')
		if question == "" {
			return nil
		}

		// -------------------------------------------------------------------------

		fmt.Println("\nPerforming vector search:")

		searchResults, err := vectorSearch(ctx, llmEmbed, col, question)
		if err != nil {
			return fmt.Errorf("vectorSearch: %w", err)
		}

		for _, result := range searchResults {
			fmt.Printf("FileName[%s] Score[%.2f]\n", result.FileName, result.Score)
		}

		if err := questionResponse(ctx, llmChat, question, searchResults); err != nil {
			return fmt.Errorf("questionResponse: %w", err)
		}
	}
}

func saveImagesInDB(ctx context.Context, llmChat *client.LLM, llmEmbed *client.LLM, col *mongo.Collection) error {
	const prompt = `Describe the image. Be concise and accurate. Do not be overly
	verbose or stylistic. Make sure all the elements in the image are
	enumerated and described. Do not include any additional details. Keep
	the description under 200 words. At the end of the description, create
	a list of tags with the names of all the elements in the image. Do not
	output anything past this list.
	Encode the list as valid JSON, as in this example:
	[
		"tag1",
		"tag2",
		"tag3",
		...
	]
	Make sure the JSON is valid, doesn't have any extra spaces, and is
	properly formatted.`

	files, err := getFilesFromDirectory(gallaryPath)
	if err != nil {
		return fmt.Errorf("get files: %w", err)
	}

	for _, fileName := range files {
		fmt.Printf("\nProcessing image: %s\n", fileName)

		findRes := col.FindOne(ctx, bson.D{{Key: "file_name", Value: fileName}})
		if findRes.Err() == nil {
			fmt.Println("  - Image already exists")
			continue
		}

		image, mimeType, err := readImage(fileName)
		if err != nil {
			return fmt.Errorf("readImage: %w", err)
		}

		fmt.Println("  - Generating image description")

		results, err := llmChat.ChatCompletions(ctx, prompt, client.WithImage(mimeType, image))
		if err != nil {
			return fmt.Errorf("llmChat.ChatCompletions: %w", err)
		}

		fmt.Println("  - Generate embeddings for the image description")

		vector, err := llmEmbed.EmbedText(ctx, results)
		if err != nil {
			return fmt.Errorf("llmEmbed.EmbedText: %w", err)
		}

		fmt.Println("  - Inserting image description into the database")

		d1 := document{
			FileName:    fileName,
			Description: results,
			Embedding:   vector,
		}

		res, err := col.InsertOne(ctx, d1)
		if err != nil {
			return fmt.Errorf("col.InsertOne: %w", err)
		}

		fmt.Printf("  - Inserted db id: %s\n", res.InsertedID)
	}

	// We need to give mongodb some time to index the documents.
	// There is no way to know when this gets done.
	time.Sleep(time.Second)

	return nil
}

func getFilesFromDirectory(directoryPath string) ([]string, error) {
	var files []string

	err := filepath.Walk(directoryPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && (filepath.Ext(info.Name()) == ".jpg" || filepath.Ext(info.Name()) == ".jpeg" || filepath.Ext(info.Name()) == ".png") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk directory: %w", err)
	}

	return files, nil
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

	const prompt = `
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

	fmt.Print("\nModel Response:\n\n")

	for resp := range ch {
		fmt.Print(resp.Choices[0].Delta.Content)
	}

	fmt.Printf("\n\n")

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

	const indexName = "vector_index"

	settings := mongodb.VectorIndexSettings{
		NumDimensions: dimensions,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	if err := mongodb.CreateVectorIndex(ctx, col, indexName, settings); err != nil {
		return nil, fmt.Errorf("createVectorIndex: %w", err)
	}

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "id", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	col.Indexes().CreateOne(ctx, indexModel)

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
				"index":       "vector_index",
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
