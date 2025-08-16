// This example takes step4 and shows you how to process a set of images
// from a location on disk and provide search capabilities.
//
// # Running the example:
//
//	$ make example9-step5
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
package main

import (
	"bufio"
	"context"
	"encoding/base64"
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
	urlChat        = "http://localhost:11434/v1/chat/completions"
	urlEmbedding   = "http://localhost:11434/v1/embeddings"
	modelChat      = "qwen2.5vl:latest"
	modelEmbedding = "bge-m3:latest"
	dbName         = "example9"
	collectionName = "images-5"
	gallaryPath    = "cmd/samples/gallery/"
)

type document struct {
	FileName    string    `bson:"file_name"`
	Description string    `bson:"description"`
	Embedding   []float64 `bson:"embedding"`
}

type searchResult struct {
	FileName    string    `bson:"file_name" json:"file_name"`
	Description string    `bson:"description" json:"image_description"`
	Embedding   []float64 `bson:"embedding" json:"-"`
	Score       float64   `bson:"score" json:"-"`
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	// -------------------------------------------------------------------------

	cln := client.New(client.StdoutLogger)
	clnSSE := client.NewSSE[client.ChatSSE](client.StdoutLogger)

	// -------------------------------------------------------------------------

	mongoClient, err := initDatabase(dbName, collectionName)
	if err != nil {
		return fmt.Errorf("initDatabase: %w", err)
	}

	col := mongoClient.Database(dbName).Collection(collectionName)

	// -------------------------------------------------------------------------

	prompt := `Describe the image. Be concise and accurate. Do not be overly
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

	// -------------------------------------------------------------------------

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

		data, mimeType, err := readImage(fileName)
		if err != nil {
			return fmt.Errorf("read image: %w", err)
		}

		fmt.Println("  - Generating image description")

		dataBase64 := base64.StdEncoding.EncodeToString(data)

		d := client.D{
			"model": modelChat,
			"messages": []client.D{
				{
					"role": "user",
					"content": []client.D{
						{
							"type": "text",
							"text": prompt,
						},
						{
							"type": "image_url",
							"image_url": client.D{
								"url": fmt.Sprintf("data:%s;base64,%s", mimeType, dataBase64),
							},
						},
					},
				},
			},
			"temperature": 1.0,
			"top_p":       0.5,
			"top_k":       20,
		}

		var result client.Chat
		if err := cln.Do(ctx, http.MethodPost, urlChat, d, &result); err != nil {
			return fmt.Errorf("do: %w", err)
		}

		// -------------------------------------------------------------------------

		fmt.Println("  - Generate embeddings for the image description")

		d = client.D{
			"model":              modelEmbedding,
			"truncate":           true,
			"truncate_direction": "right",
			"input":              result.Choices[0].Message.Content,
		}

		// Get the vector embedding for this question.
		var resp client.Embedding
		if err := cln.Do(ctx, http.MethodPost, urlEmbedding, d, &resp); err != nil {
			return fmt.Errorf("do: %w", err)
		}

		vector := resp.Data[0].Embedding

		// -------------------------------------------------------------------------

		fmt.Println("  - Inserting image description into the database")

		d1 := document{
			FileName:    fileName,
			Description: result.Choices[0].Message.Content,
			Embedding:   vector,
		}

		res, err := col.InsertOne(ctx, d1)
		if err != nil {
			return fmt.Errorf("insert: %w", err)
		}

		fmt.Printf("  - Inserted db id: %s\n", res.InsertedID)
	}

	// We need to give mongodb some time to index the documents.
	// There is no way to know when this gets done.
	time.Sleep(time.Second)

	fmt.Print("\nAsk questions about images (use 'ctrl-c' to quit)\n\n")

	for {
		// -------------------------------------------------------------------------
		// Continue to the LLM search/query part after processing all files

		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Question: ")

		question, _ := reader.ReadString('\n')
		if question == "" {
			return nil
		}

		fmt.Print("\nSearching...\n\n")

		// -------------------------------------------------------------------------

		ctx, cancel := context.WithTimeout(ctx, 240*time.Second)
		defer cancel()

		results, err := vectorSearch(ctx, cln, col, question)
		if err != nil {
			return fmt.Errorf("vectorSearch: %w", err)
		}

		if err := questionResponse(ctx, clnSSE, question, results); err != nil {
			return fmt.Errorf("questionResponse: %w", err)
		}
	}
}

func initDatabase(dbName string, collectionName string) (*mongo.Client, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// -------------------------------------------------------------------------
	// Connect to mongo

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return nil, fmt.Errorf("connectToMongo: %w", err)
	}

	fmt.Println("\nConnected to MongoDB")

	// -------------------------------------------------------------------------
	// Create database and collection

	db := client.Database(dbName)

	col, err := mongodb.CreateCollection(ctx, db, collectionName)
	if err != nil {
		return nil, fmt.Errorf("createCollection: %w", err)
	}

	fmt.Println("Created Collection")

	// -------------------------------------------------------------------------
	// Create vector index

	const indexName = "vector_index"

	settings := mongodb.VectorIndexSettings{
		NumDimensions: 1024,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	if err := mongodb.CreateVectorIndex(ctx, col, indexName, settings); err != nil {
		return nil, fmt.Errorf("createVectorIndex: %w", err)
	}

	fmt.Println("Created Vector Index")

	// -------------------------------------------------------------------------
	// Apply a unique index just to be safe.

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "file_name", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	if _, err := col.Indexes().CreateOne(ctx, indexModel); err != nil {
		return nil, fmt.Errorf("createUniqueIndex: %w", err)
	}

	fmt.Println("Created Unique file_name Index")

	return client, nil
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

func vectorSearch(ctx context.Context, cln *client.Client, col *mongo.Collection, question string) ([]searchResult, error) {

	// -------------------------------------------------------------------------
	// Get the vector embedding for the question.

	d := client.D{
		"model":              modelEmbedding,
		"truncate":           true,
		"truncate_direction": "right",
		"input":              question,
	}

	// Get the vector embedding for this question.
	var resp client.Embedding
	if err := cln.Do(ctx, http.MethodPost, urlEmbedding, d, &resp); err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}

	vector := resp.Data[0].Embedding

	// -------------------------------------------------------------------------
	// We want to find the nearest neighbors from the question vector embedding.

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

	// -------------------------------------------------------------------------
	// Return and display the results.

	var results []searchResult
	if err := cur.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("all: %w", err)
	}

	return results, nil
}

func questionResponse(ctx context.Context, cln *client.SSEClient[client.ChatSSE], question string, results []searchResult) error {
	type searchResult struct {
		FileName    string `json:"file_name"`
		Description string `json:"image_description"`
	}

	var finalResults []searchResult

	fmt.Print("Data:\n\n")
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

	d := client.D{
		"model": modelChat,
		"messages": []client.D{
			{
				"role":    "user",
				"content": finalPrompt,
			},
		},
		"temperature": 1.0,
		"top_p":       0.5,
		"top_k":       20,
		"stream":      true,
	}

	ch := make(chan client.ChatSSE, 100)
	if err := cln.Do(ctx, http.MethodPost, urlChat, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	fmt.Print("\nModel Response:\n\n")

	for resp := range ch {
		fmt.Print(resp.Choices[0].Delta.Content)
	}

	fmt.Printf("\n\n")

	return nil
}
