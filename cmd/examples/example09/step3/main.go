// This example takes step2 and shows you how to store the image details
// into a vector database for similarity searching.
//
// # Running the example:
//
//	$ make example9-step3
//
// # This requires running the following commands:
//
//	$ make ollama-up
//	$ make compose-up

package main

import (
	"context"
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

const (
	urlChat    = "http://localhost:11434/v1/chat/completions"
	urlEmbed   = "http://localhost:11434/v1/embeddings"
	modelChat  = "qwen2.5vl:latest"
	modelEmbed = "bge-m3:latest"
	imagePath  = "zarf/samples/gallery/roseimg.png"
	dbName     = "example9"
	colName    = "images-3"
	dimensions = 1024
)

// =============================================================================

type document struct {
	FileName    string    `bson:"file_name"`
	Description string    `bson:"description"`
	Embedding   []float64 `bson:"embedding"`
}

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

	// ---------------------------------------------------------------------

	fmt.Println("\nGenerating embeddings for the image description:")

	llm = client.NewLLM(urlEmbed, modelEmbed)

	vector, err := llm.EmbedText(ctx, results)
	if err != nil {
		return fmt.Errorf("llm.EmbedText: %w", err)
	}

	fmt.Printf("%v...%v\n", vector[0:3], vector[len(vector)-3:])

	// ---------------------------------------------------------------------

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

	// ---------------------------------------------------------------------

	fmt.Println("\nDONE")
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

// =============================================================================

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

	return col, nil
}
