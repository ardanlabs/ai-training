// This example builds on example12-step2 and shows you how to
// interact with the data via the database.
//
// # Running the example:
//
//	$ make example12-step3
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
//	$ make compose-up // This starts the Mongo service.

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// =============================================================================

type document struct {
	FileName    string    `bson:"file_name"`
	Description string    `bson:"description"`
	Embedding   []float64 `bson:"embedding"`
}

// =============================================================================

type searchResult struct {
	FileName    string    `bson:"file_name" json:"file_name"`
	Description string    `bson:"description" json:"image_description"`
	Embedding   []float64 `bson:"embedding" json:"-"`
	Score       float64   `bson:"score" json:"-"`
}

// =============================================================================

const (
	urlChat  = "http://localhost:11434/v1/chat/completions"
	urlEmbed = "http://localhost:11434/v1/embeddings"

	modelChat   = "gpt-oss:latest"
	modelVision = "qwen2.5vl:latest"
	modelEmbed  = "bge-m3:latest"

	dbName     = "example12"
	colName    = "step-3"
	dimensions = 1024

	similarityThreshold = 0.80
)

// =============================================================================

// The context window represents the maximum number of tokens that can be sent
// and received by the model. The default for Ollama is 8K. In the makefile
// it has been increased to 64K.
var contextWindow = 1024 * 8

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := ingestVideo(); err != nil {
		return fmt.Errorf("ingest video: %w", err)
	}

	if err := chatLoop(); err != nil {
		return fmt.Errorf("chat loop: %w", err)
	}

	return nil
}

func ingestVideo() error {
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()

	if err := processVideo(); err != nil {
		return fmt.Errorf("process video: %w", err)
	}

	const prompt = `
		Give me all the text in the image.
		Do not describe the image.
		Make sure that ALL the text is in the image.
		Make sure that you include any text editor/IDE tool too.
	
		Output the text in a valid JSON format MATCHING this format:
		{
			"text": "<Text that can be found in this image>"
		}

		If no text is found, output MUST BE:
		{
			"error": "NO TEXT FOUND"
		}

		Encode any special characters in the JSON output.
		
		Make sure that there's no extra whitespace or formatting, or markdown surrounding the json output.
		MAKE SURE THAT THE JSON IS VALID.
		DO NOT INCLUDE ANYTHING ELSE BUT THE JSON DOCUMENT IN THE RESPONSE.
`

	// -------------------------------------------------------------------------

	fmt.Println("\nConnecting to MongoDB")

	dbClient, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("connectToMongo: %w", err)
	}

	fmt.Println("Initializing Database")

	col, err := initDB(ctx, dbClient)
	if err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	// -------------------------------------------------------------------------

	llmChat := client.NewLLM(urlChat, modelVision)
	llmEmbed := client.NewLLM(urlEmbed, modelEmbed)

	// -------------------------------------------------------------------------

	files, err := getFilesFromDirectory("zarf/samples/videos/frames")
	if err != nil {
		return fmt.Errorf("get files from directory: %w", err)
	}

	previousEmbedding := make([]float64, dimensions)

	for _, fileName := range files {
		fmt.Printf("\nProcessing image: %s\n", fileName)

		findRes := col.FindOne(ctx, bson.D{{Key: "file_name", Value: fileName}})
		if findRes.Err() == nil {
			fmt.Println("  - Image already exists")

			// If the image already exists, we'll use the existing embedding as
			// a value for the "previousEmbedding" variable.
			d1 := document{}
			err := findRes.Decode(&d1)
			if err != nil {
				return fmt.Errorf(" - Failed to decode document from DB: %w", err)
			}

			// Assign the embedding of this document to the "previousEmbedding" variable.
			copy(previousEmbedding, d1.Embedding)

			continue
		}

		// -------------------------------------------------------------------------

		image, mimeType, err := readImage(fileName)
		if err != nil {
			return fmt.Errorf("read image: %w", err)
		}

		results, err := llmChat.ChatCompletions(ctx, prompt, client.WithImage(mimeType, image))
		if err != nil {
			return fmt.Errorf("chat completions: %w", err)
		}

		fmt.Printf("LLM RESPONSE: %s\n", results)

		// -------------------------------------------------------------------------

		if strings.Contains(results, "\"error\": \"NO TEXT FOUND\"") {
			fmt.Println("  - No text found in image")
			continue
		}

		// -------------------------------------------------------------------------

		fmt.Println("\nGenerating embeddings for the image description:")

		embedding, err := llmEmbed.EmbedText(ctx, results)
		if err != nil {
			return fmt.Errorf("llm.EmbedText: %w", err)
		}

		fmt.Printf("%v...%v\n", embedding[0:3], embedding[len(embedding)-3:])

		// -------------------------------------------------------------------------

		similarity := vector.CosineSimilarity(previousEmbedding, embedding)
		fmt.Printf("  - Image similarity compared to the previous image: %.6f\n", similarity)

		if similarity > similarityThreshold {
			fmt.Println("  - Image is similar to previous image")
			continue
		}

		copy(previousEmbedding, embedding)

		// -------------------------------------------------------------------------

		fmt.Println("\nInserting frame information into the database:")

		d1 := document{
			FileName:    fileName,
			Description: results,
			Embedding:   embedding,
		}

		res, err := col.InsertOne(ctx, d1)
		if err != nil {
			return fmt.Errorf("col.InsertOne: %w", err)
		}

		fmt.Printf("%s\n", res.InsertedID)

		// We need to give mongodb some time to index the document.
		// There is no way to know when this gets done.
		time.Sleep(time.Second)
	}

	// -------------------------------------------------------------------------

	fmt.Println("\nDONE")
	return nil
}

func processVideo() error {
	fmt.Println("Processing Video ...")
	defer fmt.Println("\nDONE Processing Video")

	ffmpegCommand := "ffmpeg -i zarf/samples/videos/training.mp4 -vf \"select='eq(pict_type,I)'\" -vsync vfr zarf/samples/videos/frames/frame-%03d.jpg"
	_, err := exec.Command("/bin/sh", "-c", ffmpegCommand).Output()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w", err)
	}

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
		Keys:    bson.D{{Key: "file_name", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	col.Indexes().CreateOne(ctx, indexModel)

	return col, nil
}

func chatLoop() error {
	// -------------------------------------------------------------------------
	// Declare a function that can accept user input which the agent will use
	// when it's the users turn.

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	// -------------------------------------------------------------------------
	// Construct the agent and get it started.

	agent, err := NewAgent(getUserMessage)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	return agent.Run(context.TODO())
}
