// This example takes step4 and shows you how to process a set of images
// from a location on disk and provide search capabilities by text query.
//
// # Running the example:
//
//	$ make example28-step5
//
// # This requires running the following commands:
//
//	$ make kronk-up
//	$ make compose-up

package main

import (
	"bufio" // Step 05
	"context"
	"encoding/json" // Step 04
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath" // Step 05
	"strings"       // Step 05
	"time"          // Step 04

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb" // Step 03
	"go.mongodb.org/mongo-driver/v2/bson"                 // Step 03
	"go.mongodb.org/mongo-driver/v2/mongo"                // Step 03
)

var (
	urlChat    = "http://localhost:11435/v1/chat/completions"
	urlEmbed   = "http://localhost:11435/v1/embeddings" // Step 02
	modelChat  = "Qwen2.5-VL-3B-Instruct-Q8_0"
	modelEmbed = "embeddinggemma-300m-qat-Q8_0" // Step 02

	imagePath   = "zarf/samples/gallery/roseimg.png"
	dbName      = "example28"             // Step 03
	colName     = "example28"             // Step 03
	dimensions  = 768                     // Step 03
	galleryPath = "zarf/samples/gallery/" // Step 05
)

func init() {
	if v := os.Getenv("LLM_CHAT_SERVER"); v != "" {
		urlChat = v
	}

	// Step 02
	if v := os.Getenv("LLM_EMBED_SERVER"); v != "" {
		urlEmbed = v
	}

	if v := os.Getenv("LLM_CHAT_MODEL"); v != "" {
		modelChat = v
	}

	// Step 02
	if v := os.Getenv("LLM_EMBED_MODEL"); v != "" {
		modelEmbed = v
	}
}

// =============================================================================

// Step 03
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
	ctx := context.Background()

	// Step 03 -----------------------------------------------------------------

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

	// Step 03 -----------------------------------------------------------------

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

	// Step 02 -----------------------------------------------------------------

	fmt.Println("\nGenerating embeddings for the image description:")

	embedLLM := client.NewLLM(urlEmbed, modelEmbed)

	vector, err := embedLLM.EmbedText(ctx, results)
	if err != nil {
		return fmt.Errorf("llm.EmbedText: %w", err)
	}

	fmt.Printf("%v...%v\n", vector[0:3], vector[len(vector)-3:])

	// Step 03 -----------------------------------------------------------------

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

	// Step 04 -----------------------------------------------------------------
	// We need to give mongodb some time to index the document.
	// There is no way to know when this gets done.
	time.Sleep(time.Second)

	// Step 04 -----------------------------------------------------------------

	fmt.Println("\nAsk a single question about images:")

	question := "Do you have any images of roses?"
	fmt.Printf("%s\n", question)

	// Step 04 -----------------------------------------------------------------

	fmt.Println("\nPerforming vector search:")

	searchResults, err := vectorSearch(ctx, embedLLM, col, question)
	if err != nil {
		return fmt.Errorf("vectorSearch: %w", err)
	}

	for _, result := range searchResults {
		fmt.Printf("FileName[%s] Score[%.2f]\n", result.FileName, result.Score)
	}

	// Step 04 -----------------------------------------------------------------

	fmt.Println("\nProviding response")

	if err := questionResponse(ctx, llm, question, searchResults); err != nil {
		return fmt.Errorf("questionResponse: %w", err)
	}

	// Step 05 -----------------------------------------------------------------
	// Bulk-ingest every image in the gallery directory. saveImagesInDB skips
	// files that are already stored.

	fmt.Println("\nSaving all gallery images in DB")

	if err := saveImagesInDB(ctx, llm, embedLLM, col); err != nil {
		return fmt.Errorf("saveImagesInDB: %w", err)
	}

	// Step 05 -----------------------------------------------------------------
	// Interactive REPL — keep asking the user for questions and run the same
	// text vector search used above.

	fmt.Print("\nAsk questions about images (use 'ctrl-c' to quit)\n\n")

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("Question: ")

		q, _ := reader.ReadString('\n')
		q = strings.TrimSpace(q)
		if q == "" {
			break
		}

		fmt.Println("\nPerforming vector search using image description:")

		replResults, err := vectorSearch(ctx, embedLLM, col, q)
		if err != nil {
			return fmt.Errorf("vectorSearch: %w", err)
		}

		for _, result := range replResults {
			fmt.Printf("FileName[%s] Score[%.2f]\n", result.FileName, result.Score)
		}

		if err := questionResponse(ctx, llm, q, replResults); err != nil {
			return fmt.Errorf("questionResponse: %w", err)
		}

		fmt.Printf("\n\n")
	}

	// -------------------------------------------------------------------------

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

// Step 04
// questionResponse uses the search results to answer the user's question.
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
	// region Let's ask the LLM to provide a response

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

	// endregion

	return nil
}

// Step 03 =====================================================================

// Step 04
type searchResult struct {
	FileName    string    `bson:"file_name" json:"file_name"`
	Description string    `bson:"description" json:"image_description"`
	Embedding   []float64 `bson:"embedding" json:"-"`
	Score       float64   `bson:"score" json:"-"`
}

// Step 03
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

// Step 04
// vectorSearch embeds the question and runs a similarity search against the
// collection.
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

// Step 05 =====================================================================
// Bulk ingest + directory walker.

// Step 05
// saveImagesInDB walks the gallery directory, generates a description +
// embedding for each new image, and inserts it into the collection. Existing
// entries are skipped.
func saveImagesInDB(ctx context.Context, llm *client.LLM, embedLLM *client.LLM, col *mongo.Collection) error {
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

	files, err := getFilesFromDirectory(galleryPath)
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

		results, err := llm.ChatCompletions(ctx, prompt, client.WithImage(mimeType, image))
		if err != nil {
			return fmt.Errorf("llmChat.ChatCompletions: %w", err)
		}

		fmt.Println("  - Generate embeddings for the image description")

		vector, err := embedLLM.EmbedText(ctx, results)
		if err != nil {
			return fmt.Errorf("llm.EmbedText: %w", err)
		}

		// ---------------------------------------------------------------------

		fmt.Println("  - Inserting image information into the database")

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

// Step 05
// getFilesFromDirectory walks the directory and returns all .jpg/.jpeg/.png
// file paths.
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
