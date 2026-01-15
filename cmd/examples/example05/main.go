// This example shows you how to use MongoDB and Kronk to create a proper vector
// embedding database for the Ultimate Go Notebook. With this vector database,
// you will be able to query for content that has a strong similarity to your
// question.
//
// The book has already been pre-processed into chunks based on the books TOC.
// For chunks over 500 words, those chunks have been chunked again into 250
// blocks. The code will create a vector embedding for each chunk.
// That data can be found under `zarf/data/book.chunks`.
//
// The original version of the book in text format has been retained. The program
// to clean that document into chunks can be found under `cmd/cleaner`. You can
// run that program using `make clean-data`. This is here if you want to play
// with your own chunking. How you chunk the data is critical to accuracy.
//
// # Running the example:
//
//	$ make example05
//
// # This requires running the following command:
//
//	$ make compose-up // This starts MongoDB and OpenWebUI in docker compose.
//  $ make kronk-up  // This starts the Kronk service.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

var (
	url   = "http://localhost:8080/v1/embeddings"
	model = "embeddinggemma-300m-qat-Q8_0"

	dbName     = "example06"
	colName    = "book"
	dimensions = 768
)

func init() {
	if v := os.Getenv("LLM_SERVER"); v != "" {
		url = v
	}

	if v := os.Getenv("LLM_MODEL"); v != "" {
		model = v
	}
}

// =============================================================================

type document struct {
	ID        int       `bson:"id"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	fmt.Println("\nCreating Embeddings")

	if err := createBookEmbeddings(ctx); err != nil {
		return fmt.Errorf("createBookEmbeddings: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Println("Initializing Database")

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("mongodb.Connect: %w", err)
	}

	col, err := initDB(ctx, client)
	if err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	// -------------------------------------------------------------------------

	if err := insertBookEmbeddings(ctx, col); err != nil {
		return fmt.Errorf("insertBookEmbeddings: %w", err)
	}

	fmt.Println("\nYou can now use example06 to ask questions about this content.")

	return nil
}

func createBookEmbeddings(ctx context.Context) error {
	llm := client.NewLLM(url, model)

	if _, err := os.Stat("zarf/data/book.embeddings"); err == nil {
		return nil
	}

	data, err := os.ReadFile("zarf/data/book.chunks")
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	output, err := os.Create("zarf/data/book.embeddings")
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer output.Close()

	fmt.Print("\n")
	fmt.Print("\033[s")

	r := regexp.MustCompile(`<CHUNK>[\w\W]*?<\/CHUNK>`)
	chunks := r.FindAllString(string(data), -1)

	// Read one chunk at a time (each line) and get the vector embedding.
	for counter, chunk := range chunks {
		fmt.Print("\033[u\033[K")
		fmt.Printf("Vectorizing Data: %d of %d", counter, len(chunks))

		chunk = strings.Trim(chunk, "<CHUNK>")
		chunk = strings.Trim(chunk, "</CHUNK>")

		// YOU WILL WANT TO KNOW HOW MANY TOKENS ARE CURRENTLY IN THE CHUNK
		// SO YOU DON'T EXCEED THE NUMBER OF TOKENS THE MODEL WILL USE TO
		// CREATE THE VECTOR EMBEDDING. THE MODEL WILL TRUNCATE YOUR CHUNK IF IT
		// EXCEEDS THE NUMBER OF TOKENS IT CAN USE TO CREATE THE VECTOR
		// EMBEDDING. THERE ARE MODELS THAT ONLY VECTORIZE AS LITTLE AS 512
		// TOKENS. THERE IS A TIKTOKEN PACKAGE IN FOUNDATION TO HELP YOU WITH
		// THIS.

		vector, err := llm.EmbedText(ctx, chunk)
		if err != nil {
			return fmt.Errorf("embedding: %w", err)
		}

		doc := document{
			ID:        counter,
			Text:      chunk,
			Embedding: vector,
		}

		data, err := json.Marshal(doc)
		if err != nil {
			return fmt.Errorf("marshal: %w", err)
		}

		// Write the json document to the embeddings file.
		if _, err := output.Write(data); err != nil {
			return fmt.Errorf("write: %w", err)
		}

		// Write a crlf for easier read access.
		if _, err := output.Write([]byte{'\n'}); err != nil {
			return fmt.Errorf("write crlf: %w", err)
		}
	}

	fmt.Print("\n")

	return nil
}

func insertBookEmbeddings(ctx context.Context, col *mongo.Collection) error {
	input, err := os.Open("zarf/data/book.embeddings")
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer input.Close()

	var counter int

	fmt.Print("\n")
	fmt.Print("\033[s")

	// Read one document at a time (each line) and insert into mongodb.
	scanner := bufio.NewScanner(input)
	for scanner.Scan() {
		counter++

		// Pull the next document from the file.
		doc := scanner.Text()

		fmt.Print("\033[u\033[K")
		fmt.Printf("Insering Data: %d", counter)

		var d document
		if err := json.Unmarshal([]byte(doc), &d); err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		res := col.FindOne(ctx, bson.D{{Key: "id", Value: d.ID}})
		if res.Err() == nil {
			continue
		}

		if !errors.Is(res.Err(), mongo.ErrNoDocuments) {
			return fmt.Errorf("find: %w", err)
		}

		if _, err := col.InsertOne(ctx, d); err != nil {
			return fmt.Errorf("insert: %w", err)
		}
	}

	fmt.Print("\n")

	return nil
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
