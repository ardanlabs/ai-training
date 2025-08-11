// This example shows you how to use MongoDB and Ollama to create a proper vector
// embedding database of the Ultimate Go Notebook. With this vector database,
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
//	$ make example6
//
// # This requires running the following command:
//
//	$ make compose-up // This starts MongoDB and OpenWebUI in docker compose.
//  $ make ollama-up  // This starts the Ollama service.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/tmc/langchaingo/llms/ollama"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type document struct {
	ID        int       `bson:"id"`
	Text      string    `bson:"text"`
	Embedding []float32 `bson:"embedding"`
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := createEmbeddings(); err != nil {
		return fmt.Errorf("createEmbeddings: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	col, err := setupDatabase(ctx)
	if err != nil {
		return fmt.Errorf("setupDatabase: %w", err)
	}

	if err := insertEmbeddings(ctx, col); err != nil {
		return fmt.Errorf("insertEmbeddings: %w", err)
	}

	return nil
}

func createEmbeddings() error {

	// If the embeddings already exist, we don't need to do this again.
	if _, err := os.Stat("zarf/data/book.embeddings"); err == nil {
		return nil
	}

	// Open a connection with ollama to access the model.
	llm, err := ollama.New(
		ollama.WithModel("mxbai-embed-large"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	// Open the book file with the pre-processed chunks.
	input, err := os.Open("zarf/data/book.chunks")
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer input.Close()

	// Create the embeddings.
	output, err := os.Create("zarf/data/book.embeddings")
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer output.Close()

	fmt.Print("\n")
	fmt.Print("\033[s")

	data, err := io.ReadAll(input)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	r := regexp.MustCompile(`<CHUNK>[\w\W]*?<\/CHUNK>`)
	chunks := r.FindAllString(string(data), -1)

	// Read one chunk at a time (each line) and get the vector embedding.
	for counter, chunk := range chunks {
		fmt.Print("\033[u\033[K")
		fmt.Printf("Vectorizing Data: %d of %d", counter, len(chunks))

		chunk := strings.Trim(chunk, "<CHUNK>")
		chunk = strings.Trim(chunk, "</CHUNK>")

		// Get the vector embedding for this chunk.
		embedding, err := llm.CreateEmbedding(context.Background(), []string{chunk})
		if err != nil {
			return fmt.Errorf("create embedding: %w", err)
		}

		// Create the document with the vector embedding.
		doc := document{
			ID:        counter,
			Text:      chunk,
			Embedding: embedding[0],
		}

		// Convert to json.
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

func setupDatabase(ctx context.Context) (*mongo.Collection, error) {

	// Connect to mongodb.
	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return nil, fmt.Errorf("connectToMongo: %w", err)
	}

	const dbName = "example5"
	const collectionName = "book"

	db := client.Database(dbName)

	// Create database and collection.
	col, err := mongodb.CreateCollection(ctx, db, collectionName)
	if err != nil {
		return nil, fmt.Errorf("createCollection: %w", err)
	}

	fmt.Println("Created Collection")

	const indexName = "vector_index"
	settings := mongodb.VectorIndexSettings{
		NumDimensions: 1024,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	// Create vector index.
	if err := mongodb.CreateVectorIndex(ctx, col, indexName, settings); err != nil {
		return nil, fmt.Errorf("createVectorIndex: %w", err)
	}

	fmt.Println("Created Vector Index")

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "id", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}

	// Create a unique index for the document.
	col.Indexes().CreateOne(ctx, indexModel)

	fmt.Println("Created Unique Index")

	return col, nil
}

func insertEmbeddings(ctx context.Context, col *mongo.Collection) error {

	// Open the book file with the pre-processed chunks.
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

		// Decode json to a go struct.
		var d document
		if err := json.Unmarshal([]byte(doc), &d); err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		// Check if this document is already in the database.
		res := col.FindOne(ctx, bson.D{{Key: "id", Value: d.ID}})
		if res.Err() == nil {
			continue
		}

		// We had an error the fact there are no documents.
		if !errors.Is(res.Err(), mongo.ErrNoDocuments) {
			return fmt.Errorf("find: %w", err)
		}

		// Insert the document into mongodb.
		if _, err := col.InsertOne(ctx, d); err != nil {
			return fmt.Errorf("insert: %w", err)
		}
	}

	fmt.Print("\n")

	return nil
}
