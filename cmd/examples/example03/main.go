// This example shows you how to use MongoDB as a vector database to perform
// a nearest neighbor vector search. The example will create a vector search
// index, store 2 documents, and perform a vector search.
//
// # Running the example:
//
//  $ make example03
//
// # This requires running the following command:
//
//	$ make compose-up
//
// # You can use this command to open a prompt to mongodb:
//
//  $ make mongo

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	dbName     = "example3"
	colName    = "book"
	dimensions = 768
)

var (
	url   = "http://localhost:8080/v1/embeddings"
	model = "embeddinggemma-300m-qat-Q8_0"
)

// =============================================================================

type document struct {
	ID        int       `bson:"id"`
	Name      string    `bson:"name"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
}

// Vector can convert the specified data into a vector.
func (d document) Vector() []float64 {
	return d.Embedding
}

type searchResult struct {
	ID        int       `bson:"id"`
	Name      string    `bson:"name"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
	Score     float64   `bson:"score"`
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Construct the llm client for access the model server.
	llm := client.NewLLM(url, model)

	// -------------------------------------------------------------------------

	fmt.Println("\nConnecting to MongoDB")

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("mongodb.Connect: %w", err)
	}
	defer client.Disconnect(ctx)

	// -------------------------------------------------------------------------

	fmt.Println("Initializing Database")

	col, err := initDB(ctx, client)
	if err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	// -------------------------------------------------------------------------

	if err := insertDocuments(ctx, llm, col); err != nil {
		return err
	}

	// -------------------------------------------------------------------------

	fmt.Print("\n---- VECTOR SEARCH ----\n\n")

	search := func(searchDocument string) {
		fmt.Printf("Searching for: %q\n", searchDocument)

		results, err := vectorSearch(ctx, col, llm, searchDocument, 10)
		if err != nil {
			fmt.Printf("error while searching: %v", fmt.Errorf("storeDocuments: %w", err))
		}

		for _, result := range results {
			fmt.Printf("%s -> %s: %.2f%% similar\n",
				result.Name,
				result.Text,
				result.Score)
		}

		fmt.Printf("\n\n")
	}

	search("worker")
	search("worker woman")
	search("human worker woman")

	fmt.Printf("\n\n")

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

	// Delete any documents that might be there.
	col.DeleteMany(ctx, bson.D{})

	return col, nil
}

func insertDocuments(ctx context.Context, llm *client.LLM, col *mongo.Collection) error {
	fmt.Println("Inserting Documents")

	// Apply the feature vectors to the hand crafted data points.
	// This time you need to use words since we are using a word based model.
	documents := []vector.Data{
		document{ID: 1, Name: "Horse   ", Text: "Animal Female"},
		document{ID: 2, Name: "Man     ", Text: "Human  Male   Pants Poor Worker"},
		document{ID: 3, Name: "Woman   ", Text: "Human  Female Dress Poor Worker"},
		document{ID: 4, Name: "King    ", Text: "Human  Male   Pants Rich Ruler"},
		document{ID: 5, Name: "Queen   ", Text: "Human  Female Dress Rich Ruler"},
	}

	fmt.Print("\n")

	var data []any

	// Iterate over each data point and use the LLM to generate the vector
	// embedding related to the model.
	for i, dp := range documents {
		dataPoint := dp.(document)

		embedding, err := llm.EmbedText(ctx, dataPoint.Text)
		if err != nil {
			return fmt.Errorf("embedding: %w", err)
		}

		dataPoint.Embedding = embedding
		documents[i] = dataPoint

		data = append(data, dataPoint)

		fmt.Printf("Vector: Name(%s) len(%d) %v...%v\n", dataPoint.Name, len(embedding), embedding[0:2], embedding[len(embedding)-2:])
	}

	res, err := col.InsertMany(ctx, data)
	if err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	fmt.Println("\nInserted IDs:")

	for _, insertedID := range res.InsertedIDs {
		fmt.Printf("%v\n", insertedID)
	}

	// We need to give Mongo a little time to index the documents.
	time.Sleep(time.Second)

	return nil
}

func vectorSearch(ctx context.Context, col *mongo.Collection, llm *client.LLM, searchDocument string, limit int) ([]searchResult, error) {
	embedding, err := llm.EmbedText(ctx, searchDocument)
	if err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}

	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":       "vector_index",
				"exact":       true,
				"path":        "embedding",
				"queryVector": embedding,
				"limit":       limit,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"id":        1,
				"name":      1,
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
