// This example shows you how to use the Llama3.2 vision model to generate
// an image description and update the image with the description.
// After generating the embeddings, we'll store the image path and its embeddings
// into a database so we can later search for it.
//
// # Running the example:
//
//	$ make example9-step3
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
//	$ make compose-up // This starts the MongoDB service.
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/dsoprea/go-exif/v3"
	exifcommon "github.com/dsoprea/go-exif/v3/common"
	jpg "github.com/dsoprea/go-jpeg-image-structure/v2"
	pis "github.com/dsoprea/go-png-image-structure/v2"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type document struct {
	FileName    string    `bson:"file_name"`
	Description string    `bson:"description"`
	Embedding   []float32 `bson:"embedding"`
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	fileName := "cmd/samples/roseimg.png"

	data, err := readImage(fileName)
	if err != nil {
		return fmt.Errorf("read image: %w", err)
	}

	prompt := `Describe the image.
Be concise and accurate.
Do not be overly verbose or stylistic.
Make sure all the elements in the image are enumerated and described.
Do not include any additional details.
Keep the description under 200 words.
At the end of the description, create a list of tags with the names of all the elements in the image.
Do no output anything past this list.
Encode the list as valid JSON, as in this example:
[
  "tag1",
  "tag2",
  "tag3",
  ...
]
Make sure the JSON is valid, doesn't have any extra spaces, and is properly formatted.
`

	var mimeType string
	switch filepath.Ext(fileName) {
	case ".jpg", ".jpeg":
		mimeType = "image/jpg"
	case ".png":
		mimeType = "image/png"
	default:
		return fmt.Errorf("unsupported file type: %s", filepath.Ext(fileName))
	}

	// -------------------------------------------------------------------------

	fmt.Println("Generating image description...")

	llm, err := ollama.New(
		ollama.WithModel("llama3.2-vision"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.BinaryContent{
					MIMEType: mimeType,
					Data:     data,
				},
				llms.TextContent{
					Text: prompt,
				},
			},
		},
	}

	cr, err := llm.GenerateContent(context.Background(), messages)
	if err != nil {
		return fmt.Errorf("generate content: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Printf("Updating Image description: %s\n", cr.Choices[0].Content)

	err = updateImage(fileName, cr.Choices[0].Content)
	if err != nil {
		return fmt.Errorf("update image: %w", err)
	}

	fmt.Printf("Inserting image description into the database: %s\n", cr.Choices[0].Content)

	vector, err := generateEmbeddings(cr.Choices[0].Content)
	if err != nil {
		return fmt.Errorf("generate embeddings: %w", err)
	}

	return updateDatabase(fileName, cr.Choices[0].Content, vector)
}

func readImage(fileName string) ([]byte, error) {
	f, err := os.OpenFile(fileName, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	return data, nil
}

func updateImage(fileName string, description string) error {
	im, err := exifcommon.NewIfdMappingWithStandard()
	if err != nil {
		return fmt.Errorf("new idf mapping: %w", err)
	}

	ti := exif.NewTagIndex()
	ib := exif.NewIfdBuilder(im, ti, exifcommon.IfdStandardIfdIdentity, exifcommon.EncodeDefaultByteOrder)

	err = ib.AddStandardWithName("ImageDescription", description)
	if err != nil {
		return fmt.Errorf("add standard: %w", err)
	}

	// -------------------------------------------------------------------------

	switch filepath.Ext(fileName) {
	case ".jpg", ".jpeg":
		intfc, err := jpg.NewJpegMediaParser().ParseFile(fileName)
		if err != nil {
			return fmt.Errorf("parse file: %w", err)
		}

		cs := intfc.(*jpg.SegmentList)
		err = cs.SetExif(ib)
		if err != nil {
			return fmt.Errorf("set ib: %w", err)
		}

		f, err := os.Create(fileName)
		if err != nil {
			return fmt.Errorf("create: %w", err)
		}

		err = cs.Write(f)
		if err != nil {
			return fmt.Errorf("write: %w", err)
		}
		defer f.Close()

	case ".png":
		intfc, err := pis.NewPngMediaParser().ParseFile(fileName)
		if err != nil {
			return fmt.Errorf("parse file: %w", err)
		}

		cs := intfc.(*pis.ChunkSlice)
		err = cs.SetExif(ib)
		if err != nil {
			return fmt.Errorf("set ib: %w", err)
		}

		f, err := os.Create(fileName)
		if err != nil {
			return fmt.Errorf("create: %w", err)
		}

		err = cs.WriteTo(f)
		if err != nil {
			return fmt.Errorf("write: %w", err)
		}
		defer f.Close()

	default:
		return fmt.Errorf("unsupported file type: %s", filepath.Ext(fileName))
	}

	return nil
}

func generateEmbeddings(description string) ([]float32, error) {
	llm, err := ollama.New(
		ollama.WithModel("mxbai-embed-large"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		log.Fatal(err)
	}

	vectors, err := llm.CreateEmbedding(context.Background(), []string{description})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Received embeddings from model")

	return vectors[0], nil
}

func updateDatabase(fileName string, description string, vector []float32) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// -------------------------------------------------------------------------
	// Connect to mongo

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("connectToMongo: %w", err)
	}
	defer client.Disconnect(ctx)

	fmt.Println("Connected to MongoDB")

	// -------------------------------------------------------------------------
	// Create database and collection

	const dbName = "example9"
	const collectionName = "images"

	db := client.Database(dbName)

	col, err := mongodb.CreateCollection(ctx, db, collectionName)
	if err != nil {
		return fmt.Errorf("createCollection: %w", err)
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
		return fmt.Errorf("createVectorIndex: %w", err)
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
		return fmt.Errorf("createUniqueIndex: %w", err)
	}

	fmt.Println("Created Unique file_name Index")

	// -------------------------------------------------------------------------
	// Store some documents with their embeddings.

	if err := storeDocuments(ctx, col, fileName, description, vector); err != nil {
		return fmt.Errorf("storeDocuments: %w", err)
	}

	return nil
}

func storeDocuments(ctx context.Context, col *mongo.Collection, fileName string, description string, vector []float32) error {

	// If this record already exist, we don't need to add it again.
	findRes := col.FindOne(ctx, bson.D{{Key: "file_name", Value: d.FileName}})
	if findRes.Err() != nil && !errors.Is(res.Err(), mongo.ErrNoDocuments) {
		return fmt.Errorf("find: %w", err)
	}

	// -------------------------------------------------------------------------

	// Let's add the document to the database.

	d1 := document{
		FileName:    fileName,
		Description: description,
		Embedding:   vector,
	}

	res, err := col.InsertOne(ctx, d1)
	if err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	fmt.Printf("Inserted db id: %s\n", res.InsertedID)

	return nil
}
