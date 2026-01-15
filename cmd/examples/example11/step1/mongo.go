package main

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	dbName     = "example11"
	colName    = "trainingvideo"
	dimensions = 768
)

type document struct {
	Video     string    `bson:"video"`
	Chunk     string    `bson:"chunk"`
	StartTime float64   `bson:"start_time"`
	Duration  float64   `bson:"duration"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
}

// =============================================================================

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

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "video", Value: 1}, {Key: "chunk", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	col.Indexes().CreateOne(ctx, indexModel)

	return col, nil
}

func existsDocument(ctx context.Context, col *mongo.Collection, videoFileName string, videoChunkFile string) (bool, error) {
	sRes := col.FindOne(ctx, bson.D{{Key: "video", Value: videoFileName}, {Key: "chunk", Value: filepath.Base(videoChunkFile)}})

	// If there is no error, the document already exists.
	if sRes.Err() == nil {
		return true, nil
	}

	// If the error is anything other than not exist, return the error.
	if sRes.Err() != nil {
		if !errors.Is(sRes.Err(), mongo.ErrNoDocuments) {
			return false, fmt.Errorf("find one: %w", sRes.Err())
		}
	}

	return false, nil
}

func insertDocument(ctx context.Context, col *mongo.Collection, embed []float64, input string, videoFileName string, videoChunkFile string, startingVideoTime float64, duration float64) error {
	doc := document{
		Video:     videoFileName,
		Chunk:     filepath.Base(videoChunkFile),
		StartTime: startingVideoTime,
		Duration:  duration,
		Text:      input,
		Embedding: embed,
	}

	res, err := col.InsertOne(ctx, doc)
	if err != nil {
		return fmt.Errorf("col.InsertOne: %w", err)
	}

	fmt.Printf("Inserted into Mongo: %v\n", res.InsertedID)

	return nil
}
