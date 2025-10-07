package main

import (
	"context"
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/client"
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

type searchResult struct {
	FileName  string    `bson:"file_name" json:"file_name"`
	Duration  string    `bson:"duration" json:"duration"`
	Text      string    `bson:"text" json:"text"`
	Embedding []float64 `bson:"embedding" json:"-"`
	Score     float64   `bson:"score" json:"-"`
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

func textVectorSearch(ctx context.Context, llm *client.LLM, col *mongo.Collection, question string) ([]searchResult, error) {
	vector, err := llm.EmbedText(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("embedText: %w", err)
	}

	return vectorSearch(ctx, col, vector)
}

func vectorSearch(ctx context.Context, col *mongo.Collection, vector []float64) ([]searchResult, error) {
	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":       "vector_embedding_index",
				"exact":       true,
				"path":        "embedding",
				"queryVector": vector,
				"limit":       2,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"file_name": 1,
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
