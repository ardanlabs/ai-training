// This example shows you how to use an LLM to create vector embeddings and
// get the same results from the hand crafted solution.
//
// # Running the example:
//
//  $ make example2
//
// # This requires running the following commands:
//
//  $ make ollama-up // This starts the Ollama service.
//
// # Extra reading and watching:
//
//	https://www.youtube.com/watch?v=Fuw0wv3X-0o&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=40
//  https://www.youtube.com/watch?v=hQwFeIupNP0&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=41
//  https://machinelearningmastery.com/what-are-word-embeddings/
//  https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/tmc/langchaingo/llms/ollama"
)

type data struct {
	Name      string
	Text      string
	Embedding []float32 // The vector where the data is embedded in space.
}

// Vector can convert the specified data into a vector.
func (d data) Vector() []float32 {
	return d.Embedding
}

// =============================================================================

func main() {
	llm, err := ollama.New(
		ollama.WithModel("mxbai-embed-large"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		log.Fatal(err)
	}

	// -------------------------------------------------------------------------

	// Apply the feature vectors to the hand crafted data points.
	// This time you need to use words since we are using a word based model.
	dataPoints := []vector.Data{
		data{Name: "Horse   ", Text: "Animal, Female"},
		data{Name: "Man     ", Text: "Human,  Male,   Pants, Poor, Worker"},
		data{Name: "Woman   ", Text: "Human,  Female, Dress, Poor, Worker"},
		data{Name: "King    ", Text: "Human,  Male,   Pants, Rich, Ruler"},
		data{Name: "Queen   ", Text: "Human,  Female, Dress, Rich, Ruler"},
	}

	// Iterate over each data point and use the LLM to generate the vector
	// embedding related to the model.
	for i, dp := range dataPoints {
		dataPoint := dp.(data)

		vectors, err := llm.CreateEmbedding(context.Background(), []string{dataPoint.Text})
		if err != nil {
			log.Fatal(err)
		}

		dataPoint.Embedding = vectors[0]
		dataPoints[i] = dataPoint
	}

	// -------------------------------------------------------------------------

	// Compare each data point to every other by performing a cosine
	// similarity comparison using the vector embedding from the LLM.
	for _, target := range dataPoints {
		results := vector.Similarity(target, dataPoints...)

		for _, result := range results {
			fmt.Printf("%s -> %s: %.3f%% similar\n",
				result.Target.(data).Name,
				result.DataPoint.(data).Name,
				result.Percentage)
		}
		fmt.Print("\n")
	}

	// -------------------------------------------------------------------------

	// Perform the same vector math as in example2 using the LLM vector embedding.

	// You can perform vector math by adding and subtracting vectors.
	kingSubMan := vector.Sub(dataPoints[3].Vector(), dataPoints[1].Vector())
	kingSubManPlusWoman := vector.Add(kingSubMan, dataPoints[2].Vector())
	queen := dataPoints[4].Vector()

	// Now compare a (King - Man + Woman) to a Queen.
	result := vector.CosineSimilarity(kingSubManPlusWoman, queen)
	fmt.Printf("King - Man + Woman ~= Queen similarity: %.3f%%\n", result*100)
}
