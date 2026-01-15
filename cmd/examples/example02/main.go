// This example shows you how to use an LLM to create vector embeddings and
// get the same results from the hand crafted solution.
//
// # Running the example:
//
//  $ make example02
//
// # This requires running the following commands:
//
//  $ make kronk-up
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
	"os"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/vector"
)

var (
	url   = "http://localhost:8080/v1/embeddings"
	model = "embeddinggemma-300m-qat-Q8_0"
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

type data struct {
	Name      string
	Text      string
	Embedding []float64 // The vector where the data is embedded in space.
}

// Vector can convert the specified data into a vector.
func (d data) Vector() []float64 {
	return d.Embedding
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

	// Old way of representing this data with our own vector data points.
	// dataPoints := []vector.Data{
	// 	data{Name: "Horse   ", Authority: 0.0, Animal: 1.0, Human: 0.0, Rich: 0.0, Gender: +1.0},
	// 	data{Name: "Man     ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: -1.0},
	// 	data{Name: "Woman   ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: +1.0},
	// 	data{Name: "King    ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: -1.0},
	// 	data{Name: "Queen   ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: +1.0},
	// }

	// Apply the feature vectors to the hand crafted data points.
	// This time you need to use words since we are using a word based model.
	dataPoints := []vector.Data{
		data{Name: "Horse   ", Text: "Animal, Female"},
		data{Name: "Man     ", Text: "Human,  Male,   Pants, Poor, Worker"},
		data{Name: "Woman   ", Text: "Human,  Female, Dress, Poor, Worker"},
		data{Name: "King    ", Text: "Human,  Male,   Pants, Rich, Ruler"},
		data{Name: "Queen   ", Text: "Human,  Female, Dress, Rich, Ruler"},
	}

	fmt.Print("\n")

	// Iterate over each data point and use the LLM to generate the vector
	// embedding related to the model.
	for i, dp := range dataPoints {
		dataPoint := dp.(data)

		vector, err := llm.EmbedText(ctx, dataPoint.Text)
		if err != nil {
			return fmt.Errorf("embedding: %w", err)
		}

		dataPoint.Embedding = vector
		dataPoints[i] = dataPoint

		fmt.Printf("Vector: Name(%s) len(%d) %v...%v\n", dataPoint.Name, len(vector), vector[0:2], vector[len(vector)-2:])
	}

	fmt.Print("\n")

	// -------------------------------------------------------------------------

	// Compare each data point to every other by performing a cosine
	// similarity comparison using the vector embedding from the LLM.
	for _, target := range dataPoints {
		results := vector.Similarity(target, dataPoints...)

		for _, result := range results {
			fmt.Printf("%s -> %s: %.2f%% similar\n",
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
	fmt.Printf("King - Man + Woman ~= Queen similarity: %.2f%%\n", result*100)

	return nil
}
