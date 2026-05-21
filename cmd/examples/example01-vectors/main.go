// This example shows you what a vector and embedding is by hand crafting
// a relationship of data. It also shows you how cosine similarity works
// between different vectors.
//
// # Running the example
//
//	$ make example01

// Example 01 — Vectors
package main

import (
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/vector"
)

type data struct {
	Name      string
	Authority float64 // These fields are called features.
	Animal    float64
	Human     float64
	Rich      float64
	Gender    float64
}

// Vector can convert the specified data into a vector.
func (d data) Vector() []float64 {
	return []float64{
		d.Authority,
		d.Animal,
		d.Human,
		d.Rich,
		d.Gender,
	}
}

// String pretty prints an embedding to a vector representation.
func (d data) String() string {
	return fmt.Sprintf("%f", d.Vector())
}

// =============================================================================

func main() {

	// Apply the feature dataPoints to the handcrafted embeddings.
	dataPoints := []vector.Data{
		data{Name: "Horse   ", Authority: 0.0, Animal: 1.0, Human: 0.0, Rich: 0.0, Gender: +1.0},
		data{Name: "Man     ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: -1.0},
		data{Name: "Woman   ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: +1.0},
		data{Name: "King    ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: -1.0},
		data{Name: "Queen   ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: +1.0},
	}

	// -------------------------------------------------------------------------
	// Compare data points
	// region Compare data points

	// Display the data points.
	fmt.Print("\n")
	for _, v := range dataPoints {
		fmt.Printf("Vector: Name(%s) len(%d) %v\n", v.(data).Name, len(v.(data).Vector()), v.(data).Vector())
	}
	fmt.Print("\n")

	// Compare each data point to every other by performing a cosine
	// similarity comparison. This requires converting each data point
	// into a vector.
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

	// endregion

	// -------------------------------------------------------------------------
	// Vector math
	// region Vector math

	// You can perform vector math by adding and subtracting vectors.
	kingSubMan := vector.Sub(dataPoints[3].Vector(), dataPoints[1].Vector())
	kingSubManPlusWoman := vector.Add(kingSubMan, dataPoints[2].Vector())
	queen := dataPoints[4].Vector()

	// Now compare a (King - Man + Woman) to a Queen.
	result := vector.CosineSimilarity(kingSubManPlusWoman, queen)
	fmt.Printf("King - Man + Woman ~= Queen similarity: %.2f%%\n", result*100)

	// endregion
}
