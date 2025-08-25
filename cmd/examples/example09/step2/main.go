// This examples takes step1 and shows you how to generate a vector embedding
// from the image description.
//
// # Running the example:
//
//	$ make example9-step2
//
// # This requires running the following commands:
//
//	$ make ollama-up

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

const (
	urlChat    = "http://localhost:11434/v1/chat/completions"
	urlEmbed   = "http://localhost:11434/v1/embeddings"
	modelChat  = "qwen2.5vl:latest"
	modelEmbed = "bge-m3:latest"
	imagePath  = "zarf/samples/gallery/roseimg.png"
)

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()

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

	// -------------------------------------------------------------------------

	fmt.Println("\nGenerating embeddings for the image description:")

	llm = client.NewLLM(urlEmbed, modelEmbed)

	vector, err := llm.EmbedText(ctx, results)
	if err != nil {
		return fmt.Errorf("llm.EmbedText: %w", err)
	}

	fmt.Printf("%v...%v\n", vector[0:3], vector[len(vector)-3:])

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
		return nil, "", fmt.Errorf("unsupported file type:%s: filename: %s", mimeType, fileName)
	}
}
