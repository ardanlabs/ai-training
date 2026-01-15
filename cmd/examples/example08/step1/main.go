// This example shows you how to use a vision model to generate
// an image description.
//
// # Running the example:
//
//	$ make example08-step1
//
// # This requires running the following commands:
//
//	$ make kronk-up

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

var (
	url   = "http://localhost:8080/v1/chat/completions"
	model = "Qwen2.5-VL-3B-Instruct-Q8_0"

	imagePath = "zarf/samples/gallery/roseimg.png"
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
		return fmt.Errorf("read image: %w", err)
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

	llm := client.NewLLM(url, model)

	results, err := llm.ChatCompletions(ctx, prompt, client.WithImage(mimeType, image))
	if err != nil {
		return fmt.Errorf("chat completions: %w", err)
	}

	fmt.Printf("%s\n", results)

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
		return nil, "", fmt.Errorf("unsupported file type: %s: filename: %s", mimeType, fileName)
	}
}
