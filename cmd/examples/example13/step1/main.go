// This example builds on example12-step1 and shows you how to
// process the extracted frames from a video using an LLM.
// Before images are inserted in the DB, we'll check if they are
// similar and how similar they are.
//
// # Running the example:
//
//	$ make example12-step2
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
//	$ make compose-up // This starts the Mongo service.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/vector"
)

type frame struct {
	fileName       string
	description    string
	classification string
	embedding      []float64
}

const (
	urlChat    = "http://localhost:11434/v1/chat/completions"
	urlEmbed   = "http://localhost:11434/v1/embeddings"
	modelChat  = "mistral-small3.2:latest"
	modelEmbed = "bge-m3:latest"

	dimensions = 1024

	similarityThreshold = 0.80
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	if err := processVideo(); err != nil {
		return fmt.Errorf("process video: %w", err)
	}

	// -------------------------------------------------------------------------

	const prompt = `
		Provide a detailed description of this image in 300 words or less.
		Also, classify this image as: "source code", "diagram", "terminal", or "other" depending on the content it features the most.
		If icons are present in the middle of the image and blocking the main content, classify them as "icon".
		
		Output the text in a valid JSON format MATCHING this format:
		{
			"text": "<image description>",
			"classification": "<image classification>"
		}

		Encode any special characters in the JSON output.
		
		Make sure that there's no extra whitespace or formatting, or markdown surrounding the json output.
		MAKE SURE THAT THE JSON IS VALID.
		DO NOT INCLUDE ANYTHING ELSE BUT THE JSON DOCUMENT IN THE RESPONSE.
`

	// -------------------------------------------------------------------------

	llmChat := client.NewLLM(urlChat, modelChat)
	llmEmbed := client.NewLLM(urlEmbed, modelEmbed)

	// -------------------------------------------------------------------------

	files, err := getFilesFromDirectory("zarf/samples/videos/frames")
	if err != nil {
		return fmt.Errorf("get files from directory: %w", err)
	}

	frames := make([]frame, 0, len(files))

	for _, fileName := range files {
		f := frame{
			fileName: fileName,
		}

		fmt.Printf("\nProcessing image: %s\n", fileName)

		// -------------------------------------------------------------------------

		image, mimeType, err := readImage(fileName)
		if err != nil {
			return fmt.Errorf("read image: %w", err)
		}

		// -------------------------------------------------------------------------

		description, err := llmChat.ChatCompletions(ctx, prompt, client.WithImage(mimeType, image))
		if err != nil {
			return fmt.Errorf("chat completions: %w", err)
		}

		description = strings.Trim(description, "`")
		description = strings.TrimPrefix(description, "json")

		fmt.Printf("LLM RESPONSE: %s\n", description)

		var descr struct {
			Text           string `json:"text"`
			Classification string `json:"classification"`
		}
		if err := json.Unmarshal([]byte(description), &descr); err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		if descr.Classification == "icon" {
			fmt.Println("  - Icon classification detected, skipping...")
			continue
		}

		f.description = descr.Text
		f.classification = descr.Classification

		// -------------------------------------------------------------------------

		fmt.Println("\nGenerating embeddings for the image description:")

		embedding, err := llmEmbed.EmbedText(ctx, description)
		if err != nil {
			return fmt.Errorf("llm.EmbedText: %w", err)
		}

		fmt.Printf("%v...%v\n", embedding[0:3], embedding[len(embedding)-3:])

		f.embedding = make([]float64, dimensions)
		copy(f.embedding, embedding)

		frames = append(frames, f)
	}

	// -------------------------------------------------------------------------

	var uniqueFrames []frame
	for idx, f := range frames {
		if idx == 0 {
			uniqueFrames = append(uniqueFrames, f)
			continue
		}

		var isDuplicate bool
		for _, previousFrame := range uniqueFrames {
			fmt.Printf("Checking image similarity between: %s - %s\n", previousFrame.fileName, f.fileName)
			similarity := vector.CosineSimilarity(previousFrame.embedding, f.embedding)
			fmt.Printf("  - Image similarity: %.3f\n", similarity)

			if similarity > similarityThreshold {
				isDuplicate = true
				fmt.Println("  - Image is similar to previous image")
				break
			}
		}

		if !isDuplicate {
			uniqueFrames = append(uniqueFrames, f)
		}
	}

	// -------------------------------------------------------------------------

	fmt.Println("\nUnique Frames:")

	for _, f := range uniqueFrames {
		fmt.Printf("\t- FileName: %s - [%s]\n", f.fileName, f.classification)
	}

	fmt.Println("\nDONE")
	return nil
}

func processVideo() error {
	fmt.Println("Processing Video ...")
	defer fmt.Println("\nDONE Processing Video")

	ffmpegCommand := "ffmpeg -i zarf/samples/videos/output_0089.mp4 -vf \"select='eq(pict_type,I)'\" -loglevel error -vsync vfr zarf/samples/videos/frames/frame-%03d.jpg"
	_, err := exec.Command("/bin/sh", "-c", ffmpegCommand).Output()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w", err)
	}

	return nil
}

func getFilesFromDirectory(directoryPath string) ([]string, error) {
	var files []string

	err := filepath.Walk(directoryPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && (filepath.Ext(info.Name()) == ".jpg" || filepath.Ext(info.Name()) == ".jpeg" || filepath.Ext(info.Name()) == ".png") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk directory: %w", err)
	}

	return files, nil
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
