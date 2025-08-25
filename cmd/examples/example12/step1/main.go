// This example shows you how to extract frames from a video.
//
// # Running the example:
//
//	$ make example12-step1
//

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

const (
	urlChat   = "http://localhost:11434/v1/chat/completions"
	modelChat = "qwen2.5vl:latest"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()

	if err := processVideo(); err != nil {
		return fmt.Errorf("process video: %w", err)
	}

	const prompt = `
		Give me all the text in the image.
		Do not describe the image.
		Make sure that ALL the text is in the image.
		Make sure that you include any text editor/IDE tool too.
		Output the text in a JSON format similar to this:
		{
			"text": "<Text that can be found in this image>"
		}
		
		If no text is found, output:
		{
			"error": "NO TEXT FOUND"
		}
		
		Make sure that there's no extra whitespace or formatting, or markdown surrounding the json output.
`

	llmChat := client.NewLLM(urlChat, modelChat)

	files, err := getFilesFromDirectory("zarf/samples/videos/frames")
	if err != nil {
		return fmt.Errorf("get files from directory: %w", err)
	}

	for _, fileName := range files {
		fmt.Printf("\nProcessing image: %s\n", fileName)
		image, mimeType, err := readImage(fileName)
		if err != nil {
			return fmt.Errorf("read image: %w", err)
		}

		results, err := llmChat.ChatCompletions(ctx, prompt, client.WithImage(mimeType, image))
		if err != nil {
			return fmt.Errorf("chat completions: %w", err)
		}

		fmt.Printf("%s\n", results)
	}

	// -------------------------------------------------------------------------

	fmt.Println("\nDONE")
	return nil
}

func processVideo() error {
	fmt.Println("Processing Video ...")
	defer fmt.Println("\nDONE Processing Video")

	ffmpegCommand := "ffmpeg -i zarf/samples/videos/training.mp4 -vf \"select='eq(pict_type,I)'\" -vsync vfr -loglevel error zarf/samples/videos/frames/frame-%03d.jpg"
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
