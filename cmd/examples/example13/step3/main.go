// This example builds on example13-step1 and adds the extraction of code from
// the keyframes extracted in example13-step1.
//
// # Running the example:
//
//	$ make example13-step2
//
// # This requires running the following commands:
//
//	$ make ollama-up    // This starts the Ollama service.
//	$ make embedding-up // This starts the Embedding service.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/audio"
	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/vector"
)

type frame struct {
	fileName       string
	description    string
	code           string
	classification string
	embedding      []float64
	startTime      float64
	duration       float64
	mimeType       string
	image          []byte
}

const (
	urlChat    = "http://localhost:11434/v1/chat/completions"
	urlEmbed   = "http://localhost:11439/v1/embeddings"
	modelChat  = "mistral-small3.2:latest"
	modelEmbed = "nomic-embed-vision-v1.5"

	dimensions          = 768
	similarityThreshold = 0.80
	sourceDir           = "zarf/samples/videos/"
	sourceFileName      = "zarf/samples/videos/training.mp4"
)

// -----------------------------------------------------------------------------

var audioCfg = audio.Config{
	SetLanguage: "en",
	Temperature: 0.1,
	Threads:     4,
}

// -----------------------------------------------------------------------------

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	// -------------------------------------------------------------------------

	llmChat := client.NewLLM(urlChat, modelChat)
	llmEmbed := client.NewLLM(urlEmbed, modelEmbed)

	adio, err := audio.New(client.StdoutLogger, "zarf/audio/ggml-tiny.bin")
	if err != nil {
		return fmt.Errorf("starting audio: %w", err)
	}

	// -------------------------------------------------------------------------

	if err := splitVideoIntoChunks(sourceFileName); err != nil {
		return fmt.Errorf("splitting video into chunks: %w", err)
	}

	// -------------------------------------------------------------------------

	totalFramesTime := 0.0

	chunksDir := filepath.Join(sourceDir, "chunks")
	fmt.Printf("\nProcessing video chunks in directory: %s\n", chunksDir)

	err = fs.WalkDir(os.DirFS(chunksDir), ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() {
			return nil
		}

		if !strings.HasSuffix(path, ".mp4") {
			return nil
		}

		duration, err := getVideoDuration(filepath.Join(sourceDir, "chunks", path))
		if err != nil {
			return fmt.Errorf("get video duration: %w", err)
		}

		// Defer the total time computation until after processing the chunk.
		defer func() {
			totalFramesTime += duration
		}()

		return processChunk(ctx, llmChat, llmEmbed, adio, sourceDir, path, totalFramesTime, duration)
	})
	if err != nil {
		return fmt.Errorf("walk directory: %w", err)
	}

	return nil
}

// -------------------------------------------------------------------------

const extractFrameInfoPrompt = `
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

const extractCodePrompt = `
		Extract all the source code in the image.
		Do not include any other text.
		Do not interpret the code in the image.
		
		Output the text in a valid JSON format MATCHING this format:
		{
			"code": "<source code>"
		}

		If there is no code in the image, output:
		{
			"error": "NO CODE FOUND"
		}

		Encode any special characters in the JSON output.
		
		Make sure that there's no extra whitespace or formatting, or markdown surrounding the json output.
		MAKE SURE THAT THE JSON IS VALID.
		DO NOT INCLUDE ANYTHING ELSE BUT THE JSON DOCUMENT IN THE RESPONSE.
`

func processChunk(ctx context.Context, llmChat *client.LLM, llmEmbed *client.LLM, adio *audio.Audio, sourceDir string, sourceFileName string, totalFramesTime float64, duration float64) error {
	fullPath := filepath.Join(sourceDir, "chunks", sourceFileName)

	fmt.Printf("\nRemoving the frames from the previous chunk: %s\n", filepath.Join(sourceDir, "frames"))

	previousFrames, err := fs.Glob(os.DirFS(sourceDir), "frames/*")
	if err != nil {
		return fmt.Errorf("glob: %w", err)
	}

	for _, previousFrame := range previousFrames {
		if err := os.Remove(filepath.Join(sourceDir, previousFrame)); err != nil {
			return fmt.Errorf("remove previous frame: %w", err)
		}
	}

	// -------------------------------------------------------------------------

	fmt.Printf("\nProcessing video chunk: %s\n", fullPath)

	if err := extractKeyFramesFromVideo(fullPath); err != nil {
		return fmt.Errorf("process video: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Printf("\nProcessing images in directory: %s\n", sourceDir)

	files, err := getFilesFromDirectory(filepath.Join(sourceDir, "frames"))
	if err != nil {
		return fmt.Errorf("get files from directory: %w", err)
	}

	frames := make([]frame, 0, len(files))

	for _, fileName := range files {
		f := frame{
			fileName:  fileName,
			startTime: totalFramesTime,
			duration:  duration,
		}

		fmt.Printf("\nProcessing image: %s\n", fileName)

		// -------------------------------------------------------------------------

		image, mimeType, err := readImage(fileName)
		if err != nil {
			return fmt.Errorf("read image: %w", err)
		}

		f.mimeType = mimeType
		f.image = make([]byte, len(image))
		copy(f.image, image)

		// -------------------------------------------------------------------------

		fmt.Println("\nGenerating embeddings for the image description:")

		embedding, err := llmEmbed.EmbedWithImage(ctx, "", image, mimeType)
		if err != nil {
			return fmt.Errorf("llm.EmbedText: %w", err)
		}

		fmt.Printf("%v...%v\n", embedding[0:3], embedding[len(embedding)-3:])

		f.embedding = make([]float64, dimensions)
		copy(f.embedding, embedding)

		// -------------------------------------------------------------------------

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

	fmt.Printf("\nUnique Frames: %d\n", len(uniqueFrames))
	for _, f := range uniqueFrames {
		fmt.Printf("  - FileName: %s - [%.4f, %.4f]\n", f.fileName, f.startTime, f.duration)
	}

	// -------------------------------------------------------------------------

	fmt.Println("\nExtracting frame descriptions:")

	for i, f := range uniqueFrames {
		fmt.Printf("Extracting description for image: %s\n", f.fileName)

		description, err := llmChat.ChatCompletions(ctx, extractFrameInfoPrompt, client.WithImage(f.mimeType, f.image))
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

		uniqueFrames[i].description = descr.Text
		uniqueFrames[i].classification = descr.Classification

		// -------------------------------------------------------------------------

		// NOW WE WILL EXTRACT CODE SAMPLES OUT OF THE KEYFRAMES TO ADD TO THE
		// CONTEXT FOR THE EMBEDDING.

		if descr.Classification == "source code" {
			fmt.Println("  - Source code classification detected, extracting code...")
			code, err := llmChat.ChatCompletions(ctx, extractCodePrompt, client.WithImage(f.mimeType, f.image))
			if err != nil {
				return fmt.Errorf("chat completions: %w", err)
			}

			uniqueFrames[i].code = code
			fmt.Printf("LLM RESPONSE: %s\n", code)
		}
	}

	// -------------------------------------------------------------------------

	fmt.Println("\nUnique Frames:")

	for _, f := range uniqueFrames {
		fmt.Printf("\t- FileName: %s - [%.4f, %.4f, %s]\n", f.fileName, f.startTime, f.startTime, f.classification)
	}

	// -------------------------------------------------------------------------

	if err := convertVideoToWav(fullPath); err != nil {
		return fmt.Errorf("converting video to wav: %w", err)
	}

	response, err := adio.Process(ctx, audioCfg, "zarf/samples/audio/output.wav")
	if err != nil {
		return fmt.Errorf("process audio: %w", err)
	}

	fmt.Printf("\nChunk audio transcription: %s\n", response.Text)

	// -------------------------------------------------------------------------

	fmt.Println("\nDONE")
	return nil
}

// -------------------------------------------------------------------------

func splitVideoIntoChunks(source string) error {
	fmt.Println("Processing Video ...")
	defer fmt.Println("\nDONE Processing Video")

	ffmpegCommand := fmt.Sprintf("ffmpeg -i %s -c copy -map 0 -f segment -segment_time 15 -reset_timestamps 1 -loglevel error zarf/samples/videos/chunks/output_%%05d.mp4", source)
	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w: %s", err, string(out))
	}

	return nil
}

// -------------------------------------------------------------------------

func extractKeyFramesFromVideo(source string) error {
	fmt.Printf("Extracting Video %s keyframes...\n", source)
	defer fmt.Println("\nDONE Extracting Video keyframes")

	ffmpegCommand := fmt.Sprintf("ffmpeg -skip_frame nokey -i %s -frame_pts true -fps_mode vfr -loglevel error zarf/samples/videos/frames/%%05d.jpg", source)

	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w: %s", err, string(out))
	}

	return nil
}

// -------------------------------------------------------------------------

func getVideoDuration(filePath string) (float64, error) {
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json",
		"-show_entries", "format=duration", filePath)

	output, err := cmd.Output()
	if err != nil {
		return 0, err
	}

	var probe struct {
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}

	if err := json.Unmarshal(output, &probe); err != nil {
		return 0, err
	}

	duration, err := strconv.ParseFloat(probe.Format.Duration, 64)
	if err != nil {
		return 0, err
	}

	return duration, nil
}

// -------------------------------------------------------------------------

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

// -------------------------------------------------------------------------

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

// -------------------------------------------------------------------------

func convertVideoToWav(source string) error {
	fmt.Printf("Converting Video %s to Audio...\n", source)
	defer fmt.Println("\nDONE Converting Video to audio")

	// Ensure there is no previous file to allow ffmpeg to create the new one.
	_ = os.Remove("zarf/samples/audio/output.wav")

	ffmpegCommand := fmt.Sprintf("ffmpeg -i %s -ar 16000 -ac 1 -c:a pcm_s16le -loglevel error zarf/samples/audio/output.wav", source)
	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w: %s", err, string(out))
	}

	return nil
}
