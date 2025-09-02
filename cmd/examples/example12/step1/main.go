// This example provides a proof of concept for extracting transcriptions, code,
// diagrams, images, and text from videos using the Ollama and Embedding services.
//
// # Running the example:
//
//	$ make example12-step1
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

var (
	urlChat    = "http://localhost:11434/v1/chat/completions"
	urlEmbed   = "http://localhost:11439/v1/embeddings"
	modelChat  = "qwen2.5vl:latest" //"hf.co/mradermacher/NuMarkdown-8B-Thinking-GGUF:Q4_K_M"
	modelEmbed = "nomic-embed-vision-v1.5"

	similarityThreshold = 0.80
	sourceDir           = "zarf/samples/videos/"
	sourceFileName      = "zarf/samples/videos/training.mp4"

	audioCfg = audio.Config{
		SetLanguage: "en",
		Temperature: 0.1,
		Threads:     4,
	}
)

func init() {
	if v := os.Getenv("LLM_CHAT_SERVER"); v != "" {
		urlChat = v
	}

	if v := os.Getenv("LLM_CHAT_MODEL"); v != "" {
		modelChat = v
	}

	if v := os.Getenv("LLM_EMBED_SERVER"); v != "" {
		urlEmbed = v
	}

	if v := os.Getenv("LLM_EMBED_MODEL"); v != "" {
		modelEmbed = v
	}
}

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
		Extract all the source code in the image and provide the raw text.
		Do not include any other text.
		Do not interpret the code in the image.
`

// =============================================================================

type keyFrame struct {
	fileName       string
	description    string
	classification string
	embedding      []float64
	startTime      float64
	duration       float64
	mimeType       string
	image          []byte
	code           string
}

// =============================================================================

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

	f := func(path string, d fs.DirEntry, err error) error {
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
	}

	if err := fs.WalkDir(os.DirFS(chunksDir), ".", f); err != nil {
		return fmt.Errorf("walk directory: %w", err)
	}

	return nil
}

func processChunk(ctx context.Context, llmChat *client.LLM, llmEmbed *client.LLM, adio *audio.Audio, sourceDir string, sourceFileName string, totalFramesTime float64, duration float64) error {
	fullPath := filepath.Join(sourceDir, "chunks", sourceFileName)

	fmt.Printf("\nRemoving the frames from the previous chunk: %s\n",
		filepath.Join(sourceDir, "frames"))

	// -------------------------------------------------------------------------
	// Remove any existing key frame files from any previous run.

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
	// Produce the set of key frames from the video.

	fmt.Printf("\nProcessing video chunk: %s\n", fullPath)

	if err := extractKeyFramesFromVideo(fullPath); err != nil {
		return fmt.Errorf("process video: %w", err)
	}

	// -------------------------------------------------------------------------
	// Iterate over all the key frame images that we extracted from the chunk
	// and collect the information so we can filter out the duplicates.

	fmt.Printf("\nProcessing images in directory: %s\n", sourceDir)

	keyFramefiles, err := getFilesFromDirectory(filepath.Join(sourceDir, "frames"))
	if err != nil {
		return fmt.Errorf("get files from directory: %w", err)
	}

	keyFrames := make([]keyFrame, len(keyFramefiles))

	for i, keyFrameFile := range keyFramefiles {
		fmt.Printf("\nProcessing image: %s\n", keyFrameFile)

		// ---------------------------------------------------------------------
		// Read the key frame and get the image data and mime type.

		image, mimeType, err := readImage(keyFrameFile)
		if err != nil {
			return fmt.Errorf("read image: %w", err)
		}

		// ---------------------------------------------------------------------
		// Create an embedding vector for the images. We will use this to compare
		// the images to each other and find the most similar ones.

		fmt.Println("\nGenerating embeddings for the image description:")

		embedding, err := llmEmbed.EmbedWithImage(ctx, "", image, mimeType)
		if err != nil {
			return fmt.Errorf("llm.EmbedText: %w", err)
		}

		fmt.Printf("%v...%v\n", embedding[0:3], embedding[len(embedding)-3:])

		// ---------------------------------------------------------------------
		// Store the key frame information.

		keyFrames[i] = keyFrame{
			fileName:  keyFrameFile,
			startTime: totalFramesTime,
			duration:  duration,
			mimeType:  mimeType,
			image:     image,
			embedding: embedding,
		}
	}

	// -------------------------------------------------------------------------
	// Find and remove any duplicate key frames.

	fmt.Println("\nUnique Frames:")

	// We have identifed that 80% of the time we have 10 or less unique key frames.
	unqKeyFrames := make([]keyFrame, 0, 10)

check:
	for _, keyFrame := range keyFrames {
		for _, unqKeyFrame := range unqKeyFrames {
			fmt.Printf("Checking image similarity between: %s - %s\n", unqKeyFrame.fileName, keyFrame.fileName)
			similarity := vector.CosineSimilarity(unqKeyFrame.embedding, keyFrame.embedding)
			fmt.Printf("  - Image similarity: %.3f\n", similarity)

			if similarity > similarityThreshold {
				fmt.Println("  - Image is similar to previous image")
				continue check
			}
		}

		unqKeyFrames = append(unqKeyFrames, keyFrame)
		fmt.Printf("  - FileName: %s - [%.4f, %.4f]\n", keyFrame.fileName, keyFrame.startTime, keyFrame.duration)
	}

	// -------------------------------------------------------------------------
	// Extract descriptions for the unique key frames.

	fmt.Println("\nExtracting frame descriptions:")

	for i, unqKeyFrame := range unqKeyFrames {
		fmt.Printf("Extracting description for image: %s\n", unqKeyFrame.fileName)

		description, err := llmChat.ChatCompletions(ctx, extractFrameInfoPrompt, client.WithImage(unqKeyFrame.mimeType, unqKeyFrame.image))
		if err != nil {
			return fmt.Errorf("chat completions: %w", err)
		}

		description = strings.Trim(description, "`")
		description = strings.TrimPrefix(description, "json")

		fmt.Printf("\nLLM DESC RESPONSE: %s\n", description)

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

		unqKeyFrames[i].description = descr.Text
		unqKeyFrames[i].classification = descr.Classification

		// ---------------------------------------------------------------------
		// Extract code samples from the key frame.

		if descr.Classification == "source code" {
			fmt.Println("  - Source code classification detected, extracting code...")
			code, err := llmChat.ChatCompletions(ctx, extractCodePrompt, client.WithImage(unqKeyFrame.mimeType, unqKeyFrame.image))
			if err != nil {
				return fmt.Errorf("chat completions: %w", err)
			}

			unqKeyFrames[i].code = code
			fmt.Printf("\nLLM CODE RESPONSE: %s\n", code)
		}
	}

	// -------------------------------------------------------------------------
	// Extract a transcription of the audio.

	if err := convertVideoToWav(fullPath); err != nil {
		return fmt.Errorf("converting video to wav: %w", err)
	}

	response, err := adio.Process(ctx, audioCfg, "zarf/samples/audio/output.wav")
	if err != nil {
		return fmt.Errorf("process audio: %w", err)
	}

	fmt.Printf("\nChunk audio transcription: %s\n", response.Text)

	// -------------------------------------------------------------------------

	fmt.Println("\nUnique Frames:")

	for _, f := range unqKeyFrames {
		fmt.Printf("\t- FileName: %s - [%.4f, %.4f, %s]\n", f.fileName, f.startTime, f.startTime, f.classification)
	}

	fmt.Println("\nDONE")
	return nil
}

// =============================================================================

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
