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
	urlChat         = "http://localhost:11434/v1/chat/completions"
	urlImageEmbed   = "http://localhost:11439/v1/embeddings"
	modelChat       = "qwen2.5vl:latest"
	modelImageEmbed = "nomic-embed-vision-v1.5"

	similarityThreshold = 0.80
	sourceDir           = "zarf/samples/videos/"
	sourceFileName      = "training.mp4"

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

	if v := os.Getenv("LLM_IMAGE_EMBED_SERVER"); v != "" {
		urlImageEmbed = v
	}

	if v := os.Getenv("LLM_IMAGE_EMBED_MODEL"); v != "" {
		modelImageEmbed = v
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

type chunk struct {
	filePath      string
	duration      float64
	transcription string
	keyFrames     []keyFrame
}

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
	llmEmbed := client.NewLLM(urlImageEmbed, modelImageEmbed)

	adio, err := audio.New(client.NoopLogger, "zarf/audio/ggml-tiny.bin")
	if err != nil {
		return fmt.Errorf("starting audio: %w", err)
	}

	fmt.Print("\n---\n\n")

	// -------------------------------------------------------------------------

	videoPath := filepath.Join(sourceDir, sourceFileName)

	if err := splitVideoIntoChunks(videoPath); err != nil {
		return fmt.Errorf("splitting video into chunks: %w", err)
	}

	// -------------------------------------------------------------------------

	totalFramesTime := 0.0

	chunksDir := filepath.Join(sourceDir, "chunks")
	fmt.Printf("Processing video chunks in directory: %s\n", chunksDir)

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

		chunkFilePath := filepath.Join(sourceDir, "chunks", path)

		fmt.Print("\n=================================================\n\n")
		fmt.Printf("Processing chunk file: %s\n", chunkFilePath)

		duration, err := getVideoDuration(chunkFilePath)
		if err != nil {
			return fmt.Errorf("get video duration: %w", err)
		}

		// Defer the total time computation until after processing the chunk.
		defer func() {
			totalFramesTime += duration
		}()

		err = processChunk(ctx, llmChat, llmEmbed, adio, sourceDir, chunkFilePath, totalFramesTime, duration)
		if err != nil {
			return err
		}

		return nil
	}

	if err := fs.WalkDir(os.DirFS(chunksDir), ".", f); err != nil {
		return fmt.Errorf("walk directory: %w", err)
	}

	return nil
}

func processChunk(ctx context.Context, llmChat *client.LLM, llmEmbed *client.LLM, adio *audio.Audio, sourceDir string, chunkFilePath string, totalFramesTime float64, duration float64) error {
	transcription, err := extractAudioTranscription(ctx, chunkFilePath, adio)
	if err != nil {
		return fmt.Errorf("extract audio transcription: %w", err)
	}

	if err := createKeyFrameFiles(chunkFilePath); err != nil {
		return fmt.Errorf("create key frame files: %w", err)
	}

	keyFrames, err := processKeyFrameFiles(ctx, sourceDir, llmEmbed, llmChat, totalFramesTime, duration)
	if err != nil {
		return fmt.Errorf("extract key frames data: %w", err)
	}

	c := chunk{
		filePath:      chunkFilePath,
		duration:      duration,
		transcription: transcription,
		keyFrames:     keyFrames,
	}

	// NEXT IS TO STORE IN MONGO

	fmt.Print("\n")
	fmt.Printf("Chunk: %s\n", c.filePath)
	fmt.Printf("Duration: %f\n", c.duration)
	fmt.Printf("Transcription: %s\n", c.transcription)
	fmt.Printf("Key Frames: %d\n", len(c.keyFrames))
	for _, frame := range c.keyFrames {
		fmt.Printf("\t- %s\n", filepath.Base(frame.fileName))
		fmt.Printf("\t- %s\n", frame.classification)
		fmt.Printf("\t- %s\n", frame.description)
	}

	return nil
}

// =============================================================================

func splitVideoIntoChunks(videoPath string) error {
	fmt.Printf("Splitting video into chunks: %s\n", videoPath)

	ffmpegCommand := fmt.Sprintf("ffmpeg -i %s -c copy -map 0 -f segment -segment_time 15 -reset_timestamps 1 -loglevel error zarf/samples/videos/chunks/output_%%05d.mp4", videoPath)
	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %s, %w: %s", videoPath, err, string(out))
	}

	return nil
}

func getVideoDuration(filePath string) (float64, error) {
	fmt.Println("Getting video duration")

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

func extractAudioTranscription(ctx context.Context, chunkFilePath string, adio *audio.Audio) (string, error) {
	fmt.Println("Extracting audio transcription")

	if err := convertVideoToWav(chunkFilePath); err != nil {
		return "", fmt.Errorf("converting video to wav: %w", err)
	}

	response, err := adio.Process(ctx, audioCfg, "zarf/samples/audio/output.wav")
	if err != nil {
		return "", fmt.Errorf("process audio: %w", err)
	}

	return response.Text, nil
}

func convertVideoToWav(source string) error {
	// Ensure there is no previous file to allow ffmpeg to create the new one.
	_ = os.Remove("zarf/samples/audio/output.wav")

	ffmpegCommand := fmt.Sprintf("ffmpeg -i %s -ar 16000 -ac 1 -c:a pcm_s16le -loglevel error zarf/samples/audio/output.wav", source)
	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w: %s", err, string(out))
	}

	return nil
}

func processKeyFrameFiles(ctx context.Context, sourceDir string, llmEmbed *client.LLM, llmChat *client.LLM, startTime float64, duration float64) ([]keyFrame, error) {
	fmt.Println("Processing key frames")

	fullpath := filepath.Join(sourceDir, "frames")

	keyFramefiles, err := getFilesFromDirectory(fullpath)
	if err != nil {
		return nil, fmt.Errorf("get files from directory: %w", err)
	}

	keyFrames := make([]keyFrame, len(keyFramefiles))

	for i, keyFrameFile := range keyFramefiles {
		// ---------------------------------------------------------------------
		// Read the key frame and get the image data and mime type.

		image, mimeType, err := readImage(keyFrameFile)
		if err != nil {
			return nil, fmt.Errorf("read image: %w", err)
		}

		// ---------------------------------------------------------------------
		// Create an embedding vector for the images. We will use this to compare
		// the images to each other and find the most similar ones.

		embedding, err := llmEmbed.EmbedWithImage(ctx, "", image, mimeType)
		if err != nil {
			return nil, fmt.Errorf("llm.EmbedText: %w", err)
		}

		// ---------------------------------------------------------------------
		// Store the key frame information.

		keyFrames[i] = keyFrame{
			fileName:  keyFrameFile,
			startTime: startTime,
			duration:  duration,
			mimeType:  mimeType,
			image:     image,
			embedding: embedding,
		}
	}

	unqKeyFrames := removeDuplicateKeyFrames(keyFrames)

	if err := createKeyFrameDescriptions(ctx, unqKeyFrames, llmChat); err != nil {
		return nil, fmt.Errorf("create key frame descriptions: %w", err)
	}

	return unqKeyFrames, nil
}

func createKeyFrameFiles(chunkFilePath string) error {
	fmt.Println("Creating key frame files")

	if err := removePastKeyFrameFiles(sourceDir); err != nil {
		return fmt.Errorf("remove past work files: %w", err)
	}

	ffmpegCommand := fmt.Sprintf("ffmpeg -skip_frame nokey -i %s -frame_pts true -fps_mode vfr -loglevel error zarf/samples/videos/frames/%%05d.jpg", chunkFilePath)

	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w: %s", err, string(out))
	}

	return nil
}

func removePastKeyFrameFiles(path string) error {
	previousFrames, err := fs.Glob(os.DirFS(path), "frames/*")
	if err != nil {
		return fmt.Errorf("glob: %w", err)
	}

	for _, previousFrame := range previousFrames {
		if err := os.Remove(filepath.Join(path, previousFrame)); err != nil {
			return fmt.Errorf("remove previous frame: %w", err)
		}
	}

	return nil
}

func removeDuplicateKeyFrames(keyFrames []keyFrame) []keyFrame {
	// We have identifed that 80% of the time we have 10 or less unique key frames.
	unqKeyFrames := make([]keyFrame, 0, 10)

check:
	for _, keyFrame := range keyFrames {
		for _, unqKeyFrame := range unqKeyFrames {
			similarity := vector.CosineSimilarity(unqKeyFrame.embedding, keyFrame.embedding)

			if similarity > similarityThreshold {
				continue check
			}
		}

		unqKeyFrames = append(unqKeyFrames, keyFrame)
	}

	return unqKeyFrames
}

func createKeyFrameDescriptions(ctx context.Context, unqKeyFrames []keyFrame, llmChat *client.LLM) error {
	fmt.Printf("Creating key frame descriptions: %d\n", len(unqKeyFrames))

	for i, unqKeyFrame := range unqKeyFrames {
		fmt.Printf("\t- Creating key frame description: %s\n", filepath.Base(unqKeyFrame.fileName))

		description, err := llmChat.ChatCompletions(ctx, extractFrameInfoPrompt, client.WithImage(unqKeyFrame.mimeType, unqKeyFrame.image))
		if err != nil {
			return fmt.Errorf("chat completions: %w", err)
		}

		description = strings.Trim(description, "`")
		description = strings.TrimPrefix(description, "json")

		var descr struct {
			Text           string `json:"text"`
			Classification string `json:"classification"`
		}
		if err := json.Unmarshal([]byte(description), &descr); err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		if descr.Classification == "icon" {
			continue
		}

		unqKeyFrames[i].description = descr.Text
		unqKeyFrames[i].classification = descr.Classification

		// ---------------------------------------------------------------------
		// Extract code samples from the key frame.

		if descr.Classification == "source code" {
			code, err := llmChat.ChatCompletions(ctx, extractCodePrompt, client.WithImage(unqKeyFrame.mimeType, unqKeyFrame.image))
			if err != nil {
				return fmt.Errorf("chat completions: %w", err)
			}

			unqKeyFrames[i].code = code
		}
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
