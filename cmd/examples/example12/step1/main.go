// This example provides a proof of concept for extracting transcriptions, code,
// diagrams, images, and text from videos using the Ollama and Embedding services.
// It then stores the extracted data in a MongoDB database for vector seaarch.
//
// # Running the example:
//
//	$ make example12-step1
//
// # This requires running the following commands:
//
//	$ make ollama-up    // This starts the Ollama service.
//	$ make embedding-up // This starts the Embedding service.
//	$ make compose-up   // This starts Mongo service.

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
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"go.mongodb.org/mongo-driver/mongo"
)

var (
	urlChat         = "http://localhost:11434/v1/chat/completions"
	urlTextEmbed    = "http://localhost:11434/v1/embeddings"
	urlImageEmbed   = "http://localhost:11439/v1/embeddings"
	modelChat       = "gemma3:4b-it-qat"
	modelTextEmbed  = "bge-m3:latest"
	modelImageEmbed = "nomic-embed-vision-v1.5"

	similarityThreshold = 0.80
	videoDir            = "zarf/samples/videos/"
	videoFileName       = "test_rag_video.mp4"

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

	if v := os.Getenv("LLM_TEXT_EMBED_SERVER"); v != "" {
		urlTextEmbed = v
	}

	if v := os.Getenv("LLM_TEXT_EMBED_MODEL"); v != "" {
		modelTextEmbed = v
	}

	if v := os.Getenv("LLM_IMAGE_EMBED_SERVER"); v != "" {
		urlImageEmbed = v
	}

	if v := os.Getenv("LLM_IMAGE_EMBED_MODEL"); v != "" {
		modelImageEmbed = v
	}
}

const extractKeyFramePrompt = `
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
	llmTextEmbed := client.NewLLM(urlTextEmbed, modelTextEmbed)
	llmImageEmbed := client.NewLLM(urlImageEmbed, modelImageEmbed)

	adio, err := audio.New(client.NoopLogger, "zarf/audio/ggml-tiny.bin")
	if err != nil {
		return fmt.Errorf("starting audio: %w", err)
	}

	fmt.Print("\n---\n\n")

	// -------------------------------------------------------------------------

	fmt.Println("\nConnecting to MongoDB")

	dbClient, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("mongodb.Connect: %w", err)
	}

	fmt.Println("Initializing Database")

	col, err := initDB(ctx, dbClient)
	if err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	// -------------------------------------------------------------------------

	videoPath := filepath.Join(videoDir, videoFileName)

	if err := splitVideoIntoChunks(videoPath); err != nil {
		return fmt.Errorf("splitting video into chunks: %w", err)
	}

	// -------------------------------------------------------------------------

	startingVideoTime := 0.0

	chunksDir := filepath.Join(videoDir, "chunks")
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

		videoChunkFile := filepath.Join(videoDir, "chunks", path)

		fmt.Print("\n=================================================\n\n")
		fmt.Printf("Processing chunk file: %s\n", videoChunkFile)

		duration, err := getVideoDuration(videoChunkFile)
		if err != nil {
			return fmt.Errorf("get video duration: %w", err)
		}

		// Defer the total time computation until after processing the chunk.
		defer func() {
			startingVideoTime += duration
		}()

		err = processChunk(ctx, col, llmChat, llmTextEmbed, llmImageEmbed, adio, videoDir, videoFileName, videoChunkFile, startingVideoTime, duration)
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

func processChunk(ctx context.Context, col *mongo.Collection, llmChat *client.LLM, llmTextEmbed *client.LLM, llmImageEmbed *client.LLM, adio *audio.Audio, videoDir string, videoFileName string, videoChunkFile string, startingVideoTime float64, duration float64) error {
	exists, err := existsDocument(ctx, col, videoFileName, videoChunkFile)
	if err != nil {
		return fmt.Errorf("exists document: %w", err)
	}
	if exists {
		fmt.Printf("Document exists: %s, %s\n", videoFileName, filepath.Base(videoChunkFile))
		return nil
	}

	transcription, err := extractAudioTranscription(ctx, videoChunkFile, adio)
	if err != nil {
		return fmt.Errorf("extract audio transcription: %w", err)
	}

	if err := createKeyFrameFiles(videoChunkFile); err != nil {
		return fmt.Errorf("create key frame files: %w", err)
	}

	keyFrames, err := processKeyFrameFiles(ctx, videoDir, llmImageEmbed, llmChat, duration)
	if err != nil {
		return fmt.Errorf("process key frame files: %w", err)
	}

	fmt.Print("\n")
	fmt.Printf("Video: %s\n", videoFileName)
	fmt.Printf("Chunk: %s\n", filepath.Base(videoChunkFile))
	fmt.Printf("Starting Video Time: %f\n", startingVideoTime)
	fmt.Printf("Duration: %f\n", duration)
	fmt.Printf("Transcription: %s\n", transcription)
	fmt.Printf("Key Frames: %d\n", len(keyFrames))
	for _, frame := range keyFrames {
		fmt.Printf("\t- %s\n", filepath.Base(frame.fileName))
		fmt.Printf("\t- %s\n", frame.classification)
		fmt.Printf("\t- %s\n", frame.description)
	}

	// -------------------------------------------------------------------------

	var sb strings.Builder
	sb.WriteString(transcription)
	sb.WriteString("\n")
	for _, frame := range keyFrames {
		if frame.classification == "icon" || frame.classification == "other" {
			continue
		}
		sb.WriteString(frame.description)
		sb.WriteString("\n")
		if frame.classification == "source_code" {
			sb.WriteString(frame.code)
			sb.WriteString("\n")
		}
	}

	input := sb.String()

	if err := insertDocument(ctx, col, llmTextEmbed, input, videoFileName, videoChunkFile, startingVideoTime, duration); err != nil {
		return fmt.Errorf("insert document: %w", err)
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

func getVideoDuration(videoChunkFile string) (float64, error) {
	fmt.Println("Getting video duration")

	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json",
		"-show_entries", "format=duration", videoChunkFile)

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

func extractAudioTranscription(ctx context.Context, videoChunkFile string, adio *audio.Audio) (string, error) {
	fmt.Println("Extracting audio transcription")

	if err := convertVideoToWav(videoChunkFile); err != nil {
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

func processKeyFrameFiles(ctx context.Context, videoDir string, llmEmbed *client.LLM, llmChat *client.LLM, duration float64) ([]keyFrame, error) {
	fmt.Println("Processing key frames")

	fullpath := filepath.Join(videoDir, "frames")

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

func createKeyFrameFiles(videoChunkFile string) error {
	fmt.Println("Creating key frame files")

	if err := removePastKeyFrameFiles(videoDir); err != nil {
		return fmt.Errorf("remove past work files: %w", err)
	}

	ffmpegCommand := fmt.Sprintf("ffmpeg -skip_frame nokey -i %s -frame_pts true -fps_mode vfr -loglevel error zarf/samples/videos/frames/%%05d.jpg", videoChunkFile)

	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w: %s", err, string(out))
	}

	return nil
}

func removePastKeyFrameFiles(videoDir string) error {
	previousFrames, err := fs.Glob(os.DirFS(videoDir), "frames/*")
	if err != nil {
		return fmt.Errorf("glob: %w", err)
	}

	for _, previousFrame := range previousFrames {
		if err := os.Remove(filepath.Join(videoDir, previousFrame)); err != nil {
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

		description, err := llmChat.ChatCompletions(ctx, extractKeyFramePrompt, client.WithImage(unqKeyFrame.mimeType, unqKeyFrame.image))
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
			return fmt.Errorf("unmarshal: %w: %s", err, description)
		}

		unqKeyFrames[i].description = descr.Text
		unqKeyFrames[i].classification = descr.Classification

		// ---------------------------------------------------------------------
		// Extract code samples from the key frame.

		if descr.Classification == "source code" {
			fmt.Printf("\t- Extracting source code: %s\n", filepath.Base(unqKeyFrame.fileName))
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
