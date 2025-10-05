// This example provides a proof of concept for extracting transcriptions and,
// code examples from videos using the Ollama and a vision model. It then stores
// the extracted data in a MongoDB database for vector seaarch and RAG functionality.
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
	"errors"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"go.mongodb.org/mongo-driver/mongo"
	"golang.org/x/sync/errgroup"
)

var (
	urlVision   = "http://localhost:11434/v1/chat/completions"
	modelVision = "gemma3:12b-it-qat"

	urlTextEmbed   = "http://localhost:11434/v1/embeddings"
	modelTextEmbed = "bge-m3:latest"

	chunkSize        = 60
	frameDescTimeout = time.Second * 300
	frameWidth       = 640
	frameHeight      = 360

	videoFileName = "test_rag_video.mp4"
	videoDir      = "zarf/samples/videos/"
	framesDir     = "frames"
)

var ErrFFMPEG = errors.New("ffmpeg error")

func init() {
	if v := os.Getenv("LLM_VISION_SERVER"); v != "" {
		urlVision = v
	}

	if v := os.Getenv("LLM_VISION_MODEL"); v != "" {
		modelVision = v
	}

	if v := os.Getenv("LLM_TEXT_EMBED_SERVER"); v != "" {
		urlTextEmbed = v
	}

	if v := os.Getenv("LLM_TEXT_EMBED_MODEL"); v != "" {
		modelTextEmbed = v
	}
}

const promptKeyFrameDesc = `
	Provide a detailed description of this image in 300 words or less.
	Do not include any source code in the detailed description.
	Do not include any terminal output in the detailed description.
	
	Also, classify this image as: "source code", "diagram", "terminal", or "other" depending on the content it features the most.
	If icons are present in the middle of the image and blocking the main content, classify them as "icon".

	Extract all the text you see in the image and keep the formatting.
	Do not modify, enhance, or change any of the text you see.
	Do not add any new text that isn't part of the image.
	Keep any spacing or formatting of the text as it appears in the image.
	Place the text under the TEXT section in the final response.

	Provide the response using the following format with the provided section headers (** JSON DOCUMENT and ** TEXT):

		** JSON DOCUMENT
		{
			"description": "<image description>",
			"classification": "<image classification>"
		}
		
		** TEXT

	Encode any special characters that will be part of a JSON document.
	Make sure all text to be placed inside a JSON document is properly encoded and that the JSON is valid.
`

// =============================================================================

type keyFrame struct {
	fileName       string
	description    string
	classification string
	text           string
	embedding      []float64
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

	llmVision := client.NewLLM(urlVision, modelVision)
	llmTextEmbed := client.NewLLM(urlTextEmbed, modelTextEmbed)

	fmt.Print("\n---\n")

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

		err = processChunk(ctx, col, llmVision, llmTextEmbed, videoDir, videoFileName, videoChunkFile, startingVideoTime, duration)
		if err != nil {
			if errors.Is(err, ErrFFMPEG) {
				fmt.Printf("FFMPEG error processing chunk: %s\n", err)
				return nil
			}
			return fmt.Errorf("process chunk: %w", err)
		}

		return nil
	}

	if err := fs.WalkDir(os.DirFS(chunksDir), ".", f); err != nil {
		return fmt.Errorf("walk directory: %w", err)
	}

	return nil
}

func processChunk(ctx context.Context, col *mongo.Collection, llmVision *client.LLM, llmTextEmbed *client.LLM, videoDir string, videoFileName string, videoChunkFile string, startingVideoTime float64, duration float64) error {
	exists, err := existsDocument(ctx, col, videoFileName, videoChunkFile)
	if err != nil {
		return fmt.Errorf("exists document: %w", err)
	}
	if exists {
		fmt.Printf("Document exists: %s, %s\n", videoFileName, filepath.Base(videoChunkFile))
		return nil
	}

	transcription, err := extractAudioTranscription(videoChunkFile)
	if err != nil {
		return fmt.Errorf("extract audio transcription: %w", err)
	}

	if err := createKeyFrameFiles(videoChunkFile); err != nil {
		return fmt.Errorf("create key frame files: %w %w", ErrFFMPEG, err)
	}

	chunkName := filepath.Base(videoChunkFile)

	keyFrames, err := processKeyFrameFiles(chunkName, videoDir, llmVision)
	if err != nil {
		return fmt.Errorf("process key frame files: %w", err)
	}

	if len(keyFrames) == 0 {
		fmt.Println("No key frames found")
		return nil
	}

	// -------------------------------------------------------------------------

	fmt.Print("\n")

	var sb strings.Builder
	sb.WriteString(transcription)
	sb.WriteString("\n")
	for _, frame := range keyFrames {
		if frame.classification == "icon" || frame.classification == "other" {
			continue
		}
		sb.WriteString(frame.description)
		sb.WriteString("\n")
		if frame.classification == "source code" {
			sb.WriteString(frame.text)
			sb.WriteString("\n")
		}
	}

	input := sb.String()

	fmt.Print("\n")
	fmt.Printf("Video: %s\n", videoFileName)
	fmt.Printf("Chunk: %s\n", filepath.Base(videoChunkFile))
	fmt.Printf("Starting Video Time: %f\n", startingVideoTime)
	fmt.Printf("Duration: %f\n", duration)
	fmt.Printf("Input: %s\n", input)

	embed, err := llmTextEmbed.EmbedText(ctx, input)
	if err != nil {
		return fmt.Errorf("embed text: %w", err)
	}

	if err := insertDocument(ctx, col, embed, input, videoFileName, videoChunkFile, startingVideoTime, duration); err != nil {
		return fmt.Errorf("insert document: %w", err)
	}

	return nil
}

// =============================================================================

func splitVideoIntoChunks(videoPath string) error {
	fmt.Printf("Splitting video into chunks: %s\n", videoPath)

	ffmpegCommand := fmt.Sprintf("ffmpeg -i %s -c copy -map 0 -f segment -segment_time %d -reset_timestamps 1 -loglevel error zarf/samples/videos/chunks/output_%%05d.mp4", videoPath, chunkSize)
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

func extractAudioTranscription(videoChunkFile string) (string, error) {
	fmt.Println("Extracting audio transcription")

	queue := chunkSize + 5

	ffmpegCommand := fmt.Sprintf("ffmpeg -i %s -vn -af \"whisper=model=zarf/models/ggml-tiny.bin :destination=- :format=text :queue=%d\" -loglevel error -f null -", videoChunkFile, queue)
	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("error while running ffmpeg: %w: %s", err, string(out))
	}

	return string(out), nil
}

func processKeyFrameFiles(chunkName string, videoDir string, llmVision *client.LLM) ([]keyFrame, error) {
	fmt.Println("Processing key frames")

	fullpath := filepath.Join(videoDir, framesDir, chunkName)

	keyFramefiles, err := getFilesFromDirectory(fullpath)
	if err != nil {
		return nil, fmt.Errorf("get files from directory: %w", err)
	}

	var keyFrames []keyFrame

	switch l := len(keyFramefiles); l {
	case 0:
		return nil, nil

	case 1:
		keyFrames = []keyFrame{
			{fileName: keyFramefiles[0]},
		}

	default:
		first := 0
		last := l - 1
		keyFrames = []keyFrame{
			{fileName: keyFramefiles[first]},
			{fileName: keyFramefiles[last]},
		}
	}

	if err := createKeyFrameDescriptions(keyFrames, llmVision); err != nil {
		return nil, fmt.Errorf("create key frame descriptions: %w", err)
	}

	return keyFrames, nil
}

func createKeyFrameFiles(videoChunkFile string) error {
	fmt.Println("Creating key frame files")

	chunkName := filepath.Base(videoChunkFile)

	if err := os.RemoveAll(videoDir + "/" + framesDir + "/" + chunkName); err != nil {
		return fmt.Errorf("remove past work files: %w", err)
	}

	if err := os.MkdirAll(videoDir+"/"+framesDir+"/"+chunkName, 0755); err != nil {
		return fmt.Errorf("mkdirall: %w", err)
	}

	ffmpegCommand := fmt.Sprintf("ffmpeg -skip_frame nokey -i %s -vf \"select='gt(scene,0.05)',scale='if(gt(iw,ih),%d,-1)':'if(gt(ih,iw),%d,-1)'\" -fps_mode vfr -frame_pts true -loglevel error zarf/samples/videos/%s/%s/%%05d.png", videoChunkFile, frameWidth, frameHeight, framesDir, chunkName)

	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w: %s", err, string(out))
	}

	return nil
}

func createKeyFrameDescriptions(keyFrames []keyFrame, llmVision *client.LLM) error {
	fmt.Printf("Creating key frame descriptions: %d\n", len(keyFrames))

	semaphore := 1
	v := os.Getenv("OLLAMA_NUM_PARALLEL")
	if v != "" {
		var err error
		semaphore, err = strconv.Atoi(v)
		if err != nil {
			semaphore = 1
		}
	}

	ch := make(chan bool, semaphore)

	var g errgroup.Group

	for i, keyFrame := range keyFrames {
		g.Go(func() error {
			ch <- true
			defer func() {
				<-ch
			}()

			fmt.Printf("\t- Creating key frame description: %s\n", filepath.Base(keyFrame.fileName))

			image, mimeType, err := readImage(keyFrame.fileName)
			if err != nil {
				return fmt.Errorf("read image: %w", err)
			}

			ctx, cancel := context.WithTimeout(context.Background(), frameDescTimeout)
			defer cancel()

			p1 := client.WithImage(mimeType, image)
			p2 := client.WithParams(0.0, 0.1, 1)

			response, err := llmVision.ChatCompletions(ctx, promptKeyFrameDesc, p1, p2)
			if err != nil {
				return fmt.Errorf("chat completions: %w", err)
			}

			jsonDoc, text, err := extractParts(response)
			if err != nil {
				return fmt.Errorf("extract parts: %w: %s", err, response)
			}

			var descr struct {
				Description    string `json:"description"`
				Classification string `json:"classification"`
			}
			if err := json.Unmarshal([]byte(jsonDoc), &descr); err != nil {
				return fmt.Errorf("unmarshal: %w: %s", err, jsonDoc)
			}

			keyFrames[i].description = descr.Description
			keyFrames[i].classification = descr.Classification
			keyFrames[i].text = text

			return nil
		})
	}

	if err := g.Wait(); err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			fmt.Println("createKeyFrameDescriptions: context deadline exceeded")
			return nil
		}

		return fmt.Errorf("createKeyFrameDescriptions: %w", err)
	}

	return nil
}

func extractParts(description string) (string, string, error) {
	jsonMarker := "** JSON DOCUMENT"
	codeMarker := "** TEXT"

	jsonStart := strings.Index(description, jsonMarker)
	if jsonStart == -1 {
		return "", "", fmt.Errorf("JSON document marker not found")
	}

	codeStart := strings.Index(description, codeMarker)
	if codeStart == -1 {
		// If no source code marker, just return JSON part and empty code
		jsonDoc := strings.TrimSpace(description[jsonStart+len(jsonMarker):])
		return jsonDoc, "", nil
	}

	jsonDoc := strings.TrimSpace(description[jsonStart+len(jsonMarker) : codeStart])
	sourceCode := strings.TrimSpace(description[codeStart+len(codeMarker):])

	jsonDoc = strings.Trim(jsonDoc, "`")
	jsonDoc = strings.TrimPrefix(jsonDoc, "json")

	sourceCode = strings.Trim(sourceCode, "`")
	sourceCode = strings.TrimPrefix(sourceCode, "go")

	return jsonDoc, sourceCode, nil
}

func getFilesFromDirectory(directoryPath string) ([]string, error) {
	var files []string

	err := filepath.Walk(directoryPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() && (filepath.Ext(info.Name()) == ".png" || filepath.Ext(info.Name()) == ".jpg" || filepath.Ext(info.Name()) == ".jpeg" || filepath.Ext(info.Name()) == ".png") {
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
