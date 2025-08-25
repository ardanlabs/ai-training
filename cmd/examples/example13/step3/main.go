package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/ardanlabs/ai-training/foundation/audio"
	"github.com/ardanlabs/ai-training/foundation/client"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	if err := convertVideoToWav("zarf/samples/videos/output_0089.mp4"); err != nil {
		return fmt.Errorf("converting video to wav: %w", err)
	}

	adio, err := audio.New(client.StdoutLogger, "zarf/audio/ggml-tiny.bin")
	if err != nil {
		return fmt.Errorf("starting audio: %w", err)
	}

	audioCfg := audio.Config{
		SetLanguage:          "en",
		Translate:            false,
		Temperature:          0.1,
		Prompt:               "",
		Offset:               0,
		Duration:             0,
		Threads:              4,
		MaxLen:               0,
		MaxTokens:            0,
		WordThold:            0,
		Verbose:              false,
		SetSegmentTimestamps: false,
		SetTokenTimestamps:   false,
	}

	response, err := adio.Process(ctx, audioCfg, "zarf/samples/audio/output.wav")
	if err != nil {
		return fmt.Errorf("process audio: %w", err)
	}

	fmt.Print("\n")
	fmt.Println(response.Text)

	return nil
}

func convertVideoToWav(source string) error {
	fmt.Println("Processing Video ...")
	defer fmt.Println("\nDONE Processing Video")

	// Ensure there is no previous file to allow ffmpeg to create the new one.
	_ = os.Remove("zarf/samples/audio/output.wav")

	ffmpegCommand := fmt.Sprintf("ffmpeg -i %s -ar 16000 -ac 1 -c:a pcm_s16le -loglevel error zarf/samples/audio/output.wav", source)
	out, err := exec.Command("/bin/sh", "-c", ffmpegCommand).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error while running ffmpeg: %w: %s", err, string(out))
	}

	return nil
}
