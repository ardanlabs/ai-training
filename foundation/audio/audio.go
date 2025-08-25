// Package audio provides support for processing audio files and generating
// text transcriptions.
package audio

import (
	"context"
	"fmt"
	"time"

	whisper2 "github.com/ardanlabs/ai-training/foundation/audio/whisper.cpp/bindings/go"
)

type Logger func(ctx context.Context, msg string, args ...any)

// =============================================================================

type Error struct {
	Message string
}

func newError(format string, a ...any) *Error {
	return &Error{
		Message: fmt.Sprintf(format, a...),
	}
}

func (e *Error) Error() string {
	return e.Message
}

// =============================================================================

type Config struct {
	SetLanguage          string
	Language             string
	Translate            bool
	Temperature          float32
	Prompt               string
	Offset               time.Duration
	Duration             time.Duration
	Threads              uint
	MaxLen               uint
	MaxTokens            uint
	WordThold            float32
	Verbose              bool
	SetSegmentTimestamps bool
	SetTokenTimestamps   bool
}

type Audio struct {
	log Logger
	whs *whisp
}

func New(log Logger, modelPath string) (*Audio, error) {
	a := Audio{
		log: log,
	}

	params := whisper2.DefaultInitParams()
	params.UseGpu = true
	params.FlashAttn = true

	whs, err := newWhisper(log, modelPath, params)
	if err != nil {
		return nil, fmt.Errorf("new: %w", err)
	}

	a.whs = whs

	return &a, nil
}

func (a *Audio) Process(ctx context.Context, cfg Config, audioFile string) (WhisperResponse, error) {
	a.log(ctx, "text-processing", "status", "started")
	defer a.log(ctx, "text-processing", "status", "completed")

	return a.whs.Process(ctx, cfg, audioFile)
}
