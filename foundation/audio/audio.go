// Package audio provides support for processing audio files and generating
// text transcriptions.
package audio

import (
	"context"
	"fmt"
	"time"
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
	ch  chan *whisp
}

func New(log Logger, modelPath string, concurrency int) (*Audio, error) {
	a := Audio{
		log: log,
		ch:  make(chan *whisp, concurrency),
	}

	for range concurrency {
		log(context.Background(), "*********************> LOADING MODEL")

		whs, err := newWhisper(log, modelPath)
		if err != nil {
			return nil, fmt.Errorf("new: %w", err)
		}

		a.ch <- whs
	}

	return &a, nil
}

func (a *Audio) Process(ctx context.Context, cfg Config, audioFile string) (WhisperResponse, error) {
	a.log(ctx, "text-processing", "status", "started")
	defer a.log(ctx, "text-processing", "status", "completed")

	whs, err := a.acquire(ctx)
	if err != nil {
		return WhisperResponse{}, fmt.Errorf("acquire: %w", err)
	}
	defer func() {
		a.log(ctx, "text-processing", "status", "releasing whisper model")
		a.release(whs)
	}()

	a.log(ctx, "text-processing", "status", "acquired whisper model")

	return whs.Process(ctx, cfg, audioFile)
}

func (a *Audio) acquire(ctx context.Context) (*whisp, error) {
	select {
	case whs := <-a.ch:
		return whs, nil

	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (a *Audio) release(whs *whisp) {
	select {
	case a.ch <- whs:
	default:
	}
}
