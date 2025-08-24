package audio

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"runtime/debug"
	"strings"
	"time"

	whisper2 "github.com/ardanlabs/ai-training/foundation/audio/whisper.cpp/bindings/go"
	"github.com/ardanlabs/ai-training/foundation/audio/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/go-audio/wav"
)

var tokenPattern = regexp.MustCompile(`\[_TT_\d+\]`)

// =============================================================================

type SegmentData struct {
	Text  string
	Start time.Duration
	End   time.Duration
}

type TokenData struct {
	Text  string
	Start time.Duration
	End   time.Duration
}

type WhisperResponse struct {
	Text     string
	Segments []SegmentData
	Tokens   []TokenData
	Task     string
	Language string
	Duration float64
	Error    error
}

// =============================================================================

type whisp struct {
	log   Logger
	model whisper.Model
}

func newWhisper(log Logger, modelPath string, params whisper2.InitParams) (*whisp, error) {
	model, err := whisper.NewWithParams(modelPath, params)
	if err != nil {
		return nil, fmt.Errorf("new: %w", err)
	}

	whs := whisp{
		log:   log,
		model: model,
	}

	return &whs, nil
}

func (whs *whisp) Close() {
	whs.model.Close()
}

func (whs *whisp) Process(ctx context.Context, cfg Config, audioFile string) (WhisperResponse, error) {
	context, err := whs.model.NewContext()
	if err != nil {
		return WhisperResponse{}, fmt.Errorf("model: %w", err)
	}

	if err := setContext(cfg, context); err != nil {
		return WhisperResponse{}, fmt.Errorf("context: %w", err)
	}

	// -------------------------------------------------------------------------

	f, err := os.Open(audioFile)
	if err != nil {
		return WhisperResponse{}, fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	dec, err := whs.convertAudio(ctx, f)
	if err != nil {
		return WhisperResponse{}, err
	}

	// -------------------------------------------------------------------------

	ch := make(chan WhisperResponse, 1)

	go func() {
		whs.log(ctx, "text-processing", "status", "started translation processing")
		defer func() {
			whs.log(ctx, "text-processing", "status", "completed translation processing")
			if r := recover(); r != nil {
				whs.log(ctx, "text-processing", "PANIC", r)
				whs.log(ctx, "text-processing", "STACK", debug.Stack())
			}
		}()

		buf, err := dec.FullPCMBuffer()
		if err != nil {
			ch <- WhisperResponse{Error: fmt.Errorf("decoder: %w", err)}
			return
		}

		data := buf.AsFloat32Buffer().Data

		if len(data) == 0 {
			ch <- WhisperResponse{Error: fmt.Errorf("process: data is empty")}
			return
		}

		if err := context.Process(data, nil, nil); err != nil {
			ch <- WhisperResponse{Error: fmt.Errorf("process: %w", err)}
			return
		}

		var response WhisperResponse
		var textParts []string
		var duration float64

		for {
			segment, err := context.NextSegment()
			if err != nil {
				if err == io.EOF {
					break
				}
				ch <- WhisperResponse{Error: fmt.Errorf("next segment: %w", err)}
				return
			}

			textParts = append(textParts, segment.Text)

			if segment.End.Seconds() > duration {
				duration = segment.End.Seconds()
			}

			if cfg.SetSegmentTimestamps {
				response.Segments = append(response.Segments, SegmentData{
					Text:  segment.Text,
					Start: segment.Start,
					End:   segment.End,
				})
			}

			if cfg.SetTokenTimestamps {
				for _, token := range segment.Tokens {
					if token.Text == "[_BEG_]" || tokenPattern.MatchString(token.Text) {
						continue
					}

					response.Tokens = append(response.Tokens, TokenData{
						Text:  token.Text,
						Start: token.Start,
						End:   token.End,
					})
				}
			}
		}

		response.Text = strings.Join(textParts, " ")

		if cfg.Verbose {
			response.Task = "transcribe"
			response.Duration = duration
			response.Language = context.GetLanguageID(true)
		}

		ch <- response
	}()

	result := <-ch
	return result, result.Error
}

func (whs *whisp) convertAudio(ctx context.Context, r io.ReadSeeker) (*wav.Decoder, error) {
	dec := wav.NewDecoder(r)
	dec.ReadInfo()

	if dec.WavAudioFormat == 1 && dec.SampleRate == whisper.SampleRate && dec.NumChans == 1 {
		return dec, nil
	}

	dec.Seek(0, 0)

	// -------------------------------------------------------------------------

	var b bytes.Buffer

	whs.log(ctx, "text-processing", "status", "started ffmpeg conversion")
	defer whs.log(ctx, "text-processing", "status", "completed ffmpeg conversion")

	cmd := exec.CommandContext(ctx, "ffmpeg", "-i", "pipe:0", "-bitexact", "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", "pipe:1")
	cmd.Stdout = &b
	cmd.Stdin = r

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("run: %w", err)
	}

	// -------------------------------------------------------------------------

	// BILL: ffmpeg is writing the chunk we need as "WAVE" and not "data" which
	//       is what whisper is expecting.

	data := b.Bytes()
	copy(data[8:], "data")

	// -------------------------------------------------------------------------

	dec2 := wav.NewDecoder(bytes.NewReader(data))
	dec2.ReadInfo()

	if dec2.WavAudioFormat != 1 {
		return nil, newError("decoder: unsupported audio file, we support .wav")
	}

	if dec2.SampleRate != whisper.SampleRate {
		return nil, newError("decoder: unsupported sample rate: %d, we support %d", dec.SampleRate, whisper.SampleRate)
	}

	if dec2.NumChans != 1 {
		return nil, newError("decoder: unsupported number of channels: %d, we support 1", dec.NumChans)
	}

	dec2.Seek(0, 0)

	return dec2, nil
}

func setContext(cfg Config, context whisper.Context) error {
	switch cfg.SetLanguage {
	case "", "auto":
		if err := context.SetLanguage("auto"); err != nil {
			return newError("set language: %s", err)
		}
	default:
		if err := context.SetLanguage(cfg.SetLanguage); err != nil {
			return newError("set language: %s", err)
		}
	}

	if cfg.Translate {
		if !context.IsMultilingual() {
			return newError("set translate: model does not support translation")
		}
		context.SetTranslate(true)
	}

	if cfg.Temperature < 0 {
		return newError("set temperature: invalid temperature, %v", cfg.Offset)
	}
	if cfg.Temperature > 0 {
		context.SetTemperature(cfg.Temperature)
	}

	if cfg.Prompt != "" {
		context.SetInitialPrompt(cfg.Prompt)
	}

	if cfg.Offset < 0 {
		return newError("set offset: invalid offset, %v", cfg.Offset)
	}
	if cfg.Offset > 0 {
		context.SetOffset(cfg.Offset)
	}

	if cfg.Duration < 0 {
		return newError("set duration: invalid duration, %v", cfg.Duration)
	}
	if cfg.Duration > 0 {
		context.SetDuration(cfg.Duration)
	}

	if cfg.Threads > 0 {
		context.SetThreads(cfg.Threads)
	}

	if cfg.MaxLen > 0 {
		context.SetMaxSegmentLength(cfg.MaxLen)
	}

	if cfg.MaxTokens > 0 {
		context.SetMaxTokensPerSegment(cfg.MaxTokens)
	}

	if cfg.WordThold < 0 {
		return newError("set wordthold: invalid wordthold, %v", cfg.WordThold)
	}
	if cfg.WordThold > 0 {
		context.SetTokenThreshold(cfg.WordThold)
	}

	if cfg.SetTokenTimestamps {
		context.SetTokenTimestamps(true)
	}

	return nil
}
