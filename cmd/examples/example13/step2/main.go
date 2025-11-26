// This example shows you how to use yzma to execute a simple prompt
// against a vision model using llamacpp directly via yzma and a native Go
// application.
//
// # Running the example:
//
//	$ make example13-step2

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/install"
	"github.com/ardanlabs/kronk"
	"github.com/ardanlabs/kronk/model"
	"github.com/hybridgroup/yzma/pkg/download"
)

const (
	modelURL  = "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true"
	projURL   = "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true"
	imageFile = "zarf/samples/gallery/giraffe.jpg"
	libPath   = "zarf/llamacpp"
	modelPath = "zarf/models"
)

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelFile, projFile, err := installSystem()
	if err != nil {
		return fmt.Errorf("unable to install system: %w", err)
	}

	krn, err := newKronk(modelFile, projFile)
	if err != nil {
		return fmt.Errorf("unable to init kronk: %w", err)
	}
	defer krn.Unload()

	// -------------------------------------------------------------------------

	question := "What is in this picture?"

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	ch, err := performChat(ctx, krn, question, imageFile)
	if err != nil {
		return fmt.Errorf("perform chat: %w", err)
	}

	if err := modelResponse(krn, ch); err != nil {
		return fmt.Errorf("model response: %w", err)
	}

	return nil
}

func installSystem() (string, string, error) {
	if err := install.LlamaCPP(libPath, download.CPU, true); err != nil {
		return "", "", fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelFile, err := install.Model(modelURL, modelPath)
	if err != nil {
		return "", "", fmt.Errorf("unable to install model: %w", err)
	}

	projFile, err := install.Model(projURL, modelPath)
	if err != nil {
		return "", "", fmt.Errorf("unable to install model: %w", err)
	}

	return modelFile, projFile, nil
}

func newKronk(modelFile string, projFile string) (*kronk.Kronk, error) {
	if err := kronk.Init(libPath, kronk.LogSilent); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	const concurrency = 1

	krn, err := kronk.New(concurrency, modelFile, projFile, model.Config{})
	if err != nil {
		return nil, fmt.Errorf("unable to create inference model: %w", err)
	}

	fmt.Println("- contextWindow:", krn.ModelConfig().ContextWindow)
	fmt.Println("- embeddings   :", krn.ModelConfig().Embeddings)

	return krn, nil
}

func performChat(ctx context.Context, krn *kronk.Kronk, question string, imageFile string) (<-chan model.ChatResponse, error) {
	fmt.Printf("\nQuestion: %s\n", question)

	ch, err := krn.VisionStreaming(ctx, model.VisionRequest{
		ImageFile: imageFile,
		Message: model.ChatMessage{
			Role: "user", Content: question,
		},
		Params: model.Params{
			MaxTokens: 2048,
		},
	})

	if err != nil {
		return nil, fmt.Errorf("vision streaming: %w", err)
	}

	return ch, nil
}

func modelResponse(krn *kronk.Kronk, ch <-chan model.ChatResponse) error {
	fmt.Print("\nMODEL> ")

	var reasoning bool
	var lr model.ChatResponse

loop:
	for resp := range ch {
		switch resp.Choice[0].FinishReason {
		case model.FinishReasonStop:
			break loop

		case model.FinishReasonError:
			return fmt.Errorf("error from model: %s", resp.Choice[0].Delta.Content)
		}

		if resp.Choice[0].Delta.Reasoning != "" {
			fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choice[0].Delta.Reasoning)
			reasoning = true
			continue
		}

		if reasoning {
			reasoning = false
			fmt.Print("\n\n")
		}

		fmt.Printf("%s", resp.Choice[0].Delta.Content)
		lr = resp
	}

	// -------------------------------------------------------------------------

	contextTokens := lr.Usage.InputTokens + lr.Usage.CompletionTokens
	contextWindow := krn.ModelConfig().ContextWindow
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("\n\n\u001b[90mInput: %d  Reasoning: %d  Completion: %d  Output: %d  Window: %d (%.0f%% of %.0fK) TPS: %.2f\u001b[0m\n",
		lr.Usage.InputTokens, lr.Usage.ReasoningTokens, lr.Usage.CompletionTokens, lr.Usage.OutputTokens, contextTokens, percentage, of, lr.Usage.TokensPerSecond)

	return nil
}
