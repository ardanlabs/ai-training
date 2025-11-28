// This example shows you how to create a simple chat application against an
// inference model using llamacpp directly via yzma and a native Go application.
//
// # Running the example:
//
//	$ make example13-step1

package main

import (
	"bufio"
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
	modelURL = "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf?download=true"
	// modelURL  = "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q8_0.gguf?download=true"
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
	modelFile, err := installSystem()
	if err != nil {
		return fmt.Errorf("unable to installation system: %w", err)
	}

	krn, err := newKronk(modelFile)
	if err != nil {
		return fmt.Errorf("unable to init kronk: %w", err)
	}
	defer krn.Unload()

	// -------------------------------------------------------------------------

	tools := tools()

	var messages []model.ChatMessage

	for {
		messages, err = userInput(messages)
		if err != nil {
			return fmt.Errorf("user input: %w", err)
		}

		messages, err = func() ([]model.ChatMessage, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			ch, err := performChat(ctx, krn, messages, tools)
			if err != nil {
				return nil, fmt.Errorf("unable to perform chat: %w", err)
			}

			messages, err = modelResponse(krn, messages, ch)
			if err != nil {
				return nil, fmt.Errorf("model response: %w", err)
			}

			return messages, nil
		}()

		if err != nil {
			return fmt.Errorf("unable to perform chat: %w", err)
		}
	}
}

func installSystem() (string, error) {
	if err := install.LlamaCPP(libPath, download.CPU, true); err != nil {
		return "", fmt.Errorf("unable to install llamacpp: %w", err)
	}

	modelFile, err := install.Model(modelURL, modelPath)
	if err != nil {
		return "", fmt.Errorf("unable to install model: %w", err)
	}

	return modelFile, nil
}

func newKronk(modelFile string) (*kronk.Kronk, error) {
	if err := kronk.Init(libPath, kronk.LogSilent); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	const modelInstances = 1

	krn, err := kronk.New(modelInstances, model.Config{
		ModelFile: modelFile,
	})

	if err != nil {
		return nil, fmt.Errorf("unable to create inference model: %w", err)
	}

	fmt.Println("- contextWindow:", krn.ModelConfig().ContextWindow)
	fmt.Println("- embeddings   :", krn.ModelConfig().Embeddings)
	fmt.Println("- isGPT        :", krn.ModelInfo().IsGPT)

	return krn, nil
}

func userInput(messages []model.ChatMessage) ([]model.ChatMessage, error) {
	fmt.Print("\nUSER> ")

	reader := bufio.NewReader(os.Stdin)

	userInput, err := reader.ReadString('\n')
	if err != nil {
		return messages, fmt.Errorf("unable to read user input: %w", err)
	}

	messages = append(messages, model.ChatMessage{
		Role:    "user",
		Content: userInput,
	})

	return messages, nil
}

func tools() []model.Tool {
	tool := model.NewToolFunction(
		"get_weather",
		"Get the weather for a place",
		model.ToolParameter{
			Name:        "location",
			Type:        "string",
			Description: "The location to get the weather for, e.g. San Francisco, CA",
		},
	)

	return []model.Tool{tool}
}

func performChat(ctx context.Context, krn *kronk.Kronk, messages []model.ChatMessage, tools []model.Tool) (<-chan model.ChatResponse, error) {
	ch, err := krn.ChatStreaming(ctx, model.ChatRequest{
		Messages: messages,
		Tools:    tools,
		Params: model.Params{
			MaxTokens: 2048,
		},
	})

	if err != nil {
		return nil, fmt.Errorf("chat streaming: %w", err)
	}

	return ch, nil
}

func modelResponse(krn *kronk.Kronk, messages []model.ChatMessage, ch <-chan model.ChatResponse) ([]model.ChatMessage, error) {
	fmt.Print("\nMODEL> ")

	var reasoning bool
	var lr model.ChatResponse

loop:
	for resp := range ch {
		lr = resp

		switch resp.Choice[0].FinishReason {
		case model.FinishReasonError:
			return messages, fmt.Errorf("error from model: %s", resp.Choice[0].Delta.Content)

		case model.FinishReasonStop:
			messages = append(messages, model.ChatMessage{
				Role:    "assistant",
				Content: resp.Choice[0].Delta.Content,
			})
			break loop

		case model.FinishReasonTool:
			fmt.Println()
			fmt.Printf("\u001b[92mModel Asking For Tool Call:\nToolID[%s]: %s(%s)\u001b[0m\n",
				resp.Choice[0].Delta.ToolCalls[0].ID,
				resp.Choice[0].Delta.ToolCalls[0].Name,
				resp.Choice[0].Delta.ToolCalls[0].Arguments,
			)

			messages = append(messages, model.ChatMessage{
				Role: "tool",
				Content: fmt.Sprintf("Tool call %s: %s(%v)",
					resp.Choice[0].Delta.ToolCalls[0].ID,
					resp.Choice[0].Delta.ToolCalls[0].Name,
					resp.Choice[0].Delta.ToolCalls[0].Arguments),
			})
			break loop

		default:
			if resp.Choice[0].Delta.Reasoning != "" {
				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choice[0].Delta.Reasoning)
				reasoning = true
				continue
			}

			if reasoning {
				reasoning = false
				fmt.Println()
			}

			fmt.Printf("%s", resp.Choice[0].Delta.Content)
		}
	}

	// -------------------------------------------------------------------------

	contextTokens := lr.Usage.InputTokens + lr.Usage.CompletionTokens
	contextWindow := krn.ModelConfig().ContextWindow
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("\n\u001b[90mInput: %d  Reasoning: %d  Completion: %d  Output: %d  Window: %d (%.0f%% of %.0fK) TPS: %.2f\u001b[0m\n",
		lr.Usage.InputTokens, lr.Usage.ReasoningTokens, lr.Usage.CompletionTokens, lr.Usage.OutputTokens, contextTokens, percentage, of, lr.Usage.TokensPerSecond)

	return messages, nil
}
