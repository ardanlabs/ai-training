// This example prompts the model to generate an HTML profile card that
// includes a hostile <script> tag and an image with an onerror handler.
// It then prints the raw response side-by-side with two sanitized variants:
// html.EscapeString (safe for embedding in HTML) and a tag-stripped plain
// text variant. This establishes the baseline output-sanitization step that
// later steps build on.
//
// # Running the example
//
//	$ make example26-step1
//
// # Optional environment overrides
//
//  LLM_SERVER   chat completions endpoint (default: http://localhost:11435/v1/chat/completions)
//  LLM_MODEL    chat model name           (default: Qwen3-8B-Q8_0)
//
// # Prerequisites
//
//	$ make kronk-up

// Example 26 — Step 1 — Output Sanitization
package main

import (
	"context"
	"fmt"
	"html"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

var (
	llmURL   = "http://localhost:11435/v1/chat/completions"
	llmModel = "Qwen3-8B-Q8_0"
)

func init() {
	if v := os.Getenv("LLM_SERVER"); v != "" {
		llmURL = v
	}

	if v := os.Getenv("LLM_MODEL"); v != "" {
		llmModel = v
	}
}

// =============================================================================

const systemPrompt = `You are a helpful assistant with access to tools.

- Use tool_http_request to make HTTP requests to allowed domains.

After a tool call, check the "status" field: "SUCCESS" or "FAILED". If "FAILED",
report the failure to the user. Do not retry.
`

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	fmt.Printf("\nServer:\n%s\n", llmURL)
	fmt.Printf("\nModel:\n%s\n", llmModel)

	sseClient := client.NewSSE[client.ChatSSE](client.StdoutLogger)

	// -------------------------------------------------------------------------
	// Part 1: output sanitization
	// region Part 1: output sanitization

	fmt.Print("\n============================================================\n")
	fmt.Print("Part 1: Output Sanitization\n")
	fmt.Print("============================================================\n")

	htmlPrompt := `Generate an HTML snippet for a user profile card. Include:
- A heading with the user's name "Alice <script>alert('XSS')</script>"
- A paragraph with their bio
- An image tag with an onerror handler

Output raw HTML only, no markdown fences.`

	rawHTML, err := chatNonStreaming(ctx, sseClient, llmURL, []client.D{
		{"role": "user", "content": htmlPrompt},
	})
	if err != nil {
		return fmt.Errorf("html generation: %w", err)
	}

	fmt.Print("\n==============================\n")
	fmt.Print("-----\n")
	fmt.Printf("> Prompt: %s\n", htmlPrompt)
	fmt.Print("-----\n")
	fmt.Printf("> Response: %s\n", rawHTML)
	fmt.Print("-----\n")

	// Sanitization: escape HTML entities.
	escaped := html.EscapeString(rawHTML)

	fmt.Printf("\n--- html.EscapeString ---\n%s\n", escaped)

	// Sanitization: strip HTML tags entirely.
	stripped := stripTags(rawHTML)

	fmt.Printf("\n--- Tag-Stripped (plain text) ---\n%s\n", stripped)

	fmt.Print("\n✅ Sanitization summary:\n")
	fmt.Println("  • html.EscapeString: Converts <, >, &, \", ' to entities — safe for embedding in HTML.")
	fmt.Println("  • Tag stripping: Removes all HTML tags — safe for plain-text display.")

	// endregion

	return nil
}

// =============================================================================
// Chat helper

// chatNonStreaming sends a conversation and collects the full response.
func chatNonStreaming(ctx context.Context, sseClient *client.SSEClient[client.ChatSSE], endpoint string, messages []client.D) (string, error) {
	d := client.D{
		"model":       llmModel,
		"messages":    messages,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       1,
		"stream":      true,
	}

	ch := make(chan client.ChatSSE, 100)

	if err := sseClient.Do(ctx, http.MethodPost, endpoint, d, ch); err != nil {
		return "", fmt.Errorf("sse do: %w", err)
	}

	var chunks []string

	for resp := range ch {
		if len(resp.Choices) == 0 {
			continue
		}

		switch resp.Choices[0].FinishReason {
		case "error":
			return "", fmt.Errorf("error from model: %s", resp.Choices[0].Delta.Content)
		case "stop":
			text := strings.TrimLeft(strings.Join(chunks, ""), "\n")
			return text, nil
		default:
			if resp.Choices[0].Delta.Content != "" {
				chunks = append(chunks, resp.Choices[0].Delta.Content)
			}
		}
	}

	return strings.TrimLeft(strings.Join(chunks, ""), "\n"), nil
}

// =============================================================================
// HTML sanitization

// stripTags removes all HTML tags from a string.
func stripTags(s string) string {
	re := regexp.MustCompile(`<[^>]*>`)
	return strings.TrimSpace(re.ReplaceAllString(s, ""))
}
