package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/docling"
)

var (
	urlModel = "http://localhost:11434/v1/chat/completions"
	model    = "gpt-oss:latest"

	urlDocling = "http://localhost:5001/v1/convert/file"

	documentPath = "zarf/samples/docs/dinner_menu.pdf"

	// The context window represents the maximum number of tokens that can be sent
	// and received by the model. The default for Ollama is 4K. In the makefile
	// it has been increased to 64K.
	contextWindow = 1024 * 4
)

func init() {
	if v := os.Getenv("OLLAMA_CONTEXT_LENGTH"); v != "" {
		var err error
		contextWindow, err = strconv.Atoi(v)
		if err != nil {
			log.Fatal(err)
		}
	}

	if v := os.Getenv("LLM_SERVER"); v != "" {
		urlModel = v
	}

	if v := os.Getenv("LLM_MODEL"); v != "" {
		model = v
	}

	if v := os.Getenv("DOC_SERVER"); v != "" {
		urlDocling = v
	}
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	// -------------------------------------------------------------------------

	fmt.Println("\nExtract content from document")

	doc := docling.New(urlDocling)

	fields := map[string]string{
		"to_formats":                "md",
		"include_images":            "false",
		"table_mode":                "accurate",
		"md_page_break_placeholder": "---",
		"pdf_backend":               "dlparse_v4",
		"image_export_mode":         "placeholder",
	}

	data, err := doc.ConvertFile(ctx, documentPath, fields)
	if err != nil {
		return fmt.Errorf("docling: %w", err)
	}

	fmt.Println("\nExtracted content")
	fmt.Printf("\u001b[92m%s\u001b[0m", data)

	// -------------------------------------------------------------------------

	fmt.Println("\nProcess against the LLM")

	csvData, err := ollama(ctx, data)
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Print("\n\nParsed CSV:\n\n")

	reader := csv.NewReader(strings.NewReader(csvData))
	records, err := reader.ReadAll()
	if err != nil {
		return fmt.Errorf("parse csv: %w", err)
	}

	for _, record := range records {
		fmt.Printf("\u001b[93m%s\u001b[0m", record)
	}

	return nil
}

func ollama(ctx context.Context, data string) (string, error) {
	const prompt = `
		This data represents a menu. Structure this data to align the categories,
		items, descriptions, and prices together in a CSV format. First categorize
		the items, then make sure each item is matched to a category and
		description. Only output the CSV data and nothing else.
		
		Use this as an example:

		"CATEGORY","ITEM","DESC",PRICE
	`

	conversation := []client.D{
		{
			"role":    "user",
			"content": prompt,
		},
		{
			"role":    "user",
			"content": data,
		},
	}

	d := client.D{
		"model":       model,
		"messages":    conversation,
		"max_tokens":  contextWindow,
		"temperature": 0.0,
		"top_p":       0.1,
		"top_k":       1,
		"stream":      true,
	}

	ch := make(chan client.ChatSSE, 100)

	sseClient := client.NewSSE[client.ChatSSE](client.StdoutLogger)
	if err := sseClient.Do(ctx, http.MethodPost, urlModel, d, ch); err != nil {
		return "", fmt.Errorf("do request: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Print("\nReasoning:\n")
	reasoning := true

	var csvData strings.Builder

	for resp := range ch {
		if len(resp.Choices) == 0 {
			continue
		}

		switch {
		case resp.Choices[0].Delta.Content != "":
			if reasoning {
				fmt.Print("\n\nOutput:\n")
				reasoning = false
			}
			fmt.Print(resp.Choices[0].Delta.Content)
			csvData.WriteString(resp.Choices[0].Delta.Content)

		case resp.Choices[0].Delta.Reasoning != "":
			fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
		}
	}

	return csvData.String(), nil
}
