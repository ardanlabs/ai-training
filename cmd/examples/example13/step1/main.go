package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
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

	data, err := docling(ctx, documentPath)
	if err != nil {
		return fmt.Errorf("docling: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Println("Process against the LLM")

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
		fmt.Println(record)
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

func docling(ctx context.Context, file string) (string, error) {
	cln := http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 15 * time.Second,
				DualStack: true,
			}).DialContext,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          100,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		},
	}

	var b bytes.Buffer
	writer := multipart.NewWriter(&b)

	fileWriter, err := writer.CreateFormFile("files", file)
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}

	f, err := os.Open(file)
	if err != nil {
		return "", fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	if _, err := io.Copy(fileWriter, f); err != nil {
		return "", fmt.Errorf("copy file: %w", err)
	}

	writer.WriteField("to_formats", "md")
	writer.WriteField("include_images", "false")
	writer.WriteField("table_mode", "accurate")
	writer.WriteField("md_page_break_placeholder", "---")
	writer.WriteField("pdf_backend", "dlparse_v4")
	writer.WriteField("image_export_mode", "placeholder")

	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("close writer: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, urlDocling, &b)
	if err != nil {
		return "", fmt.Errorf("create request error: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := cln.Do(req)
	if err != nil {
		return "", fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status: %d, body: %s", resp.StatusCode, body)
	}

	// -------------------------------------------------------------------------

	var data struct {
		Document struct {
			MDContent string `json:"md_content"`
		} `json:"document"`
	}

	if err := json.Unmarshal(body, &data); err != nil {
		return "", fmt.Errorf("unmarshal: %w", err)
	}

	return data.Document.MDContent, nil
}
