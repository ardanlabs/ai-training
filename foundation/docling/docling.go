// Package docling provides a client for the Docling API.
package docling

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"time"
)

var defaultClient = http.Client{
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

type Docling struct {
	client *http.Client
	host   string
}

func New(host string, options ...func(doc *Docling)) *Docling {
	doc := &Docling{
		client: &defaultClient,
		host:   host,
	}

	for _, option := range options {
		option(doc)
	}

	return doc
}

func WithClient(http *http.Client) func(doc *Docling) {
	return func(doc *Docling) {
		doc.client = http
	}
}

func (doc *Docling) ConvertFile(ctx context.Context, fileName string, fields map[string]string) (string, error) {
	var b bytes.Buffer
	writer := multipart.NewWriter(&b)

	fileWriter, err := writer.CreateFormFile("files", fileName)
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}

	f, err := os.Open(fileName)
	if err != nil {
		return "", fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	if _, err := io.Copy(fileWriter, f); err != nil {
		return "", fmt.Errorf("copy file: %w", err)
	}

	for k, v := range fields {
		writer.WriteField(k, v)
	}

	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("close writer: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, doc.host, &b)
	if err != nil {
		return "", fmt.Errorf("create request error: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := doc.client.Do(req)
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
