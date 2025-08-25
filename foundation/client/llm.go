package client

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
)

type LLM struct {
	cln    *Client
	clnSSE *SSEClient[ChatSSE]
	url    string
	model  string
}

func NewLLM(url string, model string) *LLM {
	return &LLM{
		cln:    New(StdoutLogger),
		clnSSE: NewSSE[ChatSSE](StdoutLogger),
		url:    url,
		model:  model,
	}
}

func WithImage(mimeType string, image []byte) D {
	dataBase64 := base64.StdEncoding.EncodeToString(image)

	return D{
		"type": "image_url",
		"image_url": D{
			"url": fmt.Sprintf("data:%s;base64,%s", mimeType, dataBase64),
		},
	}
}

func (llm *LLM) ChatCompletions(ctx context.Context, text string, options ...D) (string, error) {
	content := []D{
		{
			"type": "text",
			"text": text,
		},
	}

	content = append(content, options...)

	d := D{
		"model": llm.model,
		"messages": []D{
			{
				"role":    "user",
				"content": content,
			},
		},
		"temperature": 1.0,
		"top_p":       0.5,
		"top_k":       20,
	}

	var chat Chat
	if err := llm.cln.Do(ctx, http.MethodPost, llm.url, d, &chat); err != nil {
		return "", fmt.Errorf("do: %w", err)
	}

	if len(chat.Choices) == 0 {
		return "", fmt.Errorf("no response")
	}

	return chat.Choices[0].Message.Content, nil
}

func (llm *LLM) ChatCompletionsSSE(ctx context.Context, content string) (chan ChatSSE, error) {
	d := D{
		"model": llm.model,
		"messages": []D{
			{
				"role":    "user",
				"content": content,
			},
		},
		"temperature": 1.0,
		"top_p":       0.5,
		"top_k":       20,
		"stream":      true,
	}

	ch := make(chan ChatSSE, 100)
	if err := llm.clnSSE.Do(ctx, http.MethodPost, llm.url, d, ch); err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}

	return ch, nil
}

func (llm *LLM) EmbedText(ctx context.Context, input string) ([]float64, error) {
	d := D{
		"model":              llm.model,
		"truncate":           true,
		"truncate_direction": "right",
		"input":              input,
	}

	var resp Embedding
	if err := llm.cln.Do(ctx, http.MethodPost, llm.url, d, &resp); err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding")
	}

	return resp.Data[0].Embedding, nil
}

func (llm *LLM) EmbedWithImage(ctx context.Context, description string, image []byte, mimeType string) ([]float64, error) {
	dataBase64 := base64.StdEncoding.EncodeToString(image)

	d := D{
		"model": llm.model,
		"input": []D{
			{
				"type": "image_url",
				"image_url": D{
					"url": fmt.Sprintf("data:%s;base64,%s", mimeType, dataBase64),
				},
			},
		},
	}

	var resp Embedding
	if err := llm.cln.Do(ctx, http.MethodPost, llm.url, d, &resp); err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding")
	}

	return resp.Data[0].Embedding, nil
}
