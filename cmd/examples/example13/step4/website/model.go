package website

import (
	"encoding/json"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Request struct {
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature"`
	TopP        *float64  `json:"top_p"`
	TopK        *int      `json:"top_k"`
}

type Response struct {
	ID      string  `json:"id,omitempty"`
	Created int64   `json:"created,omitempty"`
	Model   string  `json:"model,omitempty"`
	Delta   Message `json:"delta,omitempty"`
	Final   string  `json:"final,omitempty"`
	Error   string  `json:"error,omitempty"`
}

func newResponse(id string, model string, content string, final string, err error) string {
	var errStr string
	if err != nil {
		errStr = err.Error()
	}

	resp := Response{
		ID:      id,
		Created: time.Now().UTC().UnixMilli(),
		Model:   model,
		Delta:   Message{Role: "assistant", Content: content},
		Final:   final,
		Error:   errStr,
	}

	d, _ := json.Marshal(resp)
	return string(d)
}
