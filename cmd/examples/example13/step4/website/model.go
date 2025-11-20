package website

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/ardanlabs/kronk"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Request struct {
	Messages    []Message `json:"messages"`
	TopK        *int32    `json:"top_k"`
	TopP        *float32  `json:"top_p"`
	Temperature *float32  `json:"temperature"`
}

func getParams(traceID string, req Request) kronk.Params {
	topK := int32(50)
	if req.TopK != nil {
		fmt.Printf("traceID: %s: getParams: topK: %#v\n", traceID, *req.TopK)
		topK = *req.TopK
	}

	topP := float32(0.95)
	if req.TopP != nil {
		fmt.Printf("traceID: %s: getParams: topP: %#v\n", traceID, *req.TopP)
		topP = *req.TopP
	}

	temp := float32(1.0)
	if req.Temperature != nil {
		fmt.Printf("traceID: %s: getParams: temp: %#v\n", traceID, *req.Temperature)
		temp = *req.Temperature
	}

	params := kronk.Params{
		TopK: topK,
		TopP: topP,
		Temp: temp,
	}

	return params
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
