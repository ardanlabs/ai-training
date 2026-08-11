package client

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"
)

type D map[string]any

// =============================================================================

type Error struct {
	Err struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (err *Error) Error() string {
	return err.Err.Message
}

// =============================================================================

type Time struct {
	time.Time
}

func ToTime(sec int64) Time {
	return Time{
		Time: time.Unix(sec, 0),
	}
}

func (t *Time) UnmarshalJSON(data []byte) error {
	d := strings.Trim(string(data), "\"")

	num, err := strconv.Atoi(d)
	if err != nil {
		return err
	}

	t.Time = time.Unix(int64(num), 0)

	return nil
}

func (t Time) MarshalJSON() ([]byte, error) {
	data := strconv.Itoa(int(t.Unix()))
	return []byte(data), nil
}

// =============================================================================

type Function struct {
	Name      string
	Arguments map[string]any
}

func (f *Function) UnmarshalJSON(b []byte) error {
	var tmp struct {
		Name         string `json:"name"`
		RawArguments string `json:"arguments"`
	}

	if err := json.Unmarshal(b, &tmp); err != nil {
		return err
	}

	// A streaming tool call arrives in pieces. The opening chunk carries the id
	// and the function name with "arguments" set to the empty string, and the
	// arguments themselves follow in later chunks keyed only by index. Treating
	// an empty string as a decode failure rejects that opening chunk, and
	// because the SSE reader abandons the response on a malformed line, the
	// whole completion is lost - the model asks for a tool and the caller sees
	// an empty answer. An absent argument list is a valid state, not an error.
	arguments := make(map[string]any)
	if tmp.RawArguments != "" {
		if err := json.Unmarshal([]byte(tmp.RawArguments), &arguments); err != nil {
			return err
		}
	}

	*f = Function{
		Name:      tmp.Name,
		Arguments: arguments,
	}

	return nil
}

type ToolCall struct {
	ID       string   `json:"id,omitempty"`
	Index    int      `json:"index"`
	Type     string   `json:"type,omitempty"`
	Function Function `json:"function"`
}

type ChatDeltaSSE struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Reasoning string     `json:"reasoning_content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ChatChoiceSSE struct {
	Index        int          `json:"index"`
	Delta        ChatDeltaSSE `json:"delta"`
	FinishReason string       `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	ReasoningTokens  int     `json:"reasoning_tokens,omitempty"`
	OutputTokens     int     `json:"output_tokens,omitempty"`
	TokensPerSecond  float64 `json:"tokens_per_second,omitempty"`
}

type ChatSSE struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created Time            `json:"created"`
	Model   string          `json:"model"`
	Choices []ChatChoiceSSE `json:"choices"`
	Usage   *Usage          `json:"usage,omitempty"`
}

// =============================================================================

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatChoice struct {
	Index   int         `json:"index"`
	Message ChatMessage `json:"message"`
}

type Chat struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created Time         `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
}

// =============================================================================

type EmbeddingData struct {
	Index     int       `json:"index"`
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
}

type Embedding struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created Time            `json:"created"`
	Model   string          `json:"model"`
	Data    []EmbeddingData `json:"data"`
}
