package client

import (
	"encoding/json"
	"errors"
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

	// RawArguments holds the arguments text exactly as received. When a server
	// streams the arguments in fragments (OpenAI style), each fragment is not
	// valid JSON on its own, so Arguments stays empty and the caller can join
	// RawArguments across chunks and decode the result once complete.
	RawArguments string
}

func (f *Function) UnmarshalJSON(b []byte) error {
	var tmp struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}

	if err := json.Unmarshal(b, &tmp); err != nil {
		return err
	}

	// A streaming tool call arrives in pieces. The opening chunk carries the id
	// and the function name with "arguments" set to the empty string, and the
	// arguments themselves follow in later chunks keyed only by index. Treating
	// an empty or partial argument list as a decode failure rejects that chunk,
	// and because the SSE reader abandons the response on a malformed line, the
	// whole completion is lost - the model asks for a tool and the caller sees
	// an empty answer. So nothing about the arguments is an error here.
	//
	// The OpenAI format sends the arguments as a JSON-encoded string, but some
	// servers send the object itself, so both are accepted.
	var raw string
	switch {
	case len(tmp.Arguments) == 0 || string(tmp.Arguments) == "null":

	case tmp.Arguments[0] == '"':
		if err := json.Unmarshal(tmp.Arguments, &raw); err != nil {
			return err
		}

	default:
		raw = string(tmp.Arguments)
	}

	arguments := make(map[string]any)
	if raw != "" {
		if err := json.Unmarshal([]byte(raw), &arguments); err != nil {
			arguments = make(map[string]any)
		}
	}

	*f = Function{
		Name:         tmp.Name,
		Arguments:    arguments,
		RawArguments: raw,
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

// UnmarshalJSON reads usage in the OpenAI format, where the reasoning count
// is nested under completion_tokens_details and there is no output count.
// Completion tokens cover both the reasoning and the visible answer, so the
// output count is what remains after the reasoning. Flat reasoning_tokens and
// output_tokens fields are still honored for servers that send them.
func (u *Usage) UnmarshalJSON(b []byte) error {
	type usage Usage

	var tmp struct {
		usage
		CompletionTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	}

	if err := json.Unmarshal(b, &tmp); err != nil {
		return err
	}

	*u = Usage(tmp.usage)

	if u.ReasoningTokens == 0 {
		u.ReasoningTokens = tmp.CompletionTokensDetails.ReasoningTokens
	}

	if u.OutputTokens == 0 {
		u.OutputTokens = max(u.CompletionTokens-u.ReasoningTokens, 0)
	}

	return nil
}

type ChatSSE struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created Time            `json:"created"`
	Model   string          `json:"model"`
	Choices []ChatChoiceSSE `json:"choices"`
	Usage   *Usage          `json:"usage,omitempty"`
}

// UnmarshalJSON turns an error event into a choice with the "error" finish
// reason and the message as its content. A server that fails mid-stream sends
// {"error":{"message":...}} with no choices, which callers would otherwise
// skip as an empty chunk and then treat the truncated answer as complete.
func (c *ChatSSE) UnmarshalJSON(b []byte) error {
	type chatSSE ChatSSE

	var tmp struct {
		chatSSE
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(b, &tmp); err != nil {
		return err
	}

	*c = ChatSSE(tmp.chatSSE)

	if tmp.Error != nil && len(c.Choices) == 0 {
		c.setStreamError(errors.New(tmp.Error.Message))
	}

	return nil
}

// setStreamError implements the SSE client's streamErrorSetter so a broken
// stream reaches the caller through the same "error" finish reason.
func (c *ChatSSE) setStreamError(err error) {
	c.Choices = []ChatChoiceSSE{
		{
			Delta:        ChatDeltaSSE{Content: err.Error()},
			FinishReason: "error",
		},
	}
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
