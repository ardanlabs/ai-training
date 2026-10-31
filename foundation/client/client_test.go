package client

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestUsageUnmarshal(t *testing.T) {
	tests := []struct {
		name      string
		data      string
		reasoning int
		output    int
	}{
		{
			name:      "openai nested details",
			data:      `{"prompt_tokens":10,"completion_tokens":30,"total_tokens":40,"completion_tokens_details":{"reasoning_tokens":12}}`,
			reasoning: 12,
			output:    18,
		},
		{
			name:      "flat fields win",
			data:      `{"completion_tokens":30,"reasoning_tokens":5,"output_tokens":25}`,
			reasoning: 5,
			output:    25,
		},
		{
			name:   "no reasoning",
			data:   `{"completion_tokens":7}`,
			output: 7,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var u Usage
			if err := json.Unmarshal([]byte(tt.data), &u); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}

			if u.ReasoningTokens != tt.reasoning || u.OutputTokens != tt.output {
				t.Fatalf("got reasoning=%d output=%d, want reasoning=%d output=%d", u.ReasoningTokens, u.OutputTokens, tt.reasoning, tt.output)
			}
		})
	}
}

func TestChatSSEErrorEvent(t *testing.T) {
	var c ChatSSE
	if err := json.Unmarshal([]byte(`{"error":{"message":"model crashed"}}`), &c); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(c.Choices) != 1 || c.Choices[0].FinishReason != "error" || c.Choices[0].Delta.Content != "model crashed" {
		t.Fatalf("error event not surfaced as an error choice: %+v", c.Choices)
	}

	var normal ChatSSE
	if err := json.Unmarshal([]byte(`{"id":"1","created":1,"choices":[{"delta":{"content":"hi"}}],"usage":{"completion_tokens":2}}`), &normal); err != nil {
		t.Fatalf("unmarshal normal: %v", err)
	}

	if normal.Choices[0].Delta.Content != "hi" || normal.Usage == nil || normal.Usage.OutputTokens != 2 {
		t.Fatalf("normal chunk decoded wrong: %+v usage=%+v", normal, normal.Usage)
	}
}

func TestFunctionUnmarshal(t *testing.T) {
	tests := []struct {
		name string
		data string
		args map[string]any
		raw  string
	}{
		{name: "string", data: `{"name":"f","arguments":"{\"a\":1}"}`, args: map[string]any{"a": 1.0}, raw: `{"a":1}`},
		{name: "object", data: `{"name":"f","arguments":{"a":1}}`, args: map[string]any{"a": 1.0}, raw: `{"a":1}`},
		{name: "empty", data: `{"name":"f","arguments":""}`, args: map[string]any{}},
		{name: "missing", data: `{"name":"f"}`, args: map[string]any{}},
		{name: "fragment", data: `{"name":"","arguments":"{\"a\":"}`, args: map[string]any{}, raw: `{"a":`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var f Function
			if err := json.Unmarshal([]byte(tt.data), &f); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}

			if fmt.Sprint(f.Arguments) != fmt.Sprint(tt.args) || f.RawArguments != tt.raw {
				t.Fatalf("got args=%v raw=%q, want args=%v raw=%q", f.Arguments, f.RawArguments, tt.args, tt.raw)
			}
		})
	}
}

func TestSSEClient(t *testing.T) {
	long := strings.Repeat("x", 100*1024)

	tests := []struct {
		name string
		body string
		want []string
	}{
		{
			name: "no space after data, comments and done",
			body: ": keep-alive\n\ndata:{\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\ndata: [DONE]\n\n",
			want: []string{"a", "b"},
		},
		{
			name: "line longer than 64KB",
			body: "data: {\"choices\":[{\"delta\":{\"content\":\"" + long + "\"}}]}\n\n",
			want: []string{long},
		},
		{
			name: "server error event",
			body: "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"error\":{\"message\":\"boom\"}}\n\ndata: [DONE]\n\n",
			want: []string{"a", "error:boom"},
		},
		{
			name: "malformed event",
			body: "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {not json\n\n",
			want: []string{"a", "error:sseclient: decoding event"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, tt.body)
			}))
			defer srv.Close()

			ch := make(chan ChatSSE, 10)
			if err := NewSSE[ChatSSE](NoopLogger).Do(context.Background(), http.MethodPost, srv.URL, D{}, ch); err != nil {
				t.Fatalf("do: %v", err)
			}

			var got []string
			for resp := range ch {
				c := resp.Choices[0]
				if c.FinishReason == "error" {
					got = append(got, "error:"+c.Delta.Content)
					continue
				}
				got = append(got, c.Delta.Content)
			}

			if len(got) != len(tt.want) {
				t.Fatalf("got %d events, want %d", len(got), len(tt.want))
			}

			for i := range got {
				if !strings.HasPrefix(got[i], tt.want[i]) {
					t.Fatalf("event %d: got %.60q, want prefix %.60q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestDoNon200(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "json error", body: `{"error":{"message":"bad model"}}`, want: "error: status: 400 response: bad model"},
		{name: "plain text", body: "upstream down\n", want: "error: status: 400 response: upstream down"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				fmt.Fprint(w, tt.body)
			}))
			defer srv.Close()

			var v D
			err := New(NoopLogger).Do(context.Background(), http.MethodPost, srv.URL, D{}, &v)
			if err == nil || err.Error() != tt.want {
				t.Fatalf("got %v, want %q", err, tt.want)
			}
		})
	}
}
