// Package client provides support to access an OpenAI-compatible API service.
package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"time"
)

const version = "v1.0.0"

var ErrUnauthorized = errors.New("api understands the request but refuses to authorize it")

var defaultClient = http.Client{
	Transport: &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 15 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	},
}

// =============================================================================

type Logger func(context.Context, string, ...any)

var NoopLogger = func(ctx context.Context, msg string, v ...any) {}

var StdoutLogger = func(ctx context.Context, msg string, v ...any) {
	var b strings.Builder
	fmt.Fprintf(&b, "msg: %s", msg)

	for i := 0; i < len(v); i += 2 {
		if i+1 == len(v) {
			fmt.Fprintf(&b, ", %v", v[i])
			break
		}
		fmt.Fprintf(&b, ", %s: %v", v[i], v[i+1])
	}

	log.Println(b.String())
}

// =============================================================================

type Client struct {
	log  Logger
	http *http.Client
}

func New(log Logger, options ...func(cln *Client)) *Client {
	cln := Client{
		log:  log,
		http: &defaultClient,
	}

	for _, option := range options {
		option(&cln)
	}

	return &cln
}

func WithClient(http *http.Client) func(cln *Client) {
	return func(cln *Client) {
		cln.http = http
	}
}

func (cln *Client) Do(ctx context.Context, method string, endpoint string, body D, v any) error {
	resp, err := do(ctx, cln, method, endpoint, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		return nil
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("client: copy error: %w", err)
	}

	switch d := v.(type) {
	case *string:
		*d = string(data)

	default:
		if err := json.Unmarshal(data, v); err != nil {
			return fmt.Errorf("client: response: %s, decoding error: %w ", string(data), err)
		}
	}

	return nil
}

// =============================================================================

type SSEClient[T any] struct {
	*Client
}

func NewSSE[T any](log Logger, options ...func(cln *Client)) *SSEClient[T] {
	cln := New(log, options...)

	return &SSEClient[T]{
		Client: cln,
	}
}

// streamErrorSetter is implemented by SSE event types that can carry a
// failure of the stream itself. When the stream breaks, the SSE client sends
// one such value so the caller sees the error instead of a stream that simply
// ended early, which is otherwise indistinguishable from a normal finish.
type streamErrorSetter interface {
	setStreamError(err error)
}

// maxSSELineSize bounds a single SSE line. The bufio.Scanner default is 64KB,
// and a large tool-call argument or a long chunk can exceed that.
const maxSSELineSize = 16 * 1024 * 1024

func (cln *SSEClient[T]) Do(ctx context.Context, method string, endpoint string, body D, ch chan T) error {
	resp, err := do(ctx, cln.Client, method, endpoint, body)
	if err != nil {
		return err
	}

	go func(ctx context.Context) {
		defer func() {
			resp.Body.Close()
			close(ch)
		}()

		send := func(v T) bool {
			select {
			case ch <- v:
				return true

			case <-ctx.Done():
				cln.log(ctx, "sseclient: rawRequest:", "Context", ctx.Err().Error())
				return false
			}
		}

		// sendErr reports a broken stream to the caller when T can carry it.
		sendErr := func(err error) {
			var v T
			if s, ok := any(&v).(streamErrorSetter); ok {
				s.setStreamError(err)
				send(v)
			}
		}

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), maxSSELineSize)

		for scanner.Scan() {
			line := scanner.Text()

			// Only "data:" lines carry events. Blank lines separate events and
			// lines starting with ":" are comments, such as keep-alives. The
			// space after the colon is optional in the SSE format.
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			data = strings.TrimPrefix(data, " ")

			if data == "[DONE]" {
				continue
			}

			var v T
			if err := json.Unmarshal([]byte(data), &v); err != nil {
				cln.log(ctx, "sseclient: rawRequest:", "Unmarshal", err, "line", data)
				sendErr(fmt.Errorf("sseclient: decoding event: %w", err))
				return
			}

			if !send(v) {
				return
			}
		}

		if err := scanner.Err(); err != nil && ctx.Err() == nil {
			cln.log(ctx, "sseclient: rawRequest:", "Scanner", err)
			sendErr(fmt.Errorf("sseclient: reading stream: %w", err))
		}
	}(ctx)

	return nil
}

// =============================================================================

func do(ctx context.Context, cln *Client, method string, endpoint string, body any) (*http.Response, error) {
	var statusCode int

	var b bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&b).Encode(body); err != nil {
			return nil, fmt.Errorf("encoding: error: %w", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, method, endpoint, &b)
	if err != nil {
		return nil, fmt.Errorf("create request error: %w", err)
	}

	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", fmt.Sprintf("Ardan Labs AI Training Sample Go Client: %s", version))

	resp, err := cln.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do: error: %w", err)
	}

	// Assign for logging the status code at the end of the function call.
	statusCode = resp.StatusCode

	switch statusCode {
	case http.StatusOK, http.StatusNoContent:
		return resp, nil

	default:
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("readall: error: %w", err)
		}

		switch statusCode {
		case http.StatusForbidden:
			return nil, ErrUnauthorized

		default:
			// Prefer the structured error message, but a body that is not the
			// expected JSON (a proxy page, plain text) still has to report the
			// status code rather than a decoding failure.
			var apiErr Error
			if err := json.Unmarshal(data, &apiErr); err != nil || apiErr.Err.Message == "" {
				return nil, fmt.Errorf("error: status: %d response: %s", statusCode, bytes.TrimSpace(data))
			}

			return nil, fmt.Errorf("error: status: %d response: %s", statusCode, apiErr.Err.Message)
		}
	}
}
