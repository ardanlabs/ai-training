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
	"net"
	"net/http"
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
			DualStack: true,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	},
}

type Logger func(context.Context, string, ...any)

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

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			if line == "" || line == "data: [DONE]" {
				continue
			}

			var v T
			if err := json.Unmarshal([]byte(line[6:]), &v); err != nil {
				cln.log(ctx, "sseclient: rawRequest:", "Unmarshal", err, "line", line[6:])
				return
			}

			select {
			case ch <- v:

			case <-ctx.Done():
				cln.log(ctx, "sseclient: rawRequest:", "Context", ctx.Err().Error())
				return
			}
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
			var err Error
			if err := json.Unmarshal(data, &err); err != nil {
				return nil, fmt.Errorf("decoding: response: %s, error: %w ", string(data), err)
			}

			return nil, fmt.Errorf("error: response: %s", err.Message)
		}
	}
}
