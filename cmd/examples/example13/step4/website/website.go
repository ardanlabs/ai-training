// Package website provides api and web ui for the chatbot.
package website

import (
	"context"
	"database/sql"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/duck"
	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/google/uuid"
)

//go:embed static
var website embed.FS

const (
	websiteDir  = "static"
	websitePath = "/"
)

type handlers struct {
	krnEmbed *kronk.Kronk
	krnChat  *kronk.Kronk
	timeout  time.Duration
	db       *sql.DB
}

func (h *handlers) chat(w http.ResponseWriter, r *http.Request) {
	traceID := uuid.NewString()

	fmt.Printf("traceID: %s: chat: started\n", traceID)
	defer fmt.Printf("traceID: %s: chat: complete\n", traceID)

	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, traceID, "NewDecoder", err, http.StatusBadRequest)
		return
	}

	if len(req.Messages) == 0 {
		sendError(w, traceID, "validate", errors.New("no messages provided"), http.StatusBadRequest)
		return
	}

	fmt.Printf("traceID: %s: chat: stream[%v] msgs[%#v]\n", traceID, req.Stream, req.Messages)

	// Use the request context so the model stops generating when the client
	// goes away instead of running until the timeout.
	ctx, cancel := context.WithTimeout(r.Context(), h.timeout)
	defer cancel()

	ctx = kronk.SetFmtLoggerTraceID(ctx, traceID)

	question := req.Messages[len(req.Messages)-1].Content

	docs, err := h.vectorSearch(ctx, traceID, question)
	if err != nil {
		sendError(w, traceID, "vectorSearch", err, http.StatusInternalServerError)
		return
	}

	params := getParams(traceID, req)

	d := model.D{
		"messages": h.compileChatMessages(traceID, req, docs),
		"stream":   req.Stream,
		"tools": []model.D{
			{
				"type": "function",
				"function": model.D{
					"name":        "get_weather",
					"description": "Get the current weather for a location",
					"parameters": model.D{
						"type": "object",
						"properties": model.D{
							"location": model.D{
								"type":        "string",
								"description": "The location to get the weather for, e.g. San Francisco, CA",
							},
						},
						"required": []any{"location"},
					},
				},
			},
		},
	}

	// Without this, a streaming call reports no usage at all.
	if req.Stream {
		d["stream_options"] = model.D{
			"include_usage": true,
		}
	}

	model.AddParams(params, d)

	if _, err := h.krnChat.ChatStreamingHTTP(ctx, w, d); err != nil {

		// Once the response has started there is no way to send a different
		// status code, so the error can only be logged.
		if errors.Is(err, kronk.ErrResponseCommitted) {
			fmt.Printf("traceID: %s: chat: streamResponse: ERROR: %s\n", traceID, err)
			return
		}

		sendError(w, traceID, "streamResponse", err, http.StatusInternalServerError)
		return
	}
}

func (h *handlers) fileServerReact() func(w http.ResponseWriter, r *http.Request) {
	fileMatcher := regexp.MustCompile(`\.[a-zA-Z]*$`)

	fSys, err := fs.Sub(website, websiteDir)
	if err != nil {
		fmt.Printf("switching to static folder: %s", err)
		return nil
	}

	fileServer := http.StripPrefix(websitePath, http.FileServer(http.FS(fSys)))

	f := func(w http.ResponseWriter, r *http.Request) {
		if !fileMatcher.MatchString(r.URL.Path) {
			p, err := website.ReadFile(fmt.Sprintf("%s/index.html", websiteDir))
			if err != nil {
				fmt.Printf("FileServerReact: index.html not found: %v\n", err)
				http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write(p)
			return
		}

		fileServer.ServeHTTP(w, r)
	}

	return f
}

// =============================================================================

// vectorSearch finds the chunks of the Go notebook that are the closest match
// to the user's question.
func (h *handlers) vectorSearch(ctx context.Context, traceID string, question string) ([]duck.Document, error) {
	d := model.D{
		"input": question,
	}

	resp, err := h.krnEmbed.Embeddings(ctx, d)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	if len(resp.Data) == 0 || len(resp.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("empty query vector")
	}

	docs, err := duck.Search(h.db, resp.Data[0].Embedding, 2)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	for _, doc := range docs {
		fmt.Printf("traceID: %s: vectorSearch: doc[%d] similarity[%f]\n", traceID, doc.ID, doc.Similarity)
	}

	return docs, nil
}

func (h *handlers) compileChatMessages(traceID string, req Request, docs []duck.Document) []model.D {
	fmt.Printf("traceID: %s: compileChatMessages: started: msgs: %d\n", traceID, len(req.Messages))

	const systemPrompt = `
		- Use any provided Context to answer the user's question.
		- If you don't know the answer, say that you don't know.
		- Responses should be properly formatted to be easily read.
		- Share code if code is presented in the context.
		- If relevant Context is available, use it to answer the question and don't include any additional information not present in the Context.
	`

	// Add 2 more elements for the system prompt and any context.
	msgs := make([]model.D, 0, len(req.Messages)+2)

	// Add the system prompt.
	msgs = append(msgs, model.TextMessage("system", systemPrompt))

	// Add all but the very last message in the history.
	for _, msg := range req.Messages[:len(req.Messages)-1] {
		msgs = append(msgs, model.TextMessage(msg.Role, msg.Content))
	}

	// Add the final message from the history along with the Context found by
	// the vector search. We expect this to be a question.
	var content strings.Builder
	for _, doc := range docs {
		fmt.Fprintf(&content, "%s\n", doc.Text)
	}

	question := req.Messages[len(req.Messages)-1].Content
	msgs = append(msgs, model.TextMessage("user", fmt.Sprintf("Context:\n%s\n\nQuestion:\n%s\n\n", content.String(), question)))

	fmt.Printf("traceID: %s: compileChatMessages: ended: msgs: %d\n", traceID, len(msgs))

	return msgs
}
