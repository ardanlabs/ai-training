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
	"github.com/ardanlabs/kronk"
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
		sendError(w, traceID, "NewDecoder", err)
		return
	}

	fmt.Printf("traceID: %s: chat: msgs: %#v\n", traceID, req.Messages)

	documents, err := h.findContext(traceID, req)
	if err != nil {
		sendError(w, traceID, "findContext", err)
		return
	}

	msgs := h.compileChatMessages(traceID, req, documents)
	params := getParams(traceID, req)

	ctx, cancel := context.WithTimeout(context.Background(), h.timeout)
	defer cancel()

	ch, err := h.krnChat.ChatStreaming(ctx, msgs, params)
	if err != nil {
		sendError(w, traceID, "performChat", err)
		return
	}

	if err := h.streamResponse(r.Context(), traceID, w, ch); err != nil {
		sendError(w, traceID, "streamResponse", err)
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

func (h *handlers) findContext(traceID string, req Request) ([]duck.Document, error) {
	yesSearch, err := h.needVectorSearch(traceID, req)
	if err != nil {
		return nil, fmt.Errorf("needVectorSearch: %w", err)
	}

	if yesSearch {
		docs, err := h.vectorSearch(traceID, req)
		if err != nil {
			return nil, fmt.Errorf("vectorSearch: %w", err)
		}
		return docs, nil
	}

	return nil, nil
}

func (h *handlers) needVectorSearch(traceID string, req Request) (bool, error) {
	const prompt = `
		Is the following question related to the Go programming language?
		Please response with a YES or NO answer.

		Question:
		%s
	`

	msgs := []kronk.ChatMessage{
		{
			Role:    "user",
			Content: fmt.Sprintf(prompt, req.Messages[len(req.Messages)-1].Content),
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), h.timeout)
	defer cancel()

	response, err := h.krnChat.Chat(ctx, msgs, kronk.Params{})
	if err != nil {
		return false, err
	}

	resp := strings.ToLower(response.Choice[0].Delta.Content)

	if strings.Contains(resp, "yes") {
		fmt.Printf("traceID: %s: needVectorSearch: response: YES: %s\n", traceID, resp)
		return true, nil
	}

	fmt.Printf("traceID: %s: needVectorSearch: response: NO: %s\n", traceID, resp)
	return false, nil
}

func (h *handlers) vectorSearch(traceID string, req Request) ([]duck.Document, error) {
	fmt.Printf("traceID: %s: vectorSearch: started", traceID)

	question := req.Messages[len(req.Messages)-1].Content

	fmt.Printf("traceID: %s: vectorSearch: started: question: %s\n", traceID, question)

	ctx, cancel := context.WithTimeout(context.Background(), h.timeout)
	defer cancel()

	queryVector, err := h.krnEmbed.Embed(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	docs, err := duck.Search(h.db, queryVector, 5)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	return docs, nil
}

func (h *handlers) compileChatMessages(traceID string, req Request, documents []duck.Document) []kronk.ChatMessage {
	fmt.Printf("traceID: %s: compileChatMessages: started: msgs: %d: documents: %d\n", traceID, len(req.Messages), len(documents))

	const systemPrompt = `
		- Use any provided Context to answer the user's question.
		- If you don't know the answer, say that you don't know.
		- Responses should be properly formatted to be easily read.
		- Share code if code is presented in the context.
		- If relavant Context is available, use it to answer the question and don't include any additional information not present in the Context.
	`

	// Add 2 more elements for the system prompt and any context.
	msgs := make([]kronk.ChatMessage, 0, len(req.Messages)+2)

	// Add the system prompt.
	msgs = append(msgs, kronk.ChatMessage{Role: "system", Content: systemPrompt})

	// Add all but the very last message in the history.
	for _, msg := range req.Messages[:len(req.Messages)-1] {
		msgs = append(msgs, kronk.ChatMessage{Role: "user", Content: msg.Content})
	}

	// Add the top 2 extra context if it exists.
	var count int
	var content string
	for _, doc := range documents {
		content = fmt.Sprintf("%s\n%s\n", content, doc.Text)
		count++
		if count >= 2 {
			break
		}
	}

	if count > 0 {
		msgs = append(msgs, kronk.ChatMessage{Role: "user", Content: fmt.Sprintf("Context:\n%s\n\n", content)})
	}

	// Add the final message from the history. We expect this to be a question.
	question := req.Messages[len(req.Messages)-1].Content
	msgs = append(msgs, kronk.ChatMessage{Role: "user", Content: fmt.Sprintf("Question:\n%s\n\n", question)})

	fmt.Printf("traceID: %s: compileChatMessages: ended: msgs: %d\n", traceID, len(msgs))

	return msgs
}

func (h *handlers) streamResponse(ctx context.Context, traceID string, w http.ResponseWriter, ch <-chan kronk.ChatResponse) error {
	fmt.Printf("traceID: %s: streamResponse: started\n", traceID)

	f, ok := w.(http.Flusher)
	if !ok {
		return fmt.Errorf("streaming not supported")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	f.Flush()

	var lr kronk.ChatResponse

	for resp := range ch {
		if err := ctx.Err(); err != nil {
			if errors.Is(err, context.Canceled) {
				return errors.New("client disconnected, do not send response")
			}
		}

		d, err := json.Marshal(resp)
		if err != nil {
			return fmt.Errorf("json.Marshal: %w", err)
		}

		if resp.Choice[0].FinishReason == kronk.FinishReasonError {
			fmt.Printf("traceID: %s: chat: ERROR: %s\n", traceID, resp.Choice[0].Delta.Content)
			fmt.Fprintf(w, "data: %s\n", d)
			f.Flush()
			break
		}

		fmt.Fprintf(w, "data: %s\n", d)
		f.Flush()

		lr = resp
	}

	w.Write([]byte("data: [DONE]\n"))
	f.Flush()

	// -------------------------------------------------------------------------

	contextTokens := lr.Usage.InputTokens + lr.Usage.CompletionTokens
	contextWindow := h.krnChat.ModelConfig().ContextWindow
	percentage := (float64(contextTokens) / float64(contextWindow)) * 100
	of := float32(contextWindow) / float32(1024)

	fmt.Printf("traceID: %s: chat: Input: %d  Output: %d  Context: %d (%.0f%% of %.0fK) TPS: %.2f\n",
		traceID, lr.Usage.InputTokens, lr.Usage.OutputTokens, contextTokens, percentage, of, lr.Usage.TokensPerSecond)

	return nil
}
