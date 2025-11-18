// Package website provides api and web ui for the chatbot.
package website

import (
	"context"
	"database/sql"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/duck"
	"github.com/ardanlabs/llamacpp"
	"github.com/google/uuid"
)

//go:embed static
var website embed.FS

const (
	websiteDir  = "static"
	websitePath = "/"
)

type handlers struct {
	llmEmbed *llamacpp.Llama
	llmChat  *llamacpp.Llama
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

	fmt.Printf("traceID: %s: chat: req: %#v\n", traceID, req)

	docs, err := h.vectorSearch(traceID, req)
	if err != nil {
		sendError(w, traceID, "vectorSearch", err)
		return
	}

	rankings, err := h.rerank(traceID, docs)
	if err != nil {
		sendError(w, traceID, "rerank", err)
		return
	}

	msgs := h.compileChatMessages(traceID, req, rankings)

	ch, err := h.performChat(traceID, msgs)
	if err != nil {
		sendError(w, traceID, "performChat", err)
		return
	}

	if err := h.streamResponse(traceID, w, ch); err != nil {
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

func (h *handlers) vectorSearch(traceID string, req Request) ([]duck.Document, error) {
	question := req.Messages[len(req.Messages)-1].Content

	fmt.Printf("traceID: %s: vectorSearch: started: question: %s\n", traceID, question)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	queryVector, err := h.llmEmbed.Embed(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	docs, err := duck.Search(h.db, queryVector, 5)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	return docs, nil
}

func (h *handlers) rerank(traceID string, docs []duck.Document) ([]llamacpp.Ranking, error) {
	fmt.Printf("traceID: %s: rerank: started: docs: %d\n", traceID, len(docs))

	documents := make([]llamacpp.RankingDocument, len(docs))
	for i, doc := range docs {
		documents[i] = llamacpp.RankingDocument{Document: doc.Text, Embedding: doc.Embedding}
	}

	rankings, err := h.llmChat.Rerank(documents)
	if err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}

	return rankings, nil
}

func (h *handlers) compileChatMessages(traceID string, req Request, rankings []llamacpp.Ranking) []llamacpp.ChatMessage {
	fmt.Printf("traceID: %s: compileChatMessages: started\n", traceID)

	question := req.Messages[len(req.Messages)-1].Content

	msgs := make([]llamacpp.ChatMessage, len(req.Messages)+1)
	for i, msg := range req.Messages {
		msgs[i] = llamacpp.ChatMessage{Role: "user", Content: msg.Content}
	}

	const prompt = `
		- Use the following Context to answer the user's question.
		- If you don't know the answer, say that you don't know.
		- Responses should be properly formatted to be easily read.
		- Share code if code is presented in the context.
		- Do not include any additional information not present in the context.

		Context:
		
		%s

		Question: %s
		`

	var content string
	for _, ranking := range rankings[:2] {
		content = fmt.Sprintf("%s\n%s\n", content, ranking.Document)
	}

	finalPrompt := fmt.Sprintf(prompt, content, question)
	msgs[len(msgs)-1] = llamacpp.ChatMessage{Role: "user", Content: finalPrompt}

	return msgs
}

func (h *handlers) performChat(traceID string, msgs []llamacpp.ChatMessage) (<-chan llamacpp.ChatResponse, error) {
	fmt.Printf("traceID: %s: performChat: started: msgs: %d\n", traceID, len(msgs))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ch, err := h.llmChat.ChatCompletions(ctx, msgs, llamacpp.Params{
		TopK: 1.0,
		TopP: 0.9,
		Temp: 0.7,
	})
	if err != nil {
		return nil, fmt.Errorf("chat completions: %w", err)
	}

	return ch, nil
}

func (h *handlers) streamResponse(traceID string, w http.ResponseWriter, ch <-chan llamacpp.ChatResponse) error {
	fmt.Printf("traceID: %s: streamResponse: started\n", traceID)

	f, ok := w.(http.Flusher)
	if !ok {
		return fmt.Errorf("streaming not supported")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	f.Flush()

	id := uuid.NewString()
	var finalResponse strings.Builder

	for msg := range ch {
		if msg.Err != nil {
			fmt.Printf("traceID: %s: chat: ERROR: %s\n", traceID, msg.Err)
			fmt.Fprintf(w, "data: %s\n", newResponse(id, h.llmChat.ModelName(), "", "", msg.Err))
			f.Flush()
			break
		}

		finalResponse.WriteString(msg.Response)
		fmt.Fprintf(w, "data: %s\n", newResponse(id, h.llmChat.ModelName(), msg.Response, "", nil))
		f.Flush()
	}

	fr := finalResponse.String()
	if len(fr) > 0 {
		fmt.Fprintf(w, "data: %s\n", newResponse(id, h.llmChat.ModelName(), "", fr, nil))
		f.Flush()
	}

	w.Write([]byte("data: [DONE]\n"))
	f.Flush()

	return nil
}
