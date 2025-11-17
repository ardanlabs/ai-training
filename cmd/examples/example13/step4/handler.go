package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/duck"
	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
	"github.com/google/uuid"
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

// =============================================================================

type muxConfig struct {
	llmEmbed *llamacpp.Llama
	llmChat  *llamacpp.Llama
	db       *sql.DB
}

func mux(cfg muxConfig) http.Handler {
	mux := http.NewServeMux()

	rts := routes(cfg)

	mux.HandleFunc("POST /chat", rts.chat)

	return corsMiddleware(mux)
}

func sendError(w http.ResponseWriter, traceID string, context string, err error) {
	fmt.Printf("traceID: %s: chat: %s: ERROR: %s\n", traceID, context, err)
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// =============================================================================

type routes struct {
	llmEmbed *llamacpp.Llama
	llmChat  *llamacpp.Llama
	db       *sql.DB
}

func (rts *routes) chat(w http.ResponseWriter, r *http.Request) {
	traceID := uuid.NewString()

	fmt.Printf("traceID: %s: chat: started\n", traceID)
	defer fmt.Printf("traceID: %s: chat: complete\n", traceID)

	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, traceID, "NewDecoder", err)
		return
	}

	docs, err := rts.vectorSearch(traceID, req)
	if err != nil {
		sendError(w, traceID, "vectorSearch", err)
		return
	}

	rankings, err := rts.rerank(traceID, docs)
	if err != nil {
		sendError(w, traceID, "rerank", err)
		return
	}

	msgs := rts.compileChatMessages(traceID, req, rankings)

	ch, err := rts.performChat(traceID, msgs)
	if err != nil {
		sendError(w, traceID, "performChat", err)
		return
	}

	if err := rts.streamResponse(traceID, w, ch); err != nil {
		sendError(w, traceID, "streamResponse", err)
		return
	}
}

func (rts *routes) vectorSearch(traceID string, req Request) ([]duck.Document, error) {
	question := req.Messages[len(req.Messages)-1].Content

	fmt.Printf("traceID: %s: vectorSearch: started: question: %s\n", traceID, question)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	queryVector, err := rts.llmEmbed.Embed(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	docs, err := duck.Search(rts.db, queryVector, 5)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	return docs, nil
}

func (rts *routes) rerank(traceID string, docs []duck.Document) ([]llamacpp.Ranking, error) {
	fmt.Printf("traceID: %s: rerank: started: docs: %d\n", traceID, len(docs))

	documents := make([]llamacpp.RankingDocument, len(docs))
	for i, doc := range docs {
		documents[i] = llamacpp.RankingDocument{Document: doc.Text, Embedding: doc.Embedding}
	}

	rankings, err := rts.llmChat.Rerank(documents)
	if err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}

	return rankings, nil
}

func (rts *routes) compileChatMessages(traceID string, req Request, rankings []llamacpp.Ranking) []llamacpp.ChatMessage {
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

func (rts *routes) performChat(traceID string, msgs []llamacpp.ChatMessage) (<-chan llamacpp.ChatResponse, error) {
	fmt.Printf("traceID: %s: performChat: started: msgs: %d\n", traceID, len(msgs))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ch, err := rts.llmChat.ChatCompletions(ctx, msgs, llamacpp.Params{
		TopK: 1.0,
		TopP: 0.9,
		Temp: 0.7,
	})
	if err != nil {
		return nil, fmt.Errorf("chat completions: %w", err)
	}

	return ch, nil
}

func (rts *routes) streamResponse(traceID string, w http.ResponseWriter, ch <-chan llamacpp.ChatResponse) error {
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
			fmt.Fprintf(w, "data: %s\n", newResponse(id, rts.llmChat.ModelName(), "", "", msg.Err))
			f.Flush()
			break
		}

		finalResponse.WriteString(msg.Response)
		fmt.Fprintf(w, "data: %s\n", newResponse(id, rts.llmChat.ModelName(), msg.Response, "", nil))
		f.Flush()
	}

	fr := finalResponse.String()
	if len(fr) > 0 {
		fmt.Fprintf(w, "data: %s\n", newResponse(id, rts.llmChat.ModelName(), "", fr, nil))
		f.Flush()
	}

	w.Write([]byte("data: [DONE]\n"))
	f.Flush()

	return nil
}
