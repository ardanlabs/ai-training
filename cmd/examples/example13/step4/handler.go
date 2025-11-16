package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
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

func (r *Request) Decode(data []byte) error {
	return json.Unmarshal(data, r)
}

// =============================================================================

type muxConfig struct {
	llmEmbed *llamacpp.Llama
	llmChat  *llamacpp.Llama
	db       *sql.DB
}

func mux(cfg muxConfig) *http.ServeMux {
	mux := http.NewServeMux()

	rts := routes(cfg)

	mux.HandleFunc("POST /bookquestion", rts.bookQuestion)

	return mux
}

// =============================================================================

type routes struct {
	llmEmbed *llamacpp.Llama
	llmChat  *llamacpp.Llama
	db       *sql.DB
}

func (rts *routes) bookQuestion(w http.ResponseWriter, r *http.Request) {
	traceID := uuid.NewString()

	fmt.Printf("traceID %s: bookQuestion: started\n", traceID)
	defer fmt.Printf("traceID %s: bookQuestion: complete\n", traceID)

	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fmt.Printf("traceID %s: bookQuestion: ERROR: %s\n", traceID, err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// -------------------------------------------------------------------------

	question := req.Messages[len(req.Messages)-1].Content

	fmt.Printf("traceID %s: bookQuestion: question: %s\n", traceID, question)

	queryVector, err := func() ([]float32, error) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()

		queryVector, err := rts.llmEmbed.Embed(ctx, question)
		if err != nil {
			return nil, fmt.Errorf("embed: %w", err)
		}

		return queryVector, nil
	}()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	docs, err := duck.Search(rts.db, queryVector, 5)
	if err != nil {
		fmt.Printf("traceID %s: bookQuestion: ERROR: %s\n", traceID, err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// -------------------------------------------------------------------------

	documents := make([]llamacpp.RankingDocument, len(docs))
	for i, doc := range docs {
		documents[i] = llamacpp.RankingDocument{Document: doc.Text, Embedding: doc.Embedding}
	}

	rankings, err := rts.llmChat.Rerank(documents)
	if err != nil {
		fmt.Printf("traceID %s: bookQuestion: ERROR: %s\n", traceID, err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// -------------------------------------------------------------------------

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

	// -------------------------------------------------------------------------

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	ch, err := rts.llmChat.ChatCompletions(ctx, msgs, llamacpp.Params{
		TopK: 1.0,
		TopP: 0.9,
		Temp: 0.7,
	})
	if err != nil {
		fmt.Printf("traceID %s: bookQuestion: ERROR: %s\n", traceID, err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// -------------------------------------------------------------------------

	f, ok := w.(http.Flusher)
	if !ok {
		fmt.Printf("traceID %s: bookQuestion: ERROR: %s\n", traceID, "streaming not supported")
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// Start sending Server Sent Events.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	f.Flush()

	for msg := range ch {
		if msg.Err != nil {
			fmt.Printf("bookQuestion: status: %s traceID: %s", msg.Err, traceID)
			w.Write([]byte("data: [DONE]\n\n"))
			f.Flush()
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", msg.Response)
		f.Flush()
	}

	w.Write([]byte("data: [DONE]\n\n"))
	f.Flush()
}
