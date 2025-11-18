package website

import (
	"database/sql"
	"fmt"
	"net/http"

	"github.com/ardanlabs/llamacpp"
)

type Config struct {
	LLMEmbed *llamacpp.Llama
	DBMChat  *llamacpp.Llama
	DB       *sql.DB
}

func WebAPI(cfg Config) http.Handler {
	mux := http.NewServeMux()

	rts := handlers{
		llmEmbed: cfg.LLMEmbed,
		llmChat:  cfg.DBMChat,
		db:       cfg.DB,
	}

	mux.HandleFunc("POST /chat", rts.chat)
	mux.HandleFunc("/", rts.fileServerReact())

	return corsMiddleware(mux)
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

func sendError(w http.ResponseWriter, traceID string, context string, err error) {
	fmt.Printf("traceID: %s: chat: %s: ERROR: %s\n", traceID, context, err)
	http.Error(w, err.Error(), http.StatusInternalServerError)
}
