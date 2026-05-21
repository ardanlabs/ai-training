// Package rag provides shared retrieval-augmented generation helpers.
package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/jmoiron/sqlx"
)

// Document represents a document stored in pgvector.
type Document struct {
	ID        int
	Name      string
	Text      string
	Embedding []float64
}

// SearchResult represents a result from a vector similarity search.
type SearchResult struct {
	ID         int
	Name       string
	Text       string
	Distance   float64
	Similarity float64
}

// SearchDocuments embeds the query string, then performs a cosine-distance
// search against the documents table and returns the top-N results.
func SearchDocuments(ctx context.Context, db *sqlx.DB, llm *client.LLM, query string, topN int) ([]SearchResult, error) {
	embedding, err := llm.EmbedText(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	return SearchDocumentsWithEmbedding(ctx, db, embedding, topN)
}

// SearchDocumentsWithEmbedding performs a cosine-distance search using a
// pre-computed embedding vector.
func SearchDocumentsWithEmbedding(ctx context.Context, db *sqlx.DB, embedding []float64, topN int) ([]SearchResult, error) {
	const stmt = `
SELECT
	id,
	COALESCE(name, '') AS name,
	text,
	embedding <=> $1::vector AS distance,
	1 - (embedding <=> $1::vector) AS similarity
FROM
	documents
ORDER BY
	embedding <=> $1::vector
LIMIT $2
`

	rows, err := db.QueryContext(ctx, stmt, vector.FormatPGVector(embedding), topN)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var r SearchResult
		if err := rows.Scan(&r.ID, &r.Name, &r.Text, &r.Distance, &r.Similarity); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}
		results = append(results, r)
	}

	return results, rows.Err()
}

// BuildContext formats search results into a numbered text block suitable
// for injection into an LLM prompt.
func BuildContext(results []SearchResult) string {
	var sb strings.Builder

	for i, r := range results {
		if r.Name != "" {
			fmt.Fprintf(&sb, "[%d] %s\n%s\n\n", i+1, r.Name, r.Text)
		} else {
			fmt.Fprintf(&sb, "[%d]\n%s\n\n", i+1, r.Text)
		}
	}

	return strings.TrimSpace(sb.String())
}
