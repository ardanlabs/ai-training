package main

import (
	"database/sql"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example13/llamacpp"
	"github.com/marcboeker/go-duckdb/v2"
)

func dbConnection(llm *llamacpp.Llama, dimentions int) (*sql.DB, error) {
	connector, err := duckdb.NewConnector(dbPath, nil)
	if err != nil {
		return nil, fmt.Errorf("error creating connector: %w", err)
	}
	defer connector.Close()

	db := sql.OpenDB(connector)

	// -------------------------------------------------------------------------

	// Install and load VSS extension for vector similarity search.
	sql := `
		INSTALL vss; LOAD vss;
	`

	_, err = db.Exec(sql)
	if err != nil {
		return nil, fmt.Errorf("error loading VSS extension: %w", err)
	}

	// -------------------------------------------------------------------------

	checkSQL := `
		SELECT COUNT(*) 
		FROM information_schema.tables 
		WHERE table_name = 'items';
	`

	var tableExists int
	err = db.QueryRow(checkSQL).Scan(&tableExists)
	if err != nil {
		return nil, fmt.Errorf("error checking if table exists: %w", err)
	}

	if tableExists > 0 {
		var rowCount int
		err = db.QueryRow("SELECT COUNT(*) FROM items").Scan(&rowCount)
		if err != nil {
			return nil, fmt.Errorf("error checking row count: %w", err)
		}

		fmt.Printf("- table exists with %d rows\n", rowCount)
		return db, nil
	}

	// -------------------------------------------------------------------------

	_, err = db.Exec("SET hnsw_enable_experimental_persistence = true;")
	if err != nil {
		return nil, fmt.Errorf("error setting HNSW persistence: %w", err)
	}

	// -------------------------------------------------------------------------

	sql = `
		CREATE TABLE items (
			id        INTEGER   PRIMARY KEY,
			text      VARCHAR,
			embedding FLOAT[%d]
		);
	`

	sql = fmt.Sprintf(sql, dimentions)

	if _, err = db.Exec(sql); err != nil {
		return nil, fmt.Errorf("error creating table: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Print("LOADING DATA...")
	t := time.Now()

	if err := dbLoadChunks(db, llm); err != nil {
		return nil, fmt.Errorf("error loading data: %w", err)
	}

	fmt.Printf("Loaded data in %v\n", time.Since(t))

	// -------------------------------------------------------------------------

	sql = `
		CREATE INDEX idx_embedding ON items 
		USING HNSW (embedding) 
		WITH (metric = 'cosine');
	`

	if _, err = db.Exec(sql); err != nil {
		return nil, fmt.Errorf("error creating HNSW index: %w", err)
	}

	return db, nil
}

func dbLoadChunks(db *sql.DB, llm *llamacpp.Llama) error {
	data, err := os.ReadFile("zarf/data/book.chunks")
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	fmt.Print("\n")
	fmt.Print("\033[s")

	r := regexp.MustCompile(`<CHUNK>[\w\W]*?<\/CHUNK>`)
	chunks := r.FindAllString(string(data), -1)

	for counter, chunk := range chunks {
		fmt.Print("\033[u\033[K")
		fmt.Printf("Vectorizing Data: %d of %d", counter+1, len(chunks))

		chunk = strings.Trim(chunk, "<CHUNK>")
		chunk = strings.Trim(chunk, "</CHUNK>")

		vec, err := llm.Embed(chunk)
		if err != nil {
			return fmt.Errorf("embed chunk: %w", err)
		}

		chunk = strings.ReplaceAll(chunk, "'", "''")
		vecStr := strings.ReplaceAll(fmt.Sprintf("%v", vec), " ", ",")

		sql := fmt.Sprintf("INSERT INTO items (id, text, embedding) VALUES(%d, '%s', %v);", counter, chunk, vecStr)

		if _, err := db.Exec(sql); err != nil {
			return fmt.Errorf("insert chunk: %s %w", sql, err)
		}
	}

	fmt.Print("\n")

	return nil
}

type document struct {
	ID         int
	Text       string
	Embedding  []float64
	Similarity float64
}

func dbSearch(db *sql.DB, queryVector []float32, limit int) ([]document, error) {
	sql := `
		SELECT
			id,
			text,
			embedding,
			array_cosine_similarity(embedding, ?::FLOAT[%d]) as similarity
		FROM
			items
		ORDER BY
			similarity DESC
		LIMIT %d;
	`

	sql = fmt.Sprintf(sql, len(queryVector), limit)

	rows, err := db.Query(sql, queryVector)
	if err != nil {
		return nil, fmt.Errorf("error querying similar items: %w", err)
	}
	defer rows.Close()

	var docs []document

	for rows.Next() {
		var doc document
		var embedding []any

		if err := rows.Scan(&doc.ID, &doc.Text, &embedding, &doc.Similarity); err != nil {
			return nil, fmt.Errorf("error scanning row: %w", err)
		}

		doc.Embedding = make([]float64, len(embedding))
		for i, v := range embedding {
			doc.Embedding[i] = float64(v.(float32))
		}

		docs = append(docs, doc)
	}

	return docs, nil
}
