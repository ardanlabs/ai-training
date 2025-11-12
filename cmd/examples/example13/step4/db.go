package main

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/marcboeker/go-duckdb/v2"
)

func dbConnection(em *EmbeddingModel, dimentions int) (*sql.DB, error) {
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

		fmt.Printf("Table exists with %d rows\n", rowCount)
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

	if err := loadData(db, em); err != nil {
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

func dbSearch(db *sql.DB, queryVector []float32) error {
	sql := `
		SELECT
			id,
			text,
			array_cosine_similarity(embedding, ?::FLOAT[%d]) as similarity
		FROM
			items
		ORDER BY
			similarity DESC
		LIMIT 6;
	`

	sql = fmt.Sprintf(sql, len(queryVector))

	rows, err := db.Query(sql, queryVector)
	if err != nil {
		return fmt.Errorf("error querying similar items: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var id int
		var text string
		var similarity float64

		if err := rows.Scan(&id, &text, &similarity); err != nil {
			return fmt.Errorf("error scanning row: %w", err)
		}

		fmt.Printf("---\nID: %d\nText: %s...\nSimilarity: %.4f\n\n", id, text[1:min(len(text), 200)], similarity)
	}

	return nil
}
