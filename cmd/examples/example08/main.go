// This example shows you how to use the Llama3.2 model to generate SQL queries.
//
// # Running the example:
//
//	$ make openwebui
//    Use the OpenWebUI app with the Llama3.2:latest model.
//
// # This requires running the following commands:
//
//	$ make compose-up // This starts MongoDB and OpenWebUI in docker compose.
//  $ make ollama-up  // This starts the Ollama service.

package main

import (
	"bufio"
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/jmoiron/sqlx"
	"github.com/tmc/langchaingo/llms/ollama"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	db, err := dbInit(ctx)
	if err != nil {
		return fmt.Errorf("dbInit: %w", err)
	}

	defer db.Close()

	// -------------------------------------------------------------------------

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Ask a question about the garage sale system: ")

	question, _ := reader.ReadString('\n')
	if question == "" {
		return nil
	}

	fmt.Print("\nGive me a second...\n\n")

	// -------------------------------------------------------------------------

	query, err := getQuery(ctx, question)
	if err != nil {
		return fmt.Errorf("getQuery: %w", err)
	}

	fmt.Println("QUERY:")
	fmt.Print("-----------------------------------------------\n\n")
	fmt.Println(query)
	fmt.Print("\n")

	// -------------------------------------------------------------------------

	data := []map[string]any{}
	if err := sqldb.QueryMap(ctx, db, query, &data); err != nil {
		return fmt.Errorf("execQuery: %w", err)
	}

	fmt.Println("DATA:")
	fmt.Print("-----------------------------------------------\n\n")

	for i, m := range data {
		fmt.Printf("RESULT: %d\n", i+1)
		for k, v := range m {
			fmt.Printf("KEY: %s, VAL: %v\n", k, v)
		}
		fmt.Print("\n")
	}

	// -------------------------------------------------------------------------

	answer, err := getResponse(ctx, question, data)
	if err != nil {
		return fmt.Errorf("getQuery: %w", err)
	}

	fmt.Println("ANSWER:")
	fmt.Print("-----------------------------------------------\n\n")
	fmt.Println(answer)
	fmt.Print("\n")

	return nil
}

var (
	//go:embed prompts/query.txt
	query string

	//go:embed prompts/response.txt
	response string
)

func getQuery(ctx context.Context, question string) (string, error) {

	// Open a connection with ollama to access the model.
	llm, err := ollama.New(
		ollama.WithModel("llama3.2-vision"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		return "", fmt.Errorf("ollama: %w", err)
	}

	prompt := fmt.Sprintf(query, question)

	result, err := llm.Call(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("call: %w", err)
	}

	return result, nil
}

func getResponse(ctx context.Context, question string, data []map[string]any) (string, error) {

	// Open a connection with ollama to access the model.
	llm, err := ollama.New(
		ollama.WithModel("llama3.2-vision"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		return "", fmt.Errorf("ollama: %w", err)
	}

	var builder strings.Builder
	for i, m := range data {
		builder.WriteString(fmt.Sprintf("RESULT: %d\n", i+1))
		for k, v := range m {
			builder.WriteString(fmt.Sprintf("KEY: %s, VAL: %v\n", k, v))
		}
		builder.WriteString("\n")
	}

	prompt := fmt.Sprintf(response, builder.String(), question)

	result, err := llm.Call(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("call: %w", err)
	}

	return result, nil
}

// =============================================================================

var (
	//go:embed sql/schema.sql
	schemaSQL string

	//go:embed sql/insert.sql
	insertSQL string
)

func dbInit(ctx context.Context) (*sqlx.DB, error) {
	db, err := dbConnection()
	if err != nil {
		return nil, fmt.Errorf("dbConnection: %w", err)
	}

	if err := dbExecute(ctx, db, schemaSQL); err != nil {
		return nil, fmt.Errorf("dbExecute: %w", err)
	}

	if err := dbExecute(ctx, db, insertSQL); err != nil {
		return nil, fmt.Errorf("dbExecute: %w", err)
	}

	return db, nil
}

func dbConnection() (*sqlx.DB, error) {
	db, err := sqldb.Open(sqldb.Config{
		User:         "postgres",
		Password:     "postgres",
		Host:         "localhost:5432",
		Name:         "postgres",
		MaxIdleConns: 0,
		MaxOpenConns: 0,
		DisableTLS:   true,
	})
	if err != nil {
		return nil, fmt.Errorf("connecting to db: %w", err)
	}

	return db, nil
}

func dbExecute(ctx context.Context, db *sqlx.DB, query string) error {
	if err := sqldb.StatusCheck(ctx, db); err != nil {
		return fmt.Errorf("status check database: %w", err)
	}

	tx, err := db.Begin()
	if err != nil {
		return err
	}

	defer func() {
		if errTx := tx.Rollback(); errTx != nil {
			if errors.Is(errTx, sql.ErrTxDone) {
				return
			}

			err = fmt.Errorf("rollback: %w", errTx)
			return
		}
	}()

	if _, err := tx.Exec(query); err != nil {
		return fmt.Errorf("exec: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}

	return nil
}
