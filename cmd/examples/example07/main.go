// This example shows you how to get a model to generate SQL queries.
//
// # Running the example:
//
//	$ make example07
//
// # This requires running the following commands:
//
//	$ make compose-up
//  $ make kronk-up

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
)

var (
	url   = "http://localhost:8080/v1/chat/completions"
	model = "cerebras_Qwen3-Coder-REAP-25B-A3B-Q8_0"
)

func init() {
	if v := os.Getenv("LLM_SERVER"); v != "" {
		url = v
	}

	if v := os.Getenv("LLM_MODEL"); v != "" {
		model = v
	}
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*60*time.Second)
	defer cancel()

	db, err := initSQLDB(ctx)
	if err != nil {
		return fmt.Errorf("initSQLDB: %w", err)
	}
	defer db.Close()

	// -------------------------------------------------------------------------

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("\nAsk a question about the garage sale system: ")

	question, _ := reader.ReadString('\n')
	if question == "" {
		return nil
	}

	fmt.Print("\nGive me a second...\n\n")

	// -------------------------------------------------------------------------

	llm := client.NewLLM(url, model)

	query, err := llm.ChatCompletions(ctx, fmt.Sprintf(query, question))
	if err != nil {
		return fmt.Errorf("chat completions: %w", err)
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

	var builder strings.Builder
	for i, m := range data {
		builder.WriteString(fmt.Sprintf("RESULT: %d\n", i+1))
		for k, v := range m {
			builder.WriteString(fmt.Sprintf("KEY: %s, VAL: %v\n", k, v))
		}
		builder.WriteString("\n")
	}

	answer, err := llm.ChatCompletions(ctx, fmt.Sprintf(response, builder.String(), question))
	if err != nil {
		return fmt.Errorf("do: %w", err)
	}

	fmt.Println("ANSWER:")
	fmt.Print("-----------------------------------------------\n\n")
	fmt.Println(answer)
	fmt.Print("\n")

	return nil
}
