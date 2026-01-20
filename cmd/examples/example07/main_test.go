// This test shows a way to test output from LLM.
// We know it should be a valid SQL to we use vitess to parse the SQL.
//
// # This requires running the following commands:
//
//	$ make compose-up
//	$ make kronk-up
package main

import (
	"fmt"
	"testing"

	"github.com/ardanlabs/ai-training/foundation/client"
	"vitess.io/vitess/go/vt/sqlparser"
)

func TestValidSQL(t *testing.T) {
	db, err := initSQLDB(t.Context())
	if err != nil {
		t.Fatalf("initSQLDB: %v", err)
	}
	defer db.Close()

	question := "Which user bought most products?"

	llm := client.NewLLM(url, model)

	sql, err := llm.ChatCompletions(t.Context(), fmt.Sprintf(query, question))
	if err != nil {
		t.Fatalf("LLM query: %v", err)
	}
	t.Logf("query:\n%s", sql)

	p := sqlparser.NewTestParser()
	_, err = p.Parse(sql)
	if err != nil {
		t.Fatalf("bad SQL: %v", err)
	}
}
