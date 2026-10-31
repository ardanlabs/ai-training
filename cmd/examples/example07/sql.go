package main

import (
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/jmoiron/sqlx"
)

var (
	//go:embed prompts/query.txt
	query string

	//go:embed prompts/response.txt
	response string

	//go:embed sql/schema.sql
	schemaSQL string

	//go:embed sql/insert.sql
	insertSQL string
)

func initSQLDB(ctx context.Context) (*sqlx.DB, error) {
	fmt.Println("\nConnecting to the DB")

	db, err := dbConnection()
	if err != nil {
		return nil, fmt.Errorf("dbConnection: %w", err)
	}

	fmt.Println("Creating Schema")

	if err := dbExecute(ctx, db, schemaSQL); err != nil {
		return nil, fmt.Errorf("dbExecute: %w", err)
	}

	fmt.Println("Inserting Data")

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

// cleanSQL strips the markdown code fence a model often wraps around the
// SQL it generates, so the statement can be executed as is.
func cleanSQL(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```sql")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")

	return strings.TrimSpace(s)
}

func dbExecute(ctx context.Context, db *sqlx.DB, query string) (err error) {
	if err := sqldb.StatusCheck(ctx, db); err != nil {
		return fmt.Errorf("status check database: %w", err)
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}

	defer func() {
		if errTx := tx.Rollback(); errTx != nil {
			if errors.Is(errTx, sql.ErrTxDone) {
				return
			}

			err = errors.Join(err, fmt.Errorf("rollback: %w", errTx))
			return
		}
	}()

	if _, err := tx.ExecContext(ctx, query); err != nil {
		return fmt.Errorf("exec: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}

	return nil
}
