package main

import (
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"

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
