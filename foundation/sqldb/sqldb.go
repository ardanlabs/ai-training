// Package sqldb provides support for access the database.
package sqldb

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
)

// lib/pq errorCodeNames
// https://github.com/lib/pq/blob/master/error.go#L178
const (
	uniqueViolation = "23505"
	undefinedTable  = "42P01"
)

// Set of error variables for CRUD operations.
var (
	ErrDBNotFound        = sql.ErrNoRows
	ErrDBDuplicatedEntry = errors.New("duplicated entry")
	ErrUndefinedTable    = errors.New("undefined table")
)

// Config is the required properties to use the database.
type Config struct {
	User         string
	Password     string
	Host         string
	Name         string
	Schema       string
	MaxIdleConns int
	MaxOpenConns int
	DisableTLS   bool
	ReadOnly     bool
}

// Open knows how to open a database connection based on the configuration.
func Open(cfg Config) (*sqlx.DB, error) {
	sslMode := "require"
	if cfg.DisableTLS {
		sslMode = "disable"
	}

	q := make(url.Values)
	q.Set("sslmode", sslMode)
	q.Set("timezone", "utc")
	if cfg.Schema != "" {
		q.Set("search_path", cfg.Schema)
	}
	if cfg.ReadOnly {
		q.Set("options", "-c default_transaction_read_only=on")
	}

	u := url.URL{
		Scheme:   "postgres",
		User:     url.UserPassword(cfg.User, cfg.Password),
		Host:     cfg.Host,
		Path:     cfg.Name,
		RawQuery: q.Encode(),
	}

	db, err := sqlx.Open("pgx", u.String())
	if err != nil {
		return nil, err
	}
	db.SetMaxIdleConns(cfg.MaxIdleConns)
	db.SetMaxOpenConns(cfg.MaxOpenConns)

	return db, nil
}

// ErrDBUnreachable is returned by StatusCheck when the database cannot be
// reached within the status-check timeout. It usually means the database
// server is not running (e.g. forgot to run `make compose-up`).
var ErrDBUnreachable = errors.New("database is unreachable; is the server running? try `make compose-up`")

// StatusCheckTimeout caps how long StatusCheck will wait for the database
// to respond before returning ErrDBUnreachable.
const StatusCheckTimeout = 5 * time.Second

// StatusCheck returns nil if it can successfully talk to the database. It
// returns a non-nil error otherwise. The check is bounded by
// StatusCheckTimeout regardless of the deadline on the supplied context,
// so a long-lived parent context will not cause the check to hang when
// the database is down.
func StatusCheck(ctx context.Context, db *sqlx.DB) error {
	ctx, cancel := context.WithTimeout(ctx, StatusCheckTimeout)
	defer cancel()

	var lastErr error
	for attempts := 1; ; attempts++ {
		if err := db.PingContext(ctx); err == nil {
			lastErr = nil
			break
		} else {
			lastErr = err
		}

		time.Sleep(time.Duration(attempts) * 100 * time.Millisecond)

		if ctx.Err() != nil {
			return fmt.Errorf("%w: %w", ErrDBUnreachable, lastErr)
		}
	}

	// Run a simple query to determine connectivity.
	// Running this query forces a round trip through the database.
	const q = `SELECT TRUE`
	var tmp bool
	if err := db.QueryRowContext(ctx, q).Scan(&tmp); err != nil {
		return fmt.Errorf("%w: %w", ErrDBUnreachable, err)
	}
	return nil
}

// ExecContext is a helper function to execute a CUD operation with
// logging and tracing.
func ExecContext(ctx context.Context, db sqlx.ExtContext, query string) error {
	return NamedExecContext(ctx, db, query, struct{}{})
}

// NamedExecContext is a helper function to execute a CUD operation with
// logging and tracing where field replacement is necessary.
func NamedExecContext(ctx context.Context, db sqlx.ExtContext, query string, data any) (err error) {
	if _, err := sqlx.NamedExecContext(ctx, db, query, data); err != nil {
		pqerr, ok := errors.AsType[*pgconn.PgError](err)
		if ok {
			switch pqerr.Code {
			case undefinedTable:
				return ErrUndefinedTable
			case uniqueViolation:
				return ErrDBDuplicatedEntry
			}
		}
		return err
	}

	return nil
}

// QuerySlice is a helper function for executing queries that return a
// collection of data to be unmarshalled into a slice.
func QuerySlice[T any](ctx context.Context, db sqlx.ExtContext, query string, dest *[]T) error {
	return namedQuerySlice(ctx, db, query, struct{}{}, dest, false)
}

// NamedQuerySlice is a helper function for executing queries that return a
// collection of data to be unmarshalled into a slice where field replacement is
// necessary.
func NamedQuerySlice[T any](ctx context.Context, db sqlx.ExtContext, query string, data any, dest *[]T) error {
	return namedQuerySlice(ctx, db, query, data, dest, false)
}

// NamedQuerySliceUsingIn is a helper function for executing queries that return
// a collection of data to be unmarshalled into a slice where field replacement
// is necessary. Use this if the query has an IN clause.
func NamedQuerySliceUsingIn[T any](ctx context.Context, db sqlx.ExtContext, query string, data any, dest *[]T) error {
	return namedQuerySlice(ctx, db, query, data, dest, true)
}

func namedQuerySlice[T any](ctx context.Context, db sqlx.ExtContext, query string, data any, dest *[]T, withIn bool) (err error) {
	var rows *sqlx.Rows

	switch withIn {
	case true:
		rows, err = func() (*sqlx.Rows, error) {
			named, args, err := sqlx.Named(query, data)
			if err != nil {
				return nil, err
			}

			query, args, err := sqlx.In(named, args...)
			if err != nil {
				return nil, err
			}

			query = db.Rebind(query)
			return db.QueryxContext(ctx, query, args...)
		}()

	default:
		rows, err = sqlx.NamedQueryContext(ctx, db, query, data)
	}

	if err != nil {
		var pqerr *pgconn.PgError
		if errors.As(err, &pqerr) && pqerr.Code == undefinedTable {
			return ErrUndefinedTable
		}
		return err
	}
	defer rows.Close()

	var slice []T
	for rows.Next() {
		v := new(T)
		if err := rows.StructScan(v); err != nil {
			return err
		}
		slice = append(slice, *v)
	}
	*dest = slice

	return nil
}

// QueryStruct is a helper function for executing queries that return a
// single value to be unmarshalled into a struct type where field replacement is necessary.
func QueryStruct(ctx context.Context, db sqlx.ExtContext, query string, dest any) error {
	return namedQueryStruct(ctx, db, query, struct{}{}, dest, false)
}

// NamedQueryStruct is a helper function for executing queries that return a
// single value to be unmarshalled into a struct type where field replacement is necessary.
func NamedQueryStruct(ctx context.Context, db sqlx.ExtContext, query string, data any, dest any) error {
	return namedQueryStruct(ctx, db, query, data, dest, false)
}

// NamedQueryStructUsingIn is a helper function for executing queries that return
// a single value to be unmarshalled into a struct type where field replacement
// is necessary. Use this if the query has an IN clause.
func NamedQueryStructUsingIn(ctx context.Context, db sqlx.ExtContext, query string, data any, dest any) error {
	return namedQueryStruct(ctx, db, query, data, dest, true)
}

func namedQueryStruct(ctx context.Context, db sqlx.ExtContext, query string, data any, dest any, withIn bool) (err error) {
	var rows *sqlx.Rows

	switch withIn {
	case true:
		rows, err = func() (*sqlx.Rows, error) {
			named, args, err := sqlx.Named(query, data)
			if err != nil {
				return nil, err
			}

			query, args, err := sqlx.In(named, args...)
			if err != nil {
				return nil, err
			}

			query = db.Rebind(query)
			return db.QueryxContext(ctx, query, args...)
		}()

	default:
		rows, err = sqlx.NamedQueryContext(ctx, db, query, data)
	}

	if err != nil {
		var pqerr *pgconn.PgError
		if errors.As(err, &pqerr) && pqerr.Code == undefinedTable {
			return ErrUndefinedTable
		}
		return err
	}
	defer rows.Close()

	if !rows.Next() {
		return ErrDBNotFound
	}

	if err := rows.StructScan(dest); err != nil {
		return err
	}

	return nil
}

// QueryMap executes a query and places results into a map.
func QueryMap(ctx context.Context, db sqlx.ExtContext, query string, dest *[]map[string]any) (err error) {
	var rows *sqlx.Rows

	rows, err = sqlx.NamedQueryContext(ctx, db, query, struct{}{})
	if err != nil {
		var pqerr *pgconn.PgError
		if errors.As(err, &pqerr) && pqerr.Code == undefinedTable {
			return ErrUndefinedTable
		}
		return err
	}

	defer rows.Close()

	for rows.Next() {
		m := map[string]any{}
		if err := rows.MapScan(m); err != nil {
			return err
		}

		*dest = append(*dest, m)
	}

	return nil
}

// QueryString provides a pretty print version of the query and parameters.
func QueryString(query string, args any) string {
	query, params, err := sqlx.Named(query, args)
	if err != nil {
		return err.Error()
	}

	for _, param := range params {
		var value string
		switch v := param.(type) {
		case string:
			value = fmt.Sprintf("'%s'", v)
		case []byte:
			value = fmt.Sprintf("'%s'", string(v))
		default:
			value = fmt.Sprintf("%v", v)
		}
		query = strings.Replace(query, "?", value, 1)
	}

	query = strings.ReplaceAll(query, "\t", "")
	query = strings.ReplaceAll(query, "\n", " ")

	return strings.Trim(query, " ")
}
