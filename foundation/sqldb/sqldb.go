// Package sqldb provides support for access the database.
package sqldb

import (
	"context"
	"database/sql"
	"errors"
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

	// url.Values encodes a space as "+", which pgx passes through literally, so
	// a search_path like `ag_catalog, "$user", public` arrived as
	// `ag_catalog,+"$user",+public` and public was no longer on the path.
	u := url.URL{
		Scheme:   "postgres",
		User:     url.UserPassword(cfg.User, cfg.Password),
		Host:     cfg.Host,
		Path:     cfg.Name,
		RawQuery: strings.ReplaceAll(q.Encode(), "+", "%20"),
	}

	db, err := sqlx.Open("pgx", u.String())
	if err != nil {
		return nil, err
	}
	db.SetMaxIdleConns(cfg.MaxIdleConns)
	db.SetMaxOpenConns(cfg.MaxOpenConns)

	return db, nil
}

// StatusCheck returns nil if it can successfully talk to the database. It
// returns a non-nil error otherwise.
func StatusCheck(ctx context.Context, db *sqlx.DB) error {

	// If the user doesn't give us a deadline set 1 second.
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Second)
		defer cancel()
	}

	for attempts := 1; ; attempts++ {
		if err := db.Ping(); err == nil {
			break
		}

		time.Sleep(time.Duration(attempts) * 100 * time.Millisecond)

		if ctx.Err() != nil {
			return ctx.Err()
		}
	}

	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Run a simple query to determine connectivity.
	// Running this query forces a round trip through the database.
	const q = `SELECT TRUE`
	var tmp bool
	return db.QueryRowContext(ctx, q).Scan(&tmp)
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
		var pqerr *pgconn.PgError
		if errors.As(err, &pqerr) {
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

	// Next returns false on a failure while reading rows as well as at the
	// end, so without this check a broken query looks like a short result.
	if err := rows.Err(); err != nil {
		return err
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
		if err := rows.Err(); err != nil {
			return err
		}
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

	if err := rows.Err(); err != nil {
		return err
	}

	return nil
}
