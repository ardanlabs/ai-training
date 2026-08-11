// Package age provides support for talking to a graph stored in Apache AGE,
// the graph extension for Postgres.
//
// AGE does not add a wire protocol. A Cypher query is an ordinary SQL query
// that calls the cypher() function and declares the shape of what comes back:
//
//	SELECT * FROM ag_catalog.cypher('book', $$ MATCH (c:Chunk) RETURN c.id $$)
//	AS (id agtype)
//
// Two things about that make a small package worth having. The column list has
// to be spelled out at every call site, and every value arrives as agtype,
// Postgres' JSON-ish graph type, which means a string comes back with its
// quotes still attached. Query hides the first and Scalar/ParseID handle the
// second.
//
// The cypher() call is fully qualified rather than relying on search_path.
// A pooled *sql.DB hands out a different connection per call, so per-session
// state is not something the caller can count on. The extension itself is
// loaded by the server (session_preload_libraries=age in compose.yaml).
//
// Even so, ag_catalog is put on the search_path as a connect parameter, which
// applies to every connection the pool opens. create_graph needs it: the index
// DDL it emits for the new graph's tables names the graphid_ops operator class
// unqualified, and without ag_catalog in scope that name does not resolve.
package age

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/jmoiron/sqlx"
)

// Config is the required properties to use the graph database.
type Config struct {
	User         string
	Password     string
	Host         string
	Name         string
	GraphName    string
	MaxIdleConns int
	MaxOpenConns int
	DisableTLS   bool
}

// Graph represents a connection to a single named graph.
type Graph struct {
	db   *sqlx.DB
	name string
}

// Open connects to the database and makes sure the named graph exists.
func Open(ctx context.Context, cfg Config) (*Graph, error) {
	db, err := sqldb.Open(sqldb.Config{
		User:         cfg.User,
		Password:     cfg.Password,
		Host:         cfg.Host,
		Name:         cfg.Name,
		Schema:       `ag_catalog, "$user", public`,
		MaxIdleConns: cfg.MaxIdleConns,
		MaxOpenConns: cfg.MaxOpenConns,
		DisableTLS:   cfg.DisableTLS,
	})
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	if err := sqldb.StatusCheck(ctx, db); err != nil {
		db.Close()
		return nil, fmt.Errorf("status check: %w", err)
	}

	g := Graph{
		db:   db,
		name: cfg.GraphName,
	}

	if err := g.createGraph(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("create graph: %w", err)
	}

	return &g, nil
}

// DB returns the underlying database connection. Chunk text and embeddings
// live in ordinary tables, so callers need it.
func (g *Graph) DB() *sqlx.DB {
	return g.db
}

// Close releases the database resources.
func (g *Graph) Close() error {
	return g.db.Close()
}

// createGraph creates the graph on first use. create_graph fails if the graph
// is already there and there is no IF NOT EXISTS form, so the catalog is
// checked first.
func (g *Graph) createGraph(ctx context.Context) error {
	const check = `SELECT count(*) FROM ag_catalog.ag_graph WHERE name = $1`

	var count int
	if err := g.db.QueryRowContext(ctx, check, g.name).Scan(&count); err != nil {
		return fmt.Errorf("check catalog: %w", err)
	}

	if count > 0 {
		return nil
	}

	const create = `SELECT ag_catalog.create_graph($1)`
	if _, err := g.db.ExecContext(ctx, create, g.name); err != nil {
		return fmt.Errorf("create_graph: %w", err)
	}

	return nil
}

// =============================================================================

// Exec runs a Cypher statement that returns nothing, such as CREATE or MERGE.
//
// AGE still requires a column list even when there is nothing to return, so
// one throwaway agtype column is declared.
func (g *Graph) Exec(ctx context.Context, cypher string) error {
	if _, err := g.db.ExecContext(ctx, g.wrap(cypher, "v agtype")); err != nil {
		return fmt.Errorf("exec cypher: %w", err)
	}

	return nil
}

// Query runs a Cypher statement and returns its rows as raw agtype values, one
// slice per row. The cols are the AGE column declarations matching the RETURN
// clause, for example "id agtype", "name agtype".
func (g *Graph) Query(ctx context.Context, cypher string, cols ...string) ([][]string, error) {
	if len(cols) == 0 {
		return nil, fmt.Errorf("at least one column declaration is required")
	}

	rows, err := g.db.QueryContext(ctx, g.wrap(cypher, cols...))
	if err != nil {
		return nil, fmt.Errorf("query cypher: %w", err)
	}
	defer rows.Close()

	var out [][]string

	for rows.Next() {

		// Every column is agtype, which the driver hands over as text.
		values := make([]string, len(cols))
		dest := make([]any, len(cols))
		for i := range values {
			dest[i] = &values[i]
		}

		if err := rows.Scan(dest...); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		out = append(out, values)
	}

	return out, rows.Err()
}

// Count runs a Cypher statement whose single column is a count and returns it.
func (g *Graph) Count(ctx context.Context, cypher string) (int, error) {
	rows, err := g.Query(ctx, cypher, "count agtype")
	if err != nil {
		return 0, err
	}

	if len(rows) == 0 {
		return 0, nil
	}

	count, err := ParseID(rows[0][0])
	if err != nil {
		return 0, fmt.Errorf("parse count: %w", err)
	}

	return count, nil
}

// wrap turns a Cypher statement into the SQL that AGE understands. The Cypher
// is dollar quoted, so it cannot carry parameters; string values have to go
// through Quote.
func (g *Graph) wrap(cypher string, cols ...string) string {
	return fmt.Sprintf("SELECT * FROM ag_catalog.cypher('%s', $cypher$ %s $cypher$) AS (%s)",
		g.name, cypher, strings.Join(cols, ", "))
}

// =============================================================================

// Quote renders a Go string as a Cypher string literal, escaping what would
// otherwise end the literal or start an escape sequence.
func Quote(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `'`, `\'`)

	// The Cypher is embedded in a dollar quoted SQL string. A literal
	// "$cypher$" inside it would close that string early.
	s = strings.ReplaceAll(s, "$cypher$", "")

	return "'" + s + "'"
}

// Ints renders ids as a comma separated list for a Cypher IN clause.
func Ints(ids []int) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = strconv.Itoa(id)
	}

	return strings.Join(parts, ", ")
}

// Scalar unwraps an agtype value into a plain Go string. Strings arrive with
// their quotes attached and may carry a "::vertex" style type annotation.
func Scalar(v string) string {
	v = strings.TrimSpace(v)

	if i := strings.LastIndex(v, "::"); i != -1 {
		v = v[:i]
	}

	if len(v) >= 2 && strings.HasPrefix(v, `"`) && strings.HasSuffix(v, `"`) {
		v = v[1 : len(v)-1]
		v = strings.ReplaceAll(v, `\"`, `"`)
		v = strings.ReplaceAll(v, `\\`, `\`)
	}

	return v
}

// ParseID unwraps an agtype number into an int. AGE writes whole numbers
// without a fraction, but a count that went through a float path can come back
// as "12.0", so both are accepted.
func ParseID(v string) (int, error) {
	s := Scalar(v)

	if id, err := strconv.Atoi(s); err == nil {
		return id, nil
	}

	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("parse agtype number %q: %w", v, err)
	}

	return int(f), nil
}
