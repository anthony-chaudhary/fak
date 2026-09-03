package dataslot

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	DefaultMaxRows  = 50
	AbsoluteMaxRows = 200
	DefaultMaxBytes = 65536
	DefaultTimeout  = 5 * time.Second
)

var (
	ErrMutationRefused = errors.New("dataslot: mutating SQL statements are refused; only read-only queries are permitted")
	ErrTargetNotFound  = errors.New("dataslot: target database file does not exist")
	ErrExecutionFailed = errors.New("dataslot: query execution failed")
	ErrTimeout         = errors.New("dataslot: query execution deadline exceeded")
)

// QueryOptions defines bounds and deadlines for read-only database queries.
type QueryOptions struct {
	MaxRows  int           `json:"max_rows"`
	MaxBytes int           `json:"max_bytes"`
	Timeout  time.Duration `json:"timeout"`
}

func (o *QueryOptions) normalize() QueryOptions {
	res := QueryOptions{
		MaxRows:  DefaultMaxRows,
		MaxBytes: DefaultMaxBytes,
		Timeout:  DefaultTimeout,
	}
	if o != nil {
		if o.MaxRows > 0 {
			res.MaxRows = o.MaxRows
			if res.MaxRows > AbsoluteMaxRows {
				res.MaxRows = AbsoluteMaxRows
			}
		}
		if o.MaxBytes > 0 {
			res.MaxBytes = o.MaxBytes
		}
		if o.Timeout > 0 {
			res.Timeout = o.Timeout
		}
	}
	return res
}

// QueryResult holds bounded query output.
type QueryResult struct {
	Columns    []string `json:"columns"`
	Rows       [][]any  `json:"rows"`
	RowCount   int      `json:"row_count"`
	TotalRows  int      `json:"total_rows"`
	Truncated  bool     `json:"truncated"`
	DurationMs float64  `json:"duration_ms"`
}

// QueryRunner executes bounded read-only queries against a database target.
type QueryRunner interface {
	ExecuteQuery(ctx context.Context, target, sql string, opts *QueryOptions) (*QueryResult, error)
}

var (
	commentLineRE  = regexp.MustCompile(`--[^\r\n]*`)
	commentBlockRE = regexp.MustCompile(`/\*[\s\S]*?\*/`)
	forbiddenSQLRE = regexp.MustCompile(`(?i)\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|GRANT|REVOKE|VACUUM|REINDEX)\b`)
	allowedHeadRE  = regexp.MustCompile(`(?i)^(SELECT|EXPLAIN|WITH|PRAGMA)\b`)
	safePragmaRE   = regexp.MustCompile(`(?i)^PRAGMA\s+(TABLE_INFO|INDEX_LIST|INDEX_INFO|FOREIGN_KEY_LIST|DATABASE_LIST|INTEGRITY_CHECK)\b`)
)

// ValidateReadOnlySQL asserts that a SQL string contains only read-only statements.
func ValidateReadOnlySQL(query string) error {
	stripped := commentLineRE.ReplaceAllString(query, " ")
	stripped = commentBlockRE.ReplaceAllString(stripped, " ")
	cleaned := strings.TrimSpace(stripped)

	if cleaned == "" {
		return errors.New("dataslot: empty SQL query")
	}

	// Semicolon-separated statements: each must be read-only
	statements := strings.Split(cleaned, ";")
	for _, stmt := range statements {
		stmt = strings.TrimSpace(stmt)
		if stmt == "" {
			continue
		}

		if forbiddenSQLRE.MatchString(stmt) {
			return ErrMutationRefused
		}

		if !allowedHeadRE.MatchString(stmt) {
			return fmt.Errorf("%w: statement does not start with an approved read-only keyword", ErrMutationRefused)
		}

		if strings.HasPrefix(strings.ToUpper(stmt), "PRAGMA") {
			if !safePragmaRE.MatchString(stmt) {
				return fmt.Errorf("%w: only read-only schema PRAGMAs are allowed", ErrMutationRefused)
			}
		}
	}

	return nil
}

// SQLiteDriver implements SchemaReflector and QueryRunner for local SQLite database files.
type SQLiteDriver struct {
	RunnerExecutable string
	RunnerKind       string // "sqlite3" or "python"
}

// NewSQLiteDriver discovers an available SQLite query runner on the local system.
func NewSQLiteDriver() (*SQLiteDriver, error) {
	if p, err := exec.LookPath("sqlite3"); err == nil {
		return &SQLiteDriver{RunnerExecutable: p, RunnerKind: "sqlite3"}, nil
	}
	if p, err := exec.LookPath("python3"); err == nil {
		return &SQLiteDriver{RunnerExecutable: p, RunnerKind: "python"}, nil
	}
	if p, err := exec.LookPath("python"); err == nil {
		return &SQLiteDriver{RunnerExecutable: p, RunnerKind: "python"}, nil
	}
	return nil, errors.New("dataslot: no sqlite3 or python interpreter found on PATH")
}

// RowMap converts a row slice at rowIndex into a column->value map using result.Columns.
func (r *QueryResult) RowMap(rowIndex int) map[string]any {
	if r == nil || rowIndex < 0 || rowIndex >= len(r.Rows) {
		return nil
	}
	m := make(map[string]any, len(r.Columns))
	for i, col := range r.Columns {
		if i < len(r.Rows[rowIndex]) {
			m[col] = r.Rows[rowIndex][i]
		}
	}
	return m
}

// ReflectSchema inspects tables, columns, and foreign keys from the target SQLite database.
func (d *SQLiteDriver) ReflectSchema(ctx context.Context, target string) (*DatabaseSchema, error) {
	schema := &DatabaseSchema{
		Family:   FamilySQLite,
		Database: target,
		Tables:   nil,
	}

	// 1. Fetch table names from sqlite_master
	tableListQuery := "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
	res, err := d.ExecuteQuery(ctx, target, tableListQuery, &QueryOptions{MaxRows: 500})
	if err != nil {
		return nil, fmt.Errorf("dataslot: failed to list tables: %w", err)
	}

	var tableNames []string
	for i := 0; i < res.RowCount; i++ {
		row := res.RowMap(i)
		if name, ok := row["name"].(string); ok && name != "" {
			tableNames = append(tableNames, name)
		}
	}

	// 2. For each table, fetch columns and foreign keys
	for _, tbl := range tableNames {
		tableInfoQuery := fmt.Sprintf("PRAGMA table_info(%s);", tbl)
		colRes, err := d.ExecuteQuery(ctx, target, tableInfoQuery, &QueryOptions{MaxRows: 500})
		if err != nil {
			return nil, fmt.Errorf("dataslot: table_info failed for %s: %w", tbl, err)
		}

		var cols []ColumnSchema
		var pks []string

		for i := 0; i < colRes.RowCount; i++ {
			row := colRes.RowMap(i)
			colName, _ := row["name"].(string)
			colType, _ := row["type"].(string)
			notNull := false
			switch v := row["notnull"].(type) {
			case float64:
				notNull = v == 1
			case int64:
				notNull = v == 1
			case int:
				notNull = v == 1
			case string:
				notNull = v == "1"
			}
			isPK := false
			switch v := row["pk"].(type) {
			case float64:
				isPK = v >= 1
			case int64:
				isPK = v >= 1
			case int:
				isPK = v >= 1
			case string:
				isPK = v == "1"
			}

			cols = append(cols, ColumnSchema{
				Name:       colName,
				Type:       colType,
				Nullable:   !notNull,
				PrimaryKey: isPK,
			})
			if isPK {
				pks = append(pks, colName)
			}
		}

		// Foreign keys
		fkQuery := fmt.Sprintf("PRAGMA foreign_key_list(%s);", tbl)
		fkRes, _ := d.ExecuteQuery(ctx, target, fkQuery, &QueryOptions{MaxRows: 100})
		var fks []ForeignKeySchema
		if fkRes != nil {
			for i := 0; i < fkRes.RowCount; i++ {
				row := fkRes.RowMap(i)
				refTable, _ := row["table"].(string)
				fromCol, _ := row["from"].(string)
				toCol, _ := row["to"].(string)
				if refTable != "" && fromCol != "" {
					fks = append(fks, ForeignKeySchema{
						Column:           fromCol,
						ReferencedTable:  refTable,
						ReferencedColumn: toCol,
					})
				}
			}
		}

		schema.Tables = append(schema.Tables, TableSchema{
			Name:        tbl,
			Columns:     cols,
			PrimaryKeys: pks,
			ForeignKeys: fks,
		})
	}

	return schema, nil
}

// ExecuteQuery runs a read-only query bounded by row and byte limits.
func (d *SQLiteDriver) ExecuteQuery(ctx context.Context, target, sql string, opts *QueryOptions) (*QueryResult, error) {
	norm := opts.normalize()

	// Step 1: Validate read-only SQL
	if err := ValidateReadOnlySQL(sql); err != nil {
		return nil, err
	}

	// Enforce context deadline
	child, cancel := context.WithTimeout(ctx, norm.Timeout)
	defer cancel()

	start := time.Now()
	var rawRows []map[string]any
	var err error

	if d.RunnerKind == "python" {
		rawRows, err = d.execWithPython(child, target, sql, norm.MaxRows+1)
	} else {
		rawRows, err = d.execWithCLI(child, target, sql, norm.MaxRows+1)
	}

	duration := time.Since(start)

	if err != nil {
		if child.Err() == context.DeadlineExceeded {
			return nil, ErrTimeout
		}
		return nil, fmt.Errorf("%w: %v", ErrExecutionFailed, err)
	}

	// Extract columns and normalize rows
	result := &QueryResult{
		Columns:    nil,
		Rows:       nil,
		RowCount:   0,
		TotalRows:  len(rawRows),
		Truncated:  false,
		DurationMs: float64(duration.Microseconds()) / 1000.0,
	}

	if len(rawRows) > norm.MaxRows {
		result.Truncated = true
		rawRows = rawRows[:norm.MaxRows]
	}

	if len(rawRows) > 0 {
		// Collect columns from first row deterministically
		first := rawRows[0]
		colSet := make([]string, 0, len(first))
		for col := range first {
			colSet = append(colSet, col)
		}
		result.Columns = colSet

		byteCount := 0
		for _, rowMap := range rawRows {
			rowValues := make([]any, len(colSet))
			for i, col := range colSet {
				val := rowMap[col]
				rowValues[i] = val
				if s, ok := val.(string); ok {
					byteCount += len(s)
				} else {
					byteCount += 8
				}
			}
			if byteCount > norm.MaxBytes {
				result.Truncated = true
				break
			}
			result.Rows = append(result.Rows, rowValues)
		}
	}

	result.RowCount = len(result.Rows)
	return result, nil
}

func (d *SQLiteDriver) execWithPython(ctx context.Context, target, sql string, limit int) ([]map[string]any, error) {
	script := `import json, os, sqlite3, sys
path = sys.argv[1]
query = sys.argv[2]
limit = int(sys.argv[3])

if not os.path.exists(path):
    sys.exit(2)

uri = 'file:' + path.replace('\\', '/') + '?mode=ro'
conn = sqlite3.connect(uri, uri=True, timeout=5.0)
conn.execute('PRAGMA query_only = ON;')
cursor = conn.cursor()

try:
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    rows = []
    for r in cursor.fetchmany(limit):
        row_dict = {}
        for i, val in enumerate(r):
            row_dict[columns[i]] = val
        rows.append(row_dict)
    print(json.dumps(rows))
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(1)
`
	cmd := exec.CommandContext(ctx, d.RunnerExecutable, "-c", script, target, sql, strconv.Itoa(limit))
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	err := cmd.Run()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 2 {
			return nil, ErrTargetNotFound
		}
		return nil, errors.New(strings.TrimSpace(errBuf.String()))
	}

	var rows []map[string]any
	if err := json.Unmarshal(outBuf.Bytes(), &rows); err != nil {
		return nil, fmt.Errorf("failed to decode JSON query output: %w", err)
	}
	return rows, nil
}

func (d *SQLiteDriver) execWithCLI(ctx context.Context, target, sql string, limit int) ([]map[string]any, error) {
	boundedQuery := fmt.Sprintf("PRAGMA query_only=ON; %s;", sql)
	cmd := exec.CommandContext(ctx, d.RunnerExecutable, "-readonly", "-json", target, boundedQuery)
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	err := cmd.Run()
	if err != nil {
		return nil, errors.New(strings.TrimSpace(errBuf.String()))
	}

	if strings.TrimSpace(outBuf.String()) == "" {
		return nil, nil
	}

	var rows []map[string]any
	if err := json.Unmarshal(outBuf.Bytes(), &rows); err != nil {
		return nil, fmt.Errorf("failed to decode sqlite3 JSON output: %w", err)
	}
	if len(rows) > limit {
		rows = rows[:limit]
	}
	return rows, nil
}
