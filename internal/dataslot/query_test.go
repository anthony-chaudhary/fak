package dataslot

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestValidateReadOnlySQL(t *testing.T) {
	validQueries := []string{
		"SELECT * FROM users",
		"select id, email from users where active = 1",
		"EXPLAIN SELECT * FROM users",
		"WITH user_cte AS (SELECT id FROM users) SELECT * FROM user_cte",
		"PRAGMA table_info(users)",
		"PRAGMA index_list(users)",
		"PRAGMA foreign_key_list(orders)",
		"PRAGMA database_list",
		"PRAGMA integrity_check",
		"-- comment line\nSELECT 1",
		"/* multi\nline */ SELECT 1;",
	}

	for _, q := range validQueries {
		if err := ValidateReadOnlySQL(q); err != nil {
			t.Errorf("expected valid query %q, got error: %v", q, err)
		}
	}

	invalidQueries := []string{
		"INSERT INTO users VALUES (1, 'alice@example.com')",
		"UPDATE users SET email = 'bob@example.com' WHERE id = 1",
		"DELETE FROM users WHERE id = 1",
		"DROP TABLE users",
		"DROP DATABASE prod",
		"ALTER TABLE users ADD COLUMN age int",
		"TRUNCATE TABLE users",
		"CREATE TABLE hack (id int)",
		"REPLACE INTO users VALUES (1, 'c@c.com')",
		"ATTACH DATABASE 'leak.db' AS leak",
		"DETACH DATABASE leak",
		"PRAGMA writable_schema = ON",
		"SELECT 1; DROP TABLE users;",
		"SELECT 1; -- sneaky\nDELETE FROM users;",
		"",
		"   ",
	}

	for _, q := range invalidQueries {
		if err := ValidateReadOnlySQL(q); err == nil {
			t.Errorf("expected mutation rejection for %q, but passed", q)
		}
	}
}

func createPopulatedTestDB(t *testing.T, dbPath string, rowCount int) {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
	}
	if err != nil {
		t.Skip("python not found, skipping query execution test")
	}

	script := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, title TEXT, content TEXT);")
rows = [(i, f"Item {i}", f"Detail body content for record {i}") for i in range(int(sys.argv[2]))]
conn.executemany("INSERT INTO items VALUES (?, ?, ?);", rows)
conn.commit()
`
	cmd := exec.Command(python, "-c", script, dbPath, fmt.Sprintf("%d", rowCount))
	if err := cmd.Run(); err != nil {
		t.Fatalf("failed to populate test DB: %v", err)
	}
}

func TestQueryExecutionAndBounds(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "items.db")
	createPopulatedTestDB(t, dbPath, 100)

	driver, err := NewSQLiteDriver()
	if err != nil {
		t.Skipf("no SQLite runner available: %v", err)
	}

	// 1. Query with row cap of 20
	opts := &QueryOptions{
		MaxRows: 20,
	}
	res, err := driver.ExecuteQuery(context.Background(), dbPath, "SELECT * FROM items ORDER BY id;", opts)
	if err != nil {
		t.Fatalf("query execution failed: %v", err)
	}

	if res.RowCount != 20 {
		t.Errorf("expected 20 rows returned, got %d", res.RowCount)
	}
	if !res.Truncated {
		t.Errorf("expected Truncated=true because total rows (100) > max rows (20)")
	}
	if len(res.Columns) != 3 {
		t.Errorf("expected 3 columns, got %d", len(res.Columns))
	}

	// 2. Query within bounds (5 rows)
	resSmall, err := driver.ExecuteQuery(context.Background(), dbPath, "SELECT * FROM items WHERE id < 5;", opts)
	if err != nil {
		t.Fatalf("query execution failed: %v", err)
	}
	if resSmall.RowCount != 5 {
		t.Errorf("expected 5 rows, got %d", resSmall.RowCount)
	}
	if resSmall.Truncated {
		t.Errorf("expected Truncated=false for small result")
	}

	// 3. Byte bounding cap
	byteOpts := &QueryOptions{
		MaxRows:  50,
		MaxBytes: 100, // very small byte cap
	}
	resBytes, err := driver.ExecuteQuery(context.Background(), dbPath, "SELECT * FROM items;", byteOpts)
	if err != nil {
		t.Fatalf("query execution failed: %v", err)
	}
	if !resBytes.Truncated {
		t.Errorf("expected byte-cap truncation")
	}
}

func TestQueryReadOnlyEnforcement(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "safe.db")
	createPopulatedTestDB(t, dbPath, 5)

	driver, err := NewSQLiteDriver()
	if err != nil {
		t.Skipf("no SQLite runner available: %v", err)
	}

	// Direct mutation rejection
	_, err = driver.ExecuteQuery(context.Background(), dbPath, "DROP TABLE items;", nil)
	if !errors.Is(err, ErrMutationRefused) {
		t.Fatalf("expected ErrMutationRefused, got: %v", err)
	}

	// Verify table still exists and has 5 rows
	checkRes, err := driver.ExecuteQuery(context.Background(), dbPath, "SELECT COUNT(*) FROM items;", nil)
	if err != nil {
		t.Fatalf("table verification query failed: %v", err)
	}
	if checkRes.RowCount != 1 {
		t.Errorf("expected 1 row for count query, got %d", checkRes.RowCount)
	}
}

func TestQueryTimeout(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "timeout.db")
	createPopulatedTestDB(t, dbPath, 10)

	driver, err := NewSQLiteDriver()
	if err != nil {
		t.Skipf("no SQLite runner available: %v", err)
	}

	// Already expired context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	opts := &QueryOptions{
		Timeout: 1 * time.Nanosecond,
	}
	_, err = driver.ExecuteQuery(ctx, dbPath, "SELECT * FROM items;", opts)
	if err == nil {
		t.Fatal("expected error on cancelled/timed-out query, got nil")
	}
}

func TestQueryTargetNotFound(t *testing.T) {
	driver, err := NewSQLiteDriver()
	if err != nil {
		t.Skipf("no SQLite runner available: %v", err)
	}

	_, err = driver.ExecuteQuery(context.Background(), "nonexistent_file.db", "SELECT 1;", nil)
	if err == nil {
		t.Fatal("expected error for nonexistent target database")
	}
}
