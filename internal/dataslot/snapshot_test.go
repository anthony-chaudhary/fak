package dataslot

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func createDatabaseWithRows(t *testing.T, dbPath string, rowCount int) {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
	}
	if err != nil {
		t.Skip("python not found, skipping snapshot integration test")
	}

	script := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);")
for i in range(int(sys.argv[2])):
    conn.execute("INSERT INTO users VALUES (?, ?);", (i, f"user-{i}"))
conn.commit()
conn.close()
`
	cmd := exec.Command(python, "-c", script, dbPath, string(rune('0'+rowCount)))
	if rowCount >= 10 {
		cmd = exec.Command(python, "-c", script, dbPath, "10")
	}
	if err := cmd.Run(); err != nil {
		t.Fatalf("failed creating sample sqlite database: %v", err)
	}
}

func countUsersInDB(t *testing.T, dbPath string) (int, error) {
	t.Helper()
	driver, err := NewSQLiteDriver()
	if err != nil {
		t.Skipf("no SQLite runner available: %v", err)
	}

	res, err := driver.ExecuteQuery(context.Background(), dbPath, "SELECT COUNT(*) FROM users;", nil)
	if err != nil {
		return 0, err
	}
	if len(res.Rows) == 0 || len(res.Rows[0]) == 0 {
		return 0, nil
	}
	switch v := res.Rows[0][0].(type) {
	case float64:
		return int(v), nil
	case int64:
		return int(v), nil
	case int:
		return v, nil
	default:
		return 0, nil
	}
}

func executeDestructiveMutation(t *testing.T, dbPath string) {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
	}
	if err != nil {
		t.Skip("python not found")
	}

	script := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("DROP TABLE users;")
conn.commit()
conn.close()
`
	cmd := exec.Command(python, "-c", script, dbPath)
	if err := cmd.Run(); err != nil {
		t.Fatalf("failed executing destructive SQL mutation: %v", err)
	}
}

func TestSnapshotRollbackParity(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "app.sqlite")
	createDatabaseWithRows(t, dbPath, 10)

	// Verify initial count is 10
	initialCount, err := countUsersInDB(t, dbPath)
	if err != nil {
		t.Fatalf("initial count failed: %v", err)
	}
	if initialCount != 10 {
		t.Fatalf("expected initial count 10, got %d", initialCount)
	}

	// Capture initial hash
	initialHash, _, err := computeFileHash(dbPath)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Take snapshot
	mgr := NewSnapshotManager(filepath.Join(dir, "snapshots"))
	snap, err := mgr.Snapshot(context.Background(), dbPath, "session-test-1", 1)
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if snap == nil {
		t.Fatal("expected non-nil snapshot")
	}

	// 2. Perform destructive mutation: DROP TABLE users
	executeDestructiveMutation(t, dbPath)

	// Verify table is gone
	_, err = countUsersInDB(t, dbPath)
	if err == nil {
		t.Fatal("expected error querying dropped table, got nil")
	}

	// 3. Rollback
	if err := mgr.Rollback(context.Background(), snap); err != nil {
		t.Fatalf("Rollback failed: %v", err)
	}

	// 4. Verify bit-for-bit parity
	restoredHash, _, err := computeFileHash(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if restoredHash != initialHash {
		t.Fatalf("checksum mismatch after rollback: got %s, want %s", restoredHash, initialHash)
	}

	// Verify data integrity and row count
	restoredCount, err := countUsersInDB(t, dbPath)
	if err != nil {
		t.Fatalf("restored db count failed: %v", err)
	}
	if restoredCount != 10 {
		t.Fatalf("expected restored count 10, got %d", restoredCount)
	}

	// 5. Reap
	if err := mgr.Reap(context.Background(), "session-test-1"); err != nil {
		t.Fatalf("Reap failed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "snapshots", "session-test-1")); !os.IsNotExist(err) {
		t.Errorf("expected snapshot directory to be removed after Reap")
	}
}

func TestSnapshotWithWALMode(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "wal_app.db")

	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
	}
	if err != nil {
		t.Skip("python not found")
	}

	// Create DB in WAL mode and leave uncheckpointed transactions
	script := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("PRAGMA journal_mode = WAL;")
conn.execute("CREATE TABLE metrics (key TEXT, val INT);")
conn.execute("INSERT INTO metrics VALUES ('cpu', 42);")
conn.commit()
conn.close()
`
	cmd := exec.Command(python, "-c", script, dbPath)
	if err := cmd.Run(); err != nil {
		t.Fatalf("failed setting up WAL database: %v", err)
	}

	mgr := NewSnapshotManager(filepath.Join(dir, "snaps"))
	snap, err := mgr.Snapshot(context.Background(), dbPath, "wal-session", 1)
	if err != nil {
		t.Fatalf("Snapshot in WAL mode failed: %v", err)
	}

	// Mutate WAL
	scriptMutate := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("DELETE FROM metrics;")
conn.commit()
conn.close()
`
	cmdMutate := exec.Command(python, "-c", scriptMutate, dbPath)
	if err := cmdMutate.Run(); err != nil {
		t.Fatalf("failed mutating WAL: %v", err)
	}

	// Rollback
	if err := mgr.Rollback(context.Background(), snap); err != nil {
		t.Fatalf("Rollback failed: %v", err)
	}

	// Check row restored
	scriptCheck := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
row = conn.execute("SELECT val FROM metrics WHERE key='cpu';").fetchone()
conn.close()
assert row and row[0] == 42
`
	cmdCheck := exec.Command(python, "-c", scriptCheck, dbPath)
	if err := cmdCheck.Run(); err != nil {
		t.Fatalf("verification after WAL rollback failed: %v", err)
	}
}

func TestSnapshotNotFound(t *testing.T) {
	mgr := NewSnapshotManager(t.TempDir())
	if err := mgr.Rollback(context.Background(), nil); err != ErrSnapshotNotFound {
		t.Fatalf("expected ErrSnapshotNotFound on nil snapshot, got: %v", err)
	}
}
