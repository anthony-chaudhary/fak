package issueorchestrator

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// SQLiteProbeResult holds the measured latency and lock status of a SQLite database.
type SQLiteProbeResult struct {
	Path      string        `json:"path"`
	Latency   time.Duration `json:"latency"`
	Contended bool          `json:"contended"`
	Error     string        `json:"error,omitempty"`
}

// SQLiteLockProbeFn defines a mockable function signature for probing SQLite lock latency.
type SQLiteLockProbeFn func(dbPath string) (time.Duration, error)

var (
	sqliteProbeMu   sync.RWMutex
	sqliteProbeHook SQLiteLockProbeFn
)

// SetSQLiteLockProbe overrides the lock latency probe function for testing and returns a cleanup function.
func SetSQLiteLockProbe(fn SQLiteLockProbeFn) func() {
	sqliteProbeMu.Lock()
	sqliteProbeHook = fn
	sqliteProbeMu.Unlock()
	return func() {
		sqliteProbeMu.Lock()
		sqliteProbeHook = nil
		sqliteProbeMu.Unlock()
	}
}

// DefaultOpencodeDBPath returns the standard path to opencode.db.
func DefaultOpencodeDBPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".local", "share", "opencode", "opencode.db")
}

// ProbeSQLiteLockLatency tests lock latency by attempting a quick read transaction / lock
// check without blocking. If the file does not exist or SQLite is not present, it returns
// 0 latency and nil error (fail-open).
func ProbeSQLiteLockLatency(dbPath string) (time.Duration, error) {
	sqliteProbeMu.RLock()
	hook := sqliteProbeHook
	sqliteProbeMu.RUnlock()
	if hook != nil {
		return hook(dbPath)
	}

	if dbPath == "" {
		dbPath = DefaultOpencodeDBPath()
	}
	if dbPath == "" {
		return 0, nil
	}

	// Fail-open if file does not exist
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		return 0, nil
	}

	// Fail-open if SQLite is not present
	sqlitePath, err := exec.LookPath("sqlite3")
	if err != nil {
		return 0, nil
	}

	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
	defer cancel()

	// Non-blocking transaction check: busy_timeout=0 fails fast on contention
	cmd := exec.CommandContext(ctx, sqlitePath, dbPath, "PRAGMA busy_timeout=0; BEGIN DEFERRED; ROLLBACK;")
	cmdErr := cmd.Run()
	latency := time.Since(start)

	if cmdErr != nil {
		return latency, cmdErr
	}
	return latency, nil
}

// ProbeSQLiteLockLatencyDetailed returns an annotated probe result struct.
func ProbeSQLiteLockLatencyDetailed(dbPath string) SQLiteProbeResult {
	lat, err := ProbeSQLiteLockLatency(dbPath)
	errStr := ""
	if err != nil {
		errStr = err.Error()
	}
	return SQLiteProbeResult{
		Path:      dbPath,
		Latency:   lat,
		Contended: lat > 50*time.Millisecond,
		Error:     errStr,
	}
}
