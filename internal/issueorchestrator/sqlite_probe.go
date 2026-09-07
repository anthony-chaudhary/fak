package issueorchestrator

import (
	"os"
	"path/filepath"
	"time"
)

// SQLiteProbeResult holds the measured latency and lock status of a SQLite database.
type SQLiteProbeResult struct {
	Path      string        `json:"path"`
	Latency   time.Duration `json:"latency"`
	Contended bool          `json:"contended"`
	Error     string        `json:"error,omitempty"`
}

// DefaultOpencodeDBPath returns the standard path to opencode.db.
func DefaultOpencodeDBPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".local", "share", "opencode", "opencode.db")
}

// ProbeSQLiteLockLatency tests lock latency by measuring the time to open and read
// the SQLite database file header. If the file does not exist, it reports zero latency.
func ProbeSQLiteLockLatency(path string) SQLiteProbeResult {
	if path == "" {
		path = DefaultOpencodeDBPath()
	}
	if path == "" {
		return SQLiteProbeResult{Contended: false}
	}

	start := time.Now()
	f, err := os.OpenFile(path, os.O_RDWR, 0644)
	latency := time.Since(start)

	if err != nil {
		if os.IsNotExist(err) {
			return SQLiteProbeResult{Path: path, Latency: 0, Contended: false}
		}
		return SQLiteProbeResult{
			Path:      path,
			Latency:   latency,
			Contended: true,
			Error:     err.Error(),
		}
	}
	defer f.Close()

	var header [100]byte
	startRead := time.Now()
	_, _ = f.Read(header[:])
	totalLatency := latency + time.Since(startRead)

	return SQLiteProbeResult{
		Path:      path,
		Latency:   totalLatency,
		Contended: totalLatency > 50*time.Millisecond,
	}
}
