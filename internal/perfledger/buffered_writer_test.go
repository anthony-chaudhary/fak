package perfledger

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms lane=default
func TestBufferedWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "buffered.jsonl")
	w, err := NewBufferedJSONLWriter(path, BufferedWriterConfig{
		BatchSize:          3,
		FlushInterval:      15 * time.Millisecond,
		MaxInFlightFlushes: 1,
		WireOnlyFields:     []string{"wire_secret", "raw_auth"},
	})
	if err != nil {
		t.Fatalf("NewBufferedJSONLWriter: %v", err)
	}

	for i := 0; i < 4; i++ {
		if err := w.Write(map[string]any{
			"id":          i,
			"latency_ms":  float64(10 + i),
			"wire_secret": "drop-me",
			"raw_auth":    "bearer-xyz",
		}); err != nil {
			t.Fatalf("Write(%d): %v", i, err)
		}
	}

	// Wait for periodic flush of the 4th trailing record before Close.
	deadline := time.Now().Add(time.Second)
	for w.FlushedCount() < 4 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if w.FlushedCount() != 4 {
		t.Fatalf("periodic flush did not persist all 4 records, got %d", w.FlushedCount())
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	count := 0
	for scanner.Scan() {
		var row map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatalf("unmarshal line %d: %v", count, err)
		}
		if _, has := row["wire_secret"]; has {
			t.Fatalf("wire_secret leaked into JSONL: %v", row)
		}
		if _, has := row["raw_auth"]; has {
			t.Fatalf("raw_auth leaked into JSONL: %v", row)
		}
		count++
	}
	if count != 4 {
		t.Fatalf("read %d lines, want 4", count)
	}
}
