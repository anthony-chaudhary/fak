package journal

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeJSONL writes each row as one JSON line, plus an optional raw trailing
// fragment to simulate a torn final append (a crash mid-write).
func writeJSONL(t *testing.T, path string, rows []Row, tornTail string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer f.Close()
	for _, r := range rows {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if _, err := f.Write(append(b, '\n')); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	if tornTail != "" {
		if _, err := f.WriteString(tornTail); err != nil {
			t.Fatalf("write torn tail: %v", err)
		}
	}
}

func TestReadRowsRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	want := []Row{
		{Seq: 1, Kind: "DECIDE", Tool: "search_kb", Verdict: "ALLOW"},
		{Seq: 2, Kind: "DENY", Tool: "rm_rf", Verdict: "DENY", Reason: "POLICY_BLOCK", By: "floor", Witness: "rm -rf /"},
		{Seq: 3, Kind: "DENY", Tool: "write_file", Verdict: "DENY", Reason: "DEFAULT_DENY"},
	}
	writeJSONL(t, path, want, "")

	got, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d rows, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Seq != want[i].Seq || got[i].Kind != want[i].Kind || got[i].Verdict != want[i].Verdict || got[i].Reason != want[i].Reason {
			t.Fatalf("row %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestReadRowsMissingFileIsEmpty(t *testing.T) {
	got, err := ReadRows(filepath.Join(t.TempDir(), "does-not-exist.jsonl"))
	if err != nil {
		t.Fatalf("ReadRows on a missing file errored: %v (want empty, no error)", err)
	}
	if len(got) != 0 {
		t.Fatalf("ReadRows on a missing file = %d rows, want 0", len(got))
	}
}

func TestReadRowsToleratesTornTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	good := []Row{
		{Seq: 1, Kind: "DENY", Verdict: "DENY", Reason: "POLICY_BLOCK"},
		{Seq: 2, Kind: "DENY", Verdict: "DENY", Reason: "DEFAULT_DENY"},
	}
	// A crash mid-append leaves a half-written final line with no newline.
	writeJSONL(t, path, good, `{"seq":3,"kind":"DEN`)

	got, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d rows, want 2 (torn final line must be skipped)", len(got))
	}
	if got[1].Seq != 2 {
		t.Fatalf("last good row seq = %d, want 2", got[1].Seq)
	}
}

func TestReadRowsEmptyFileIsEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.jsonl")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatalf("write empty: %v", err)
	}
	got, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows on an empty file errored: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("ReadRows on an empty file = %d rows, want 0", len(got))
	}
}
