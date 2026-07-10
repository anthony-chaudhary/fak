package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsl"
)

// writeRSL builds an RSL file by appending the given (old, new) transitions on
// refs/heads/main via the real rsl.Append, so the CLI test drives genuinely
// sound (or, when a rewind is requested, genuinely non-ff) rows.
func writeRSL(t *testing.T, transitions [][2]string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "rsl.jsonl")
	for _, tr := range transitions {
		if _, err := rsl.Append(path, rsl.Row{Ref: "refs/heads/main", OldSHA: tr[0], NewSHA: tr[1]}); err != nil {
			t.Fatalf("rsl.Append(%v): %v", tr, err)
		}
	}
	return path
}

func TestRunAuditRSL_FastForwardOnlyExitsZero(t *testing.T) {
	path := writeRSL(t, [][2]string{{"A", "B"}, {"B", "C"}, {"C", "D"}})
	var stdout, stderr bytes.Buffer
	if code := runAuditRSL(&stdout, &stderr, path); code != 0 {
		t.Fatalf("fast-forward-only RSL should exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "fast-forward-only") {
		t.Fatalf("render should confirm fast-forward-only:\n%s", stdout.String())
	}
}

func TestRunAuditRSL_ForcePushExitsOneNamingRef(t *testing.T) {
	// A->B->C, then a force reset back to B and a move to X: the 3rd row's old_sha
	// (B) no longer continues the recorded head (C).
	path := writeRSL(t, [][2]string{{"A", "B"}, {"B", "C"}, {"B", "X"}})
	var stdout, stderr bytes.Buffer
	code := runAuditRSL(&stdout, &stderr, path)
	if code != 1 {
		t.Fatalf("a non-fast-forward gap should exit 1, got %d", code)
	}
	if !strings.Contains(stderr.String(), "refs/heads/main") {
		t.Fatalf("stderr should name the offending ref:\n%s", stderr.String())
	}
}

func TestRunAuditRSL_TamperedChainExitsOne(t *testing.T) {
	path := writeRSL(t, [][2]string{{"A", "B"}, {"B", "C"}})
	// Corrupt a byte in the middle of the file: the hash no longer recomputes.
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	corrupt := bytes.Replace(b, []byte(`"new_sha":"B"`), []byte(`"new_sha":"Z"`), 1)
	if bytes.Equal(corrupt, b) {
		t.Fatal("test setup: expected to corrupt the new_sha field")
	}
	if err := os.WriteFile(path, corrupt, 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runAuditRSL(&stdout, &stderr, path); code != 1 {
		t.Fatalf("a tampered chain should exit 1, got %d (stderr=%s)", code, stderr.String())
	}
}

func TestRunAuditRSL_MissingFileIsEmptySoundLog(t *testing.T) {
	// An absent file reads as the empty log (rsl.ReadRows treats not-exist as no
	// rows) — a trivially sound, fast-forward-only history, exit 0.
	var stdout, stderr bytes.Buffer
	path := filepath.Join(t.TempDir(), "does-not-exist.jsonl")
	if code := runAuditRSL(&stdout, &stderr, path); code != 0 {
		t.Fatalf("an empty log should exit 0, got %d (stderr=%s)", code, stderr.String())
	}
}
