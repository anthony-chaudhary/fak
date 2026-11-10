package boundedlog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func appendLine(t *testing.T, path, line string, maxBytes int64) {
	t.Helper()
	if _, err := RotateIfOver(path, maxBytes); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(line + "\n"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func readSegments(t *testing.T, path string) []string {
	t.Helper()
	var lines []string
	for _, p := range Segments(path) {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		lines = append(lines, strings.Fields(string(raw))...)
	}
	return lines
}

// fak-test:runtime fast est=1s lane=default
func TestRotateIfOverRotatesPastCapAndKeepsAppending(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	const maxBytes = 64
	line := strings.Repeat("x", 15) // 16 bytes with newline
	for i := 0; i < 20; i++ {
		appendLine(t, path, line, maxBytes)
		for _, p := range []string{path, RotatedPath(path)} {
			if info, err := os.Stat(p); err == nil && info.Size() > maxBytes {
				t.Fatalf("%s grew past cap: %d", p, info.Size())
			}
		}
	}
	if _, err := os.Stat(RotatedPath(path)); err != nil {
		t.Fatalf("no rotated generation after exceeding cap: %v", err)
	}
	got := readSegments(t, path)
	if len(got) == 0 || len(got) > 2*maxBytes/16 {
		t.Fatalf("retained %d lines, want 1..%d", len(got), 2*maxBytes/16)
	}
	// The newest line always survives in the active file.
	raw, err := os.ReadFile(path)
	if err != nil || !strings.HasSuffix(string(raw), line+"\n") {
		t.Fatalf("active file lost the newest append: %q %v", raw, err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRotateIfOverNoopCases(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ledger.jsonl")
	if rotated, err := RotateIfOver(path, 10); rotated || err != nil {
		t.Fatalf("missing ledger: rotated=%v err=%v", rotated, err)
	}
	if err := os.WriteFile(path, []byte(strings.Repeat("y", 100)), 0o644); err != nil {
		t.Fatal(err)
	}
	if rotated, err := RotateIfOver(path, 0); rotated || err != nil {
		t.Fatalf("disabled cap rotated: %v %v", rotated, err)
	}
	if rotated, err := RotateIfOver(path, 1000); rotated || err != nil {
		t.Fatalf("under cap rotated: %v %v", rotated, err)
	}
	if got := Segments(path); len(got) != 1 || got[0] != path {
		t.Fatalf("segments = %v", got)
	}
	if rotated, err := RotateIfOver(path, 100); !rotated || err != nil {
		t.Fatalf("at cap did not rotate: %v %v", rotated, err)
	}
	if got := Segments(path); len(got) != 1 || got[0] != RotatedPath(path) {
		t.Fatalf("segments after rotate = %v", got)
	}
}
