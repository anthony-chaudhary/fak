package dispatchconservation

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadTailLinesBoundsAndPreservesOrder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "progress.jsonl")
	var b strings.Builder
	for i := 0; i < 25; i++ {
		fmt.Fprintf(&b, "row-%02d\n", i)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0644); err != nil {
		t.Fatal(err)
	}
	got := ReadTailLines(path, 10)
	if len(got) != 10 || got[0] != "row-15" || got[9] != "row-24" {
		t.Fatalf("got=%v", got)
	}
}

// TestReadTailBytesWindowsTheEndAndFailsOpen pins the byte-bounded tail the no-commit
// refinement reads: it must return the END of a file larger than the window (the guard
// exit summary lives there, after megabytes of streamed turns), the WHOLE file when it
// is smaller, and "" — never a panic — for an artifact that is not there.
func TestReadTailBytesWindowsTheEndAndFailsOpen(t *testing.T) {
	dir := t.TempDir()
	big := filepath.Join(dir, "big.log")
	if err := os.WriteFile(big, []byte(strings.Repeat("x", 5000)+"TAILMARK"), 0644); err != nil {
		t.Fatal(err)
	}
	got := ReadTailBytes(big, 64)
	if len(got) != 64 || !strings.HasSuffix(got, "TAILMARK") {
		t.Errorf("ReadTailBytes(big,64) len=%d suffix-ok=%v", len(got), strings.HasSuffix(got, "TAILMARK"))
	}
	small := filepath.Join(dir, "small.log")
	if err := os.WriteFile(small, []byte("short"), 0644); err != nil {
		t.Fatal(err)
	}
	if got := ReadTailBytes(small, 4096); got != "short" {
		t.Errorf("a file smaller than the window must come back whole; got %q", got)
	}
	if got := ReadTailBytes(filepath.Join(dir, "absent.log"), 4096); got != "" {
		t.Errorf("a missing artifact must read as absent evidence, got %q", got)
	}
	if got := ReadTailBytes(small, 0); got != "" {
		t.Errorf("a non-positive window must read as absent evidence, got %q", got)
	}
}
