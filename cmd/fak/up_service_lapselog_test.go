package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/boundedlog"
)

// TestUpServiceLapseLogRotationReadsSpanBothSegments pins the waker-log bound the
// darwin spawner applies before each open: an over-cap log rotates to .1, the next
// waker appends to a fresh active file, and a Segments read sees both, oldest first.
//
// fak-test:runtime fast est=50ms
func TestUpServiceLapseLogRotationReadsSpanBothSegments(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "com.fak.up.lapse.log")
	appendTo := func(line string) {
		t.Helper()
		f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if _, err := f.WriteString(line + "\n"); err != nil {
			t.Fatal(err)
		}
	}
	rotateUpServiceLapseLog(logPath, 1) // missing log: no-op
	appendTo("older-waker")
	rotateUpServiceLapseLog(logPath, 1)
	appendTo("newer-waker")

	if segs := boundedlog.Segments(logPath); len(segs) != 2 {
		t.Fatalf("segments = %v, want [.1, active]", segs)
	}
	raw, err := boundedlog.ReadAll(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(raw); got != "older-waker\nnewer-waker\n" {
		t.Fatalf("read must span .1 then active, oldest first: %q", got)
	}
	if upServiceLapseLogMaxBytes != 16<<20 {
		t.Fatalf("lapse log cap = %d, want 16 MiB", upServiceLapseLogMaxBytes)
	}
}
