package boundedlog

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func appendLine(t *testing.T, path, line string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString(line + "\n"); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=50ms
func TestRotateIfOverNoopCases(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "log.jsonl")
	if rotated, err := RotateIfOver(p, 10); rotated || err != nil {
		t.Fatalf("missing file: rotated=%v err=%v", rotated, err)
	}
	appendLine(t, p, "0123456789") // 11 bytes
	if rotated, err := RotateIfOver(p, 0); rotated || err != nil {
		t.Fatalf("zero cap: rotated=%v err=%v", rotated, err)
	}
	if rotated, err := RotateIfOver(p, 11); rotated || err != nil {
		t.Fatalf("at cap: rotated=%v err=%v", rotated, err)
	}
	if rotated, err := RotateIfOver(dir, 1); rotated || err != nil {
		t.Fatalf("directory: rotated=%v err=%v", rotated, err)
	}
	if got := Segments(p); !slices.Equal(got, []string{p}) {
		t.Fatalf("segments = %v", got)
	}
}

// fak-test:runtime fast est=50ms
func TestRotateIfOverKeepsOneGeneration(t *testing.T) {
	p := filepath.Join(t.TempDir(), "log.jsonl")
	appendLine(t, p, "first-generation-row")
	if rotated, err := RotateIfOver(p, 4); !rotated || err != nil {
		t.Fatalf("rotate 1: rotated=%v err=%v", rotated, err)
	}
	if _, err := os.Stat(p); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("active file should be gone after rotation: %v", err)
	}
	if got := Segments(p); !slices.Equal(got, []string{RotatedPath(p)}) {
		t.Fatalf("segments after rotate = %v", got)
	}
	appendLine(t, p, "second-generation-row")
	if rotated, err := RotateIfOver(p, 4); !rotated || err != nil {
		t.Fatalf("rotate 2: rotated=%v err=%v", rotated, err)
	}
	appendLine(t, p, "third-generation-row")
	if got := Segments(p); !slices.Equal(got, []string{RotatedPath(p), p}) {
		t.Fatalf("segments = %v, want [.1, active]", got)
	}
	b, err := ReadAll(p)
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	if strings.Contains(got, "first-generation-row") {
		t.Fatalf("only one old generation is kept; got %q", got)
	}
	if got != "second-generation-row\nthird-generation-row\n" {
		t.Fatalf("ReadAll = %q, want oldest-first concatenation", got)
	}
}

// fak-test:runtime fast est=50ms
func TestReadAllMissing(t *testing.T) {
	if _, err := ReadAll(filepath.Join(t.TempDir(), "absent.jsonl")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v, want ErrNotExist", err)
	}
}
