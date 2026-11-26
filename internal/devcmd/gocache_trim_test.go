package devcmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func writeCacheFile(t *testing.T, dir, name string, size int, age time.Duration) {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	mtime := time.Now().Add(-age)
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatal(err)
	}
}

func TestGoCacheTrimRemovesOldestFirstAndKeepsNewest(t *testing.T) {
	dir := t.TempDir()
	writeCacheFile(t, dir, "00/oldest", 400, 4*time.Hour)
	writeCacheFile(t, dir, "01/middle", 300, 3*time.Hour)
	writeCacheFile(t, dir, "02/newer", 200, 2*time.Hour)
	writeCacheFile(t, dir, "03/newest", 100, time.Hour)

	var stdout, stderr bytes.Buffer
	if code := RunGoCacheTrim(&stdout, &stderr, []string{dir, "350"}); code != 0 {
		t.Fatalf("exit %d, stderr %q", code, stderr.String())
	}
	_, total, err := scanGoCache(dir)
	if err != nil {
		t.Fatal(err)
	}
	if total > 350 {
		t.Fatalf("cache %d bytes, want at most 350", total)
	}
	for _, kept := range []string{"02/newer", "03/newest"} {
		if _, err := os.Stat(filepath.Join(dir, kept)); err != nil {
			t.Fatalf("newest entry %s was removed: %v", kept, err)
		}
	}
	for _, gone := range []string{"00/oldest", "01/middle"} {
		if _, err := os.Stat(filepath.Join(dir, gone)); !os.IsNotExist(err) {
			t.Fatalf("oldest entry %s survived the trim: %v", gone, err)
		}
	}
}

func TestGoCacheTrimLeavesCacheAtOrUnderCeilingUntouched(t *testing.T) {
	dir := t.TempDir()
	writeCacheFile(t, dir, "a", 100, time.Hour)
	writeCacheFile(t, dir, "b", 100, 2*time.Hour)

	var stdout, stderr bytes.Buffer
	if code := RunGoCacheTrim(&stdout, &stderr, []string{dir, strconv.Itoa(200)}); code != 0 {
		t.Fatalf("exit %d, stderr %q", code, stderr.String())
	}
	for _, name := range []string{"a", "b"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("%s removed though the cache was at the ceiling: %v", name, err)
		}
	}
}

func TestGoCacheTrimMissingDirIsNoop(t *testing.T) {
	var stdout, stderr bytes.Buffer
	missing := filepath.Join(t.TempDir(), "absent")
	if code := RunGoCacheTrim(&stdout, &stderr, []string{missing, "1"}); code != 0 {
		t.Fatalf("exit %d, stderr %q", code, stderr.String())
	}
}

func TestGoCacheTrimRejectsBadArguments(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := RunGoCacheTrim(&stdout, &stderr, []string{t.TempDir()}); code != 2 {
		t.Fatalf("missing ceiling: exit %d, want 2", code)
	}
	if code := RunGoCacheTrim(&stdout, &stderr, []string{t.TempDir(), "-5"}); code != 2 {
		t.Fatalf("negative ceiling: exit %d, want 2", code)
	}
}
