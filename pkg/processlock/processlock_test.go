package processlock

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestTryDistinguishesContentionFromHardFailures(t *testing.T) {
	dir := t.TempDir()
	release, err := Try(dir, "observer.lock")
	if err != nil {
		t.Fatalf("first Try: %v", err)
	}
	if _, err := Try(dir, "observer.lock"); !errors.Is(err, ErrBusy) {
		t.Fatalf("contended Try: want ErrBusy, got %v", err)
	}
	release()

	release, err = Try(dir, "observer.lock")
	if err != nil {
		t.Fatalf("Try after release: %v", err)
	}
	release()
	if _, err := os.Stat(filepath.Join(dir, "observer.lock")); err != nil {
		t.Fatalf("exact lock path missing: %v", err)
	}

	blocked := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(blocked, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Try(blocked, "observer.lock"); err == nil || errors.Is(err, ErrBusy) {
		t.Fatalf("hard path failure must remain non-busy, got %v", err)
	}
	if _, err := Try(dir, "../escape.lock"); err == nil || errors.Is(err, ErrBusy) {
		t.Fatalf("unsafe name must remain non-busy, got %v", err)
	}
}
