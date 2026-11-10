package fsatomic

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

var errTransient = errors.New("transient")

// fak-test:runtime fast est=5ms lane=default
func TestRenameRetrying_RetriesTransientThenSucceeds(t *testing.T) {
	calls := 0
	rename := func(string, string) error {
		calls++
		if calls < 3 {
			return errTransient
		}
		return nil
	}
	isTransient := func(err error) bool { return errors.Is(err, errTransient) }
	if err := renameRetrying("a", "b", rename, isTransient, func(time.Duration) {}); err != nil {
		t.Fatalf("renameRetrying = %v, want nil after transient failures", err)
	}
	if calls != 3 {
		t.Fatalf("rename calls = %d, want 3", calls)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestRenameRetrying_StopsOnPermanentAndBoundsTransient(t *testing.T) {
	errPermanent := errors.New("permanent")
	calls := 0
	rename := func(string, string) error { calls++; return errPermanent }
	isTransient := func(err error) bool { return errors.Is(err, errTransient) }
	if err := renameRetrying("a", "b", rename, isTransient, func(time.Duration) {}); !errors.Is(err, errPermanent) || calls != 1 {
		t.Fatalf("permanent: err=%v calls=%d, want errPermanent after 1 call", err, calls)
	}

	calls = 0
	var slept time.Duration
	always := func(string, string) error { calls++; return errTransient }
	err := renameRetrying("a", "b", always, isTransient, func(d time.Duration) { slept += d })
	if !errors.Is(err, errTransient) || calls != renameAttempts {
		t.Fatalf("transient: err=%v calls=%d, want errTransient after %d calls", err, calls, renameAttempts)
	}
	if slept > time.Second {
		t.Fatalf("total backoff %v, want bounded under 1s", slept)
	}
}

// A reader holding the destination open (Go opens without FILE_SHARE_DELETE,
// as do AV scanners and the search indexer) makes the Windows replace fail with
// "Access is denied" until it closes; the writer waits it out.
// fak-test:runtime fast est=60ms lane=default
func TestWriteFileAtomic_WaitsOutReaderHoldingDestination(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("POSIX rename replaces an open destination")
	}
	path := filepath.Join(t.TempDir(), "ledger.json")
	if err := os.WriteFile(path, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	released := make(chan struct{})
	go func() {
		time.Sleep(30 * time.Millisecond)
		_ = f.Close()
		close(released)
	}()
	defer func() { <-released }()

	if err := WriteFileAtomic(path, []byte("new\n"), 0o644); err != nil {
		t.Fatalf("WriteFileAtomic with a briefly held destination = %v, want nil", err)
	}
	if got, _ := os.ReadFile(path); string(got) != "new\n" {
		t.Fatalf("content = %q, want new", got)
	}
}
