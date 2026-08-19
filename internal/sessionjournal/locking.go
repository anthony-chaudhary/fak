package sessionjournal

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

const appendLockWait = 10 * time.Second

// withJournalLock serializes writers and maintenance across processes. The lock
// lives beside the journal so every user targeting the host-global path contends
// on the same kernel object.
func withJournalLock(path string, fn func() error) error {
	lockPath := path + ".lock"
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o755); err != nil {
		return fmt.Errorf("sessionjournal: create lock directory: %w", err)
	}
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("sessionjournal: open lock: %w", err)
	}
	defer f.Close()

	deadline := time.Now().Add(appendLockWait)
	for {
		err = flock.TryLock(f)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return fmt.Errorf("sessionjournal: acquire lock: %w", err)
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("sessionjournal: acquire lock: %w", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	defer flock.Unlock(f) // the descriptor close remains the final release fence
	return fn()
}
