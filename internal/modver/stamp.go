package modver

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// stampLockWait bounds how long AppendGuarded polls for the ledger lock before
// giving up. Each stamper holds the lock only for one short append, so a couple
// of seconds covers a realistic multi-session pile-up on the shared trunk.
const stampLockWait = 5 * time.Second

// AppendGuarded appends already-rendered JSONL bytes to the module-versions
// ledger at path while holding an exclusive cross-process advisory lock on
// <path>.lock, so two agents stamping the shared tree at once cannot interleave
// a torn or duplicated row (#2473). flock.TryLock is non-blocking, so it polls
// until the lock is free or stampLockWait elapses. The lock fd is closed on
// return, which also releases the OS lock (and the OS reclaims it if this
// process dies mid-append). The parent directory is created if missing. lines
// should already be newline-terminated JSONL — see AppendLines.
func AppendGuarded(path string, lines []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	lockPath := path + ".lock"
	lf, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return fmt.Errorf("open modver ledger lock: %w", err)
	}
	defer lf.Close()

	deadline := time.Now().Add(stampLockWait)
	for {
		lerr := flock.TryLock(lf)
		if lerr == nil {
			break
		}
		if !errors.Is(lerr, flock.ErrLockBusy) {
			return fmt.Errorf("lock modver ledger: %w", lerr)
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("modver ledger lock busy after %s: %w", stampLockWait, flock.ErrLockBusy)
		}
		time.Sleep(25 * time.Millisecond)
	}
	defer func() { _ = flock.Unlock(lf) }()

	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(lines); err != nil {
		f.Close()
		return fmt.Errorf("write modver ledger: %w", err)
	}
	return f.Close()
}
