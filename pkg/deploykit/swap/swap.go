// Package swap is the public activation facade a fak deployable uses to replace its installed
// files: one atomic swap, one all-or-nothing multi-target transaction, one swap-aside reaper,
// one single-flight lock, and one content-addressed rollback slot name.
//
// It is a facade, not a second implementation. The swap, transaction, and reaper primitives
// live in internal/selfinstall, where `fak self-update` already relies on them, and the lock is
// internal/flock. Code outside this module cannot import internal/*, so every other deployable
// grew its own copy of the swap and its own backup name. This package re-exports those
// primitives for generic target paths and drops the self-update-specific names: the lock file
// is named per deployable, and the rollback slot is `<name>.deploy-prior-<sha12>` (slots.go).
//
// `fak self-update` keeps its own `.self-update-prior` slot and `fak-selfupdate.lock`; nothing
// here changes them.
package swap

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

// Copy declares one source-to-target activation in a Transaction. The source is copied, not
// consumed, so it survives both success and rollback.
type Copy = selfinstall.Copy

// Result is the outcome of a Transaction: exactly one of Updated, RolledBack, or
// RollbackFailed. Callers type-switch on it.
type Result = selfinstall.TransactionResult

// Updated means every target was activated (Changed counts the targets whose bytes differed).
type Updated = selfinstall.Updated

// RolledBack means activation failed and every changed target was restored byte-for-byte.
type RolledBack = selfinstall.RolledBack

// RollbackFailed means activation failed and at least one changed target could not be
// restored. Snapshots names the preserved pre-state copies for manual recovery.
type RollbackFailed = selfinstall.RollbackFailed

// Swap atomically replaces dst with src, consuming src. On Unix it is a rename. On Windows the
// existing dst, which may be a running .exe, is renamed aside first and the new file is moved
// in; a concurrent reader sees either the intact old or the intact new file, never a partial
// one. An aside still held open by a running process is left as "<dst>.old.<pid>.<i>" for
// ReapAsides.
func Swap(src, dst string) error {
	return selfinstall.OSSwap(src, dst)
}

// Transaction activates every copy or none. It stages every candidate and snapshots every
// target before the first swap, activates targets in lexical path order, and on any failure
// restores every target it already changed. Targets whose bytes already equal their source are
// skipped. Every source and target must exist.
func Transaction(copies []Copy) Result {
	return transaction(copies, selfinstall.OSSwap)
}

// transaction is the swapper seam: tests inject a failing swapper here.
func transaction(copies []Copy, swap selfinstall.Swapper) Result {
	return selfinstall.RunTransaction(copies, swap)
}

// ReapAsides deletes the "<target>.old.<pid>.<i>" swap-asides that Swap leaves on Windows
// when the replaced binary was still running, but only those whose owning process has exited
// and is not this process. The live target, its plain ".old", and every rollback slot are never
// touched. It returns the paths it removed.
func ReapAsides(target string) []string {
	return selfinstall.ReapStaleAsides(target, os.Getpid(), processalive.Check)
}

// ErrBusy is returned (wrapped with the lock path) by Lock when another holder owns the lock.
var ErrBusy = errors.New("deploykit/swap: another deploy holds the lock")

// Lock takes a non-blocking single-flight lock on "<dir>/<name>.deploy.lock", so at most one
// deploy of a deployable runs at a time among the processes that share dir. A second concurrent
// Lock on the same name, from this process or another, returns an error wrapping ErrBusy
// immediately. Exclusion is only as wide as dir is shared: pass a directory every deployer of
// the deployable uses, typically its install directory. dir "" means the OS temp dir, which
// differs per user and per service environment. name must be a plain file-name component. The
// returned release frees the lock and is safe to call more than once; the OS also drops the
// lock when the process exits.
func Lock(dir, name string) (release func(), err error) {
	if err := validLockName(name); err != nil {
		return nil, err
	}
	if dir == "" {
		dir = os.TempDir()
	}
	path := filepath.Join(dir, name+".deploy.lock")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	if lerr := flock.TryLock(f); lerr != nil {
		f.Close()
		if errors.Is(lerr, flock.ErrLockBusy) {
			return nil, fmt.Errorf("%w: %s", ErrBusy, path)
		}
		return nil, lerr
	}
	var once sync.Once
	return func() {
		once.Do(func() { _ = flock.Unlock(f); _ = f.Close() })
	}, nil
}

// validLockName accepts only a plain file-name component that is valid on every OS: no path
// separators, no ':' (which on Windows opens an alternate data stream on another file), and no
// other character Windows forbids in a file name.
func validLockName(name string) error {
	invalid := strings.TrimSpace(name) == "" || name == "." || name == ".." ||
		strings.ContainsAny(name, `/\:<>"|?*`) || filepath.Base(name) != name
	for _, c := range name {
		invalid = invalid || c < 0x20
	}
	if invalid {
		return fmt.Errorf("deploykit/swap: lock name %q must be a plain file-name component", name)
	}
	return nil
}
