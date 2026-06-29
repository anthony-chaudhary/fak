package selfinstall

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// ErrBusy is returned by TrySingleFlight when another self-update already holds the lock.
var ErrBusy = errors.New("selfinstall: another self-update is in progress")

// TrySingleFlight takes a NON-BLOCKING advisory lock so at most one self-update builds at a
// time on a host. A second concurrent invocation returns ErrBusy immediately instead of
// stacking another expensive origin checkout + build — critical on a saturated box where the
// scheduled tick could otherwise pile builds on top of a slow one. The returned release frees
// the lock; the OS also drops it if the process exits. dir is where the lockfile lives (""
// => OS temp); the lock file is named fak-selfupdate.lock there.
func TrySingleFlight(dir string) (release func(), err error) {
	if dir == "" {
		dir = os.TempDir()
	}
	path := filepath.Join(dir, "fak-selfupdate.lock")
	f, oerr := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if oerr != nil {
		return nil, oerr
	}
	if lerr := flock.TryLock(f); lerr != nil {
		f.Close()
		if errors.Is(lerr, flock.ErrLockBusy) {
			return nil, ErrBusy
		}
		return nil, lerr
	}
	return func() { _ = flock.Unlock(f); _ = f.Close() }, nil
}

// RealRunner runs the command for real, merging stdout+stderr, and reports ok=false on any
// non-zero exit or exec failure (so a failed gate is a clean ok=false, not a panic).
func RealRunner(ctx context.Context, dir, name string, args ...string) (string, bool) {
	cmd := exec.CommandContext(ctx, name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// OSSwap atomically replaces dst with src. On unix os.Rename over an existing (even
// running) binary is atomic. On Windows a mapped .exe cannot be overwritten in place, so we
// rename the existing target ASIDE (dst.old) first, then move the new one in; a concurrent
// reader sees either the intact old or the intact new binary, never a partial file. The
// renamed-aside copy is best-effort removed (Windows lets you rename a running .exe but not
// delete it until its handles close — leaving a harmless dst.old is acceptable).
func OSSwap(src, dst string) error {
	if runtime.GOOS != "windows" {
		return os.Rename(src, dst)
	}
	old := dst + ".old"
	_ = os.Remove(old) // clear any prior leftover; ignore if a stale handle still holds it
	if _, err := os.Stat(dst); err == nil {
		if err := os.Rename(dst, old); err != nil {
			return err
		}
	}
	if err := os.Rename(src, dst); err != nil {
		// Roll back: put the original binary back so the fleet is never left without one.
		_ = os.Rename(old, dst)
		return err
	}
	_ = os.Remove(old) // best-effort; a held handle just leaves dst.old until it closes
	return nil
}
