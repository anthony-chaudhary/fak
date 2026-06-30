package selfinstall

import (
	"context"
	"errors"
	"fmt"
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
// rename the existing target ASIDE first, then move the new one in; a concurrent reader sees
// either the intact old or the intact new binary, never a partial file. The renamed-aside copy
// is best-effort removed. If a prior aside file is still held by a stale process, we choose a
// unique aside name rather than letting one locked dst.old wedge every future self-update.
func OSSwap(src, dst string) error {
	if runtime.GOOS != "windows" {
		return os.Rename(src, dst)
	}
	_ = os.Remove(dst + ".old") // clear the conventional aside when no stale handle holds it
	old := windowsSwapAsidePath(dst, os.Getpid(), pathExists)
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

func windowsSwapAsidePath(dst string, pid int, exists func(string) bool) string {
	base := dst + ".old"
	if !exists(base) {
		return base
	}
	for i := 0; i < 1000; i++ {
		candidate := fmt.Sprintf("%s.%d.%d", base, pid, i)
		if !exists(candidate) {
			return candidate
		}
	}
	return fmt.Sprintf("%s.%d.overflow", base, pid)
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
