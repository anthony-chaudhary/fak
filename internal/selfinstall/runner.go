package selfinstall

import (
	"context"
	"os"
	"os/exec"
	"runtime"
)

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
