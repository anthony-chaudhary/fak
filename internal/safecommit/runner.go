package safecommit

import (
	"context"
	"errors"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

// realRunner is the default Runner: it runs the real git binary. It mirrors
// witness.gitRunner's contract — a non-zero git exit is returned in code (not err); err
// signals git could not be EXECUTED at all — with one deliberate difference: it MERGES
// stderr into the returned stdout. The executor needs a hook's refusal / a push rejection
// message to surface in Result.Detail, which witness (Stderr = nil) discards.
func realRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	var buf strings.Builder
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	if err == nil {
		return buf.String(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return buf.String(), ee.ExitCode(), nil // git ran, returned non-zero
	}
	return "", -1, err // git could not be executed
}

// realLock is the default LockFunc: an advisory OS flock on <Dir>/.git/fak-commit.lock,
// reusing gpulease's cross-platform lock (flock on unix, LockFileEx on windows). gpulease
// is GPU-named but mechanically generic once its Path is overridden; its only specifics —
// the default lease path and a best-effort PID in the file — are harmless here. A held
// lock maps to ErrLockBusy (the LOCK_BUSY reason); the kernel drops the flock if the holder
// dies, so a crashed peer never wedges the lane.
func realLock(opts LockOptions) (func(), error) {
	path := opts.Path
	if path == "" {
		// Best-effort: place it under .git of the current repo. If we cannot resolve the
		// git dir, fall back to gpulease's own default path so we still serialize fak
		// writers (correctness of the post-commit assertion does not depend on the path).
		if gd, err := gitDir(); err == nil {
			path = filepath.Join(gd, "fak-commit.lock")
		}
	}
	timeout := opts.Timeout
	if timeout == 0 {
		timeout = DefaultLockTimeout
	}
	lease, err := gpulease.Acquire(gpulease.Options{
		Path:    path,
		NoWait:  opts.NoWait,
		Timeout: timeout,
		Logf:    func(string, ...any) {}, // silent: the CLI layer narrates, not the lock
	})
	if err != nil {
		if errors.Is(err, gpulease.ErrBusy) || errors.Is(err, gpulease.ErrTimeout) {
			return nil, ErrLockBusy
		}
		return nil, err
	}
	return lease.Release, nil
}

// gitDir resolves the absolute path of the current repo's .git directory.
func gitDir() (string, error) {
	out, err := exec.Command("git", "rev-parse", "--absolute-git-dir").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}
