// Package gpulease is a machine-wide advisory lease so that only one GPU-heavy
// process loads a model at a time.
//
// Motivation: a -metal modelbench process is heavy on this box — the f32/Q8 model
// on the CPU side (gigabytes) plus the full f16 weight set uploaded to unified
// memory. Two or three of them launched concurrently across separate Claude/fleet
// sessions stack their residency on the SAME unified-memory pool and overrun
// physical RAM, which on 2026-06-18 produced a jetsam-kill cascade and finally a
// kernel watchdog panic. Nothing coordinated those launches.
//
// This lease is that coordination: a GPU-heavy process Acquire()s the lease before
// loading its model and holds it until exit, so concurrent launches QUEUE instead
// of stacking. It is an OS-level flock on a single lockfile, so it works across
// unrelated processes (and is released automatically if the holder dies, since the
// kernel drops the flock when the fd closes).
package gpulease

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ErrBusy is returned by Acquire when NoWait is set and the lease is already held.
var ErrBusy = errors.New("gpulease: lease is held by another process")

// ErrTimeout is returned by Acquire when Timeout elapses before the lease is free.
var ErrTimeout = errors.New("gpulease: timed out waiting for the lease")

var errLockBusy = errors.New("gpulease: lock busy")

// Lease is a held machine-wide GPU lease. Release frees it; the OS also drops the
// underlying flock if the process exits without calling Release.
type Lease struct {
	f    *os.File
	path string
}

// Options configures Acquire.
type Options struct {
	// Path is the lockfile. Empty means $FAK_GPU_LEASE, else <tmp>/fak-gpu.lease.
	Path string
	// NoWait makes Acquire fail with ErrBusy immediately instead of waiting.
	NoWait bool
	// Timeout bounds the wait (ignored when NoWait). Zero waits indefinitely.
	Timeout time.Duration
	// Logf receives a one-line notice the first time Acquire has to wait, so a
	// queued process is never silent. Nil logs to stderr.
	Logf func(format string, args ...any)
	// pollEvery overrides the busy-poll interval (tests only; 0 => 500ms).
	pollEvery time.Duration
}

// DefaultPath is the lease lockfile path Acquire uses when Options.Path is empty.
func DefaultPath() string {
	if p := os.Getenv("FAK_GPU_LEASE"); p != "" {
		return p
	}
	return filepath.Join(os.TempDir(), "fak-gpu.lease")
}

// Acquire takes the machine-wide GPU lease. By default it blocks until the lease is
// free, logging one waiting notice (with the holder's pid) the first time it has to
// wait. With Options.NoWait it returns ErrBusy immediately when the lease is held.
// The caller holds the lease until Release (or process exit).
func Acquire(opts Options) (*Lease, error) {
	path := opts.Path
	if path == "" {
		path = DefaultPath()
	}
	logf := opts.Logf
	if logf == nil {
		logf = func(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }
	}
	poll := opts.pollEvery
	if poll <= 0 {
		poll = 500 * time.Millisecond
	}

	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("gpulease: open %s: %w", path, err)
	}

	deadline := time.Time{}
	if opts.Timeout > 0 {
		deadline = time.Now().Add(opts.Timeout)
	}
	waited := false
	for {
		err := tryLock(f, path)
		if err == nil {
			// Record our pid so a future waiter can name the holder (best-effort).
			// Write-THEN-truncate (not truncate-then-write): a concurrent waiter's
			// holderPID read must never catch a zero-length window between the two.
			rec := []byte(strconv.Itoa(os.Getpid()) + "\n")
			if _, werr := f.WriteAt(rec, 0); werr == nil {
				_ = f.Truncate(int64(len(rec)))
			}
			return &Lease{f: f, path: path}, nil
		}
		if !errors.Is(err, errLockBusy) {
			f.Close()
			return nil, fmt.Errorf("gpulease: lock %s: %w", path, err)
		}
		if opts.NoWait {
			f.Close()
			return nil, ErrBusy
		}
		if !waited {
			waited = true
			logf("gpulease: GPU busy (held by %s); waiting for %s", holderPID(f), path)
		}
		// Sleep until the next poll, but never past the deadline (otherwise a Timeout
		// shorter than poll would overshoot by most of a poll interval).
		sleep := poll
		if !deadline.IsZero() {
			rem := time.Until(deadline)
			if rem <= 0 {
				f.Close()
				return nil, ErrTimeout
			}
			if rem < sleep {
				sleep = rem
			}
		}
		time.Sleep(sleep)
	}
}

// Release frees the lease. Safe to call once; subsequent calls are no-ops.
func (l *Lease) Release() {
	if l == nil || l.f == nil {
		return
	}
	_ = unlock(l.f, l.path)
	_ = l.f.Close()
	l.f = nil
}

// holderPID reads the pid the current holder wrote into the lockfile, or "?" if it
// is empty/unreadable. Purely informational for the waiting notice.
func holderPID(f *os.File) string {
	buf := make([]byte, 16)
	n, _ := f.ReadAt(buf, 0)
	s := string(buf[:n])
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i] // first line only — ignore any stale trailing bytes
	}
	if s = strings.TrimSpace(s); s == "" {
		return "pid ?"
	}
	return "pid " + s
}
