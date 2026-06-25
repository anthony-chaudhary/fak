package safecommit

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
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
	// Cross-machine VISIBILITY tier (#825): when opted in (FAK_LEASEREF=1), publish the
	// held lease as a refs/fak/locks/<id> record ALONGSIDE the flock, so a peer on another
	// clone can SEE this same-host lock after an ordinary fetch. It is strictly ADDITIVE
	// and best-effort: the flock above is the authority for same-host serialization, and a
	// leaseref publish/delete failure NEVER blocks or fails the commit (it is the slower,
	// cross-host tier layered on top — distribution, not atomic acquisition). The record is
	// deleted on release, composed in front of the flock's own release.
	release := lease.Release
	if leaserefEnabled() {
		release = withLeasePublish(release)
	}
	return release, nil
}

// leaserefEnabled reports whether the cross-machine lease-visibility tier is opted in.
// OFF by default — the flock is the same-host fast path and stays the only behavior unless
// a fleet explicitly turns on the ref-namespaced visibility tier.
func leaserefEnabled() bool {
	return strings.EqualFold(strings.TrimSpace(os.Getenv("FAK_LEASEREF")), "1") ||
		strings.EqualFold(strings.TrimSpace(os.Getenv("FAK_LEASEREF")), "on")
}

// withLeasePublish publishes a best-effort leaseref record for the duration this commit
// holds the flock, and composes its deletion in front of the flock's release. Every step
// is best-effort: a leaseref error is swallowed (the commit's correctness rests on the
// flock + the post-commit pathspec assertion, not on the ref store). It returns the inner
// release unchanged when the record cannot be published, so release is always safe to call.
func withLeasePublish(inner func()) func() {
	store := leaseref.New()
	id := leaseID()
	rec := leaseref.Record{
		ID:          id,
		TreeGlobs:   []string{"."}, // a commit lock is whole-tree from the cross-host view
		Holder:      leaseHolder(),
		AcquiredAt:  time.Now().Unix(),
		TTLSeconds:  int64(DefaultLockTimeout/time.Second) + 60, // bounded: a crashed holder is reapable
		Description: "safecommit advisory commit lock (cross-machine visibility tier)",
	}
	published := false
	if _, err := store.Acquire(context.Background(), rec); err == nil {
		published = true
	}
	return func() {
		if published {
			_ = store.Release(context.Background(), id) // best-effort; never block the commit
		}
		inner()
	}
}

// leaseID derives a stable-enough, ref-safe lease id for this holder. It is a single safe
// ref segment (host + pid), so two concurrent fak writers on different hosts publish under
// distinct refs and a peer can attribute each.
func leaseID() string {
	host, _ := os.Hostname()
	host = sanitizeIDPart(host)
	if host == "" {
		host = "host"
	}
	return "commit-" + host + "-" + sanitizeIDPart(strconv.Itoa(os.Getpid()))
}

// leaseHolder is the free-form identity recorded in the lease (host:pid).
func leaseHolder() string {
	host, _ := os.Hostname()
	if host == "" {
		host = "host"
	}
	return host + ":" + strconv.Itoa(os.Getpid())
}

// sanitizeIDPart keeps only the characters leaseref.validID accepts in one ref segment.
func sanitizeIDPart(s string) string {
	var b strings.Builder
	for _, c := range []byte(s) {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '-', c == '_':
			b.WriteByte(c)
		default:
			b.WriteByte('-')
		}
	}
	return b.String()
}

// gitDir resolves the absolute path of the current repo's .git directory.
func gitDir() (string, error) {
	out, err := exec.Command("git", "rev-parse", "--absolute-git-dir").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}
