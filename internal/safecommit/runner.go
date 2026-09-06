package safecommit

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// newGitCmd builds the *exec.Cmd realRunner executes, applying by construction the two
// environment invariants that keep a busy shared tree DRAINED — the reason this is a
// named helper rather than inline: both defaults are one careless edit from re-opening a
// contention class, so they are pinned here and asserted by a guard test that never has
// to spawn git.
//
//   - GIT_OPTIONAL_LOCKS=0 on EVERY invocation: the read probes (rev-parse, symbolic-ref,
//     status, diff) must never take git's OPTIONAL locks — a plain `git status` otherwise
//     opportunistically refreshes the index under .git/index.lock and, on a busy shared
//     tree, collides with a concurrent writer (the documented burst-time stall class that
//     once wedged the commit lane). Mandatory write locks (add/commit) are unaffected;
//     contention on those is ridden out by runRidingLockContention instead.
//   - FAK_SAFECOMMIT_VETTED=1 on `git commit` ONLY: the handshake with the hook-layer
//     BARE_COMMIT_SWEEP gate (issue #3615). This `git commit` was issued by safecommit,
//     which already vetted the DECLARED pathspec through the path-scoped
//     PRESTAGED_PATH_OVERLAP guard; the marker rides down into the pre-commit hook so that
//     gate stands down instead of re-flagging the vetted commit as an unvetted bare sweep.
//     A raw `git commit` carries no marker and is still gated, so the marker must never
//     leak onto a read probe or push.
func newGitCmd(ctx context.Context, dir string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Stdin = strings.NewReader("")
	cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0")
	if len(args) > 0 && args[0] == "commit" {
		cmd.Env = append(cmd.Env, "FAK_SAFECOMMIT_VETTED=1")
	}
	if dir != "" {
		cmd.Dir = dir
	}
	return cmd
}

// realRunner is the default Runner: it runs the real git binary. It mirrors
// witness.gitRunner's contract — a non-zero git exit is returned in code (not err); err
// signals git could not be EXECUTED at all — with one deliberate difference: it MERGES
// stderr into the returned stdout. The executor needs a hook's refusal / a push rejection
// message to surface in Result.Detail, which witness (Stderr = nil) discards.
func realRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := newGitCmd(ctx, dir, args...)
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
// the default lease path and a best-effort PID in the file — are harmless here, and the
// recorded PID is exactly what reapStaleLock keys on.
//
// A held lock maps to ErrLockBusy (the LOCK_BUSY reason). On a clean exit the OS drops the
// flock, but an ABNORMALLY terminated committer (killed/crashed, not a clean os.Exit) can
// on Windows leave its LockFileEx region orphaned on the path — observed in the field as a
// ~56-minute fak-commit.lock wedge that stalled the WHOLE shared-trunk auto-gardening lane
// (every peer's commit blocked behind a dead PID's lock). reapStaleLock is the guard: a
// pre-flight that removes the lockfile when its recorded holder PID is no longer alive, so
// a dead committer can never wedge the lane. It runs only for THIS commit lock (never the
// GPU-lease hot path) and only deletes a provably-dead holder's file, so a live committer
// is never disturbed.
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
	// Reap-aware acquire (issue #2339): instead of a single pre-flight reap followed by a
	// blind blocking wait, poll the lock and RE-reap on every attempt. A holder that dies
	// (or whose PID is reused) mid-wait is broken within one poll interval rather than
	// stalling the waiter for the whole timeout — the "waiters do the liveness check inside
	// their wait loop" half of the fix.
	acq := func(noWait bool) (func(), error) {
		lease, err := gpulease.Acquire(gpulease.Options{
			Path:    path,
			NoWait:  noWait,
			Timeout: 0,                       // acquireWithReap owns the bound; each probe is non-blocking
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
	reap := func() {
		if path != "" {
			reapStaleLock(path)
		}
	}
	started := time.Now()
	release, err := acquireWithReap(acq, reap, opts.NoWait, timeout, lockReapPoll, time.Now, time.Sleep)
	if err != nil {
		if errors.Is(err, ErrLockBusy) {
			probe := ProbeLock(path)
			deadline := timeout
			if opts.NoWait {
				deadline = 0
			}
			return nil, &LockBusyError{Receipt: LockWaitReceipt{
				ElapsedNS:      time.Since(started).Nanoseconds(),
				DeadlineNS:     deadline.Nanoseconds(),
				HolderPID:      probe.HolderPID,
				HolderAlive:    probe.Alive,
				HolderStale:    probe.Stale,
				HolderForeign:  probe.Foreign,
				LockAgeSeconds: probe.AgeSeconds,
			}}
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
	if leaserefEnabled() {
		release = withLeasePublish(release)
	}
	return release, nil
}

const lockReapPoll = 250 * time.Millisecond

func acquireWithReap(acquire func(noWait bool) (func(), error), reap func(), noWait bool, timeout, poll time.Duration, now func() time.Time, sleep func(time.Duration)) (func(), error) {
	if poll <= 0 {
		poll = lockReapPoll
	}
	if timeout <= 0 {
		timeout = DefaultLockTimeout
	}
	deadline := now().Add(timeout)
	for {
		reap()
		release, err := acquire(true)
		if err == nil {
			return release, nil
		}
		if !errors.Is(err, ErrLockBusy) {
			return nil, err
		}
		if noWait || !now().Before(deadline) {
			return nil, ErrLockBusy
		}
		wait := poll
		if remaining := deadline.Sub(now()); remaining < wait {
			wait = remaining
		}
		if wait > 0 {
			sleep(wait)
		}
	}
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

// reapEventf receives a LOCK_BROKEN notice the first-and-only time a stale commit lock is
// actually broken (issue #2339's "logged event" acceptance), and a throttled
// LOCK_REAP_FAILED notice when a break was DECIDED and the OS refused it (#5335). It is a
// package var so both are visible to an operator by default (stderr) and capturable in
// tests. Both are rare — one fires only when a dead/foreign holder's lock is removed, the
// other only when such a removal is refused — so this is a signal, not hot-path noise.
var reapEventf = func(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "fak: "+format+"\n", args...)
}

// reapStaleLock removes the commit lockfile at path when its recorded holder is no longer
// a live committer — a DEAD PID, or a live PID whose image is provably foreign (a reused
// PID number). It is the pre-flight that stops a dead committer from wedging the
// shared-trunk commit lane (see realLock's doc): gpulease records the holder's PID in the
// lockfile, so a stale lock is one whose PID is gone or reused. We read that PID and only
// break the lock when it is provably not a live committer — gpulease.Acquire then takes a
// clean lock on a fresh inode. Every step is best-effort and fail-safe:
//   - an unreadable/absent file, an unparseable PID, a STILL-ALIVE committer, or an image
//     we cannot read => do nothing (we never delete a lock a live committer holds);
//   - a remove failure never fails the commit — Acquire's bounded wait/timeout is the
//     backstop, so the worst case is the pre-reap regression (wait it out), never a
//     corrupted lock. It is no longer SILENT, though: see LOCK_REAP_FAILED below.
//
// A successful break emits one structured LOCK_BROKEN event naming the reason, holder PID,
// and lock age, so a fleet operator can see WHY the lane was unwedged instead of a silent
// disappearance. A break that was decided but REFUSED by the OS emits a throttled
// LOCK_REAP_FAILED carrying the errno class and a cumulative attempt count — the evidence
// an 85-minute wedge produced none of (#5335). This is the in-code form of the manual
// `rm .git/fak-commit.lock` that unblocked a wedged 56-minute commit stall in the field,
// made automatic, PID-guarded, and auditable so it is safe to run on every acquire.
func reapStaleLock(path string) {
	res := ReapStaleLockResult(path)
	if !res.Reaped {
		if res.Failed() {
			// The lock is PROVABLY not held by a live committer and the OS still refused to
			// remove it. Nothing downstream will clear it, so this is the one not-reaped
			// outcome that must reach an operator (#5335): the 85-minute lane wedge ran this
			// exact path every 250ms and recorded nothing, leaving no evidence of why.
			if attempts, announce := noteReapFailure(path, time.Now()); announce {
				reapEventf("LOCK_REAP_FAILED %s pid=%d age=%ds attempts=%d err=%s path=%s detail=%q",
					res.Reason, res.HolderPID, res.AgeSeconds, attempts, res.RemoveErrClass, path, res.RemoveErr)
			}
		}
		return
	}
	clearReapFailure(path)
	if res.Reason == ReapReasonHolderForeign && res.StartedAfterLock {
		// The break was proven by start time, not by the image. Say so: the image on such a
		// break is usually an ordinary fleet name that the allowlist would have cleared, so
		// printing it alone would misattribute the evidence (#5892).
		reapEventf("LOCK_BROKEN %s pid=%d age=%ds started_after_lock=true image=%s path=%s",
			res.Reason, res.HolderPID, res.AgeSeconds, res.Image, path)
		return
	}
	if res.Reason == ReapReasonHolderForeign && res.Image != "" {
		reapEventf("LOCK_BROKEN %s pid=%d age=%ds image=%s path=%s",
			res.Reason, res.HolderPID, res.AgeSeconds, res.Image, path)
		return
	}
	reapEventf("LOCK_BROKEN %s pid=%d age=%ds path=%s",
		res.Reason, res.HolderPID, res.AgeSeconds, path)
}

// reapFailureNoticeInterval bounds how often a REPEATEDLY-failing reap re-announces
// itself. acquireWithReap calls reapStaleLock every lockReapPoll (250ms) for the whole of
// its bounded wait (DefaultLockTimeout, 10s), so one wedged acquire would otherwise emit
// ~40 identical lines — and a flood is its own kind of silence. The first failure is
// always announced immediately; after that the notice repeats at this cadence carrying a
// cumulative attempt count, so a single acquire leaves one line and a caller that keeps
// retrying against a persistent wedge leaves a readable trail instead of one line per poll.
const reapFailureNoticeInterval = 30 * time.Second

// reapFailures tracks, per lock path, how many consecutive remove refusals have been
// seen and when the last notice was emitted. It is reset the moment a reap finally
// succeeds, so a later failure streak starts from a fresh, immediately-announced first
// attempt rather than inheriting a throttle from a resolved incident.
var (
	reapFailureMu sync.Mutex
	reapFailures  = map[string]*reapFailureStreak{}
)

type reapFailureStreak struct {
	attempts   int
	lastNotice time.Time
}

// noteReapFailure records one refused remove and reports whether this occurrence should
// be announced, along with the cumulative attempt count to announce it with.
func noteReapFailure(path string, now time.Time) (attempts int, announce bool) {
	reapFailureMu.Lock()
	defer reapFailureMu.Unlock()
	st := reapFailures[path]
	if st == nil {
		st = &reapFailureStreak{}
		reapFailures[path] = st
	}
	st.attempts++
	if st.attempts == 1 || now.Sub(st.lastNotice) >= reapFailureNoticeInterval {
		st.lastNotice = now
		return st.attempts, true
	}
	return st.attempts, false
}

// clearReapFailure forgets a path's failure streak once the lock is gone.
func clearReapFailure(path string) {
	reapFailureMu.Lock()
	delete(reapFailures, path)
	reapFailureMu.Unlock()
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
	cmd := exec.Command("git", "rev-parse", "--absolute-git-dir")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}
