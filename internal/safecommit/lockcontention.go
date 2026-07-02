package safecommit

// lockcontention.go — riding out a PEER'S raw-git lock during the write steps.
//
// The advisory fak-commit.lock serializes fak writers, but a shared multi-session
// tree also carries RAW git processes fak cannot see: a harness `git status`
// refreshing the index, a peer's `git fetch`, a watchdog probe. Any of them can
// hold .git/index.lock (or a ref lock) for the instant `git add`/`git commit`
// needs it, and git then fails with "Unable to create '.git/index.lock': File
// exists" — a purely TRANSIENT breakage of the shared tree that previously
// surfaced as HOOK_REFUSED, the PERMANENT class (exit 1, "ran, result is bad —
// halt"), so a fleet caller stopped retrying the one failure class a retry
// reliably fixes. The remedy is two-layered:
//
//  1. RETRY IN PLACE: the failed write is re-run a few times with a short,
//     jittered backoff — an optional-lock holder clears in well under a second,
//     so most contention never escapes this loop and the commit just lands;
//  2. CLASSIFY HONESTLY: contention that outlives the retries surfaces as
//     LOCK_BUSY (exit 3, "blocked — retry or replan"), never HOOK_REFUSED, so
//     the caller's next action is "try again shortly", not "halt and debug a
//     hook". A genuine hook refusal never matches the lock signatures and keeps
//     its exact historical classification and single-attempt behavior.

import (
	"context"
	"math/rand"
	"strings"
	"time"
)

// gitLockContentionNeedles are the (lowercased) substrings of the messages git
// emits when it loses a repository-lock race to a concurrent process. The set is
// deliberately NARROW — every needle names a git lockfile or git's own lock
// prose — so a hook's refusal output can never be misread as contention: a false
// "transient" here would put a caller into a retry loop against a permanent
// refusal, which is worse than the historical misclassification this file fixes.
var gitLockContentionNeedles = []string{
	"index.lock",          // fatal: Unable to create '.git/index.lock': File exists.
	"packed-refs.lock",    // ref packing raced a concurrent writer
	".lock': file exists", // any other .git/*.lock (config.lock, HEAD.lock, shallow.lock, …)
	"cannot lock ref",     // error: cannot lock ref 'refs/heads/main': …
	"another git process", // the advice trailer git appends to the index.lock fatal
}

// gitLockPermanentMarkers force a failure OUT of the contention class even when
// a needle also matched. "cannot lock ref" covers more than the transient race:
// a CORRUPT ref fails as "error: cannot lock ref 'refs/heads/main': unable to
// resolve reference 'refs/heads/main': reference broken" — a permanent
// condition no retry clears, which must keep the halt-class HOOK_REFUSED so an
// operator investigates instead of a fleet retrying forever.
var gitLockPermanentMarkers = []string{
	"unable to resolve reference", // the ref itself is unreadable, not merely locked
	"reference broken",            // git's corrupt-ref suffix
}

// isGitLockContention reports whether raw git output describes a lost
// repository-lock race — the transient failure class a short retry fixes.
// Permanent markers win over the needles: a corrupt-ref failure is never
// contention, even though it carries "cannot lock ref".
func isGitLockContention(out string) bool {
	low := strings.ToLower(out)
	for _, marker := range gitLockPermanentMarkers {
		if strings.Contains(low, marker) {
			return false
		}
	}
	for _, needle := range gitLockContentionNeedles {
		if strings.Contains(low, needle) {
			return true
		}
	}
	return false
}

// lockContentionAttempts is the TOTAL number of tries one write op (`git add`,
// `git commit`) makes when each failure is lock contention. The waits between
// tries sum to ~1-2s — an optional-lock holder (a `git status` index refresh)
// clears in milliseconds and even a peer's full commit-with-hooks clears in a
// couple of seconds, while the fak advisory lock this loop runs under is bounded
// at DefaultLockTimeout (10s), so the retries stay well inside the window a
// waiting fak peer already tolerates.
const lockContentionAttempts = 4

// lockContentionBudget caps the TOTAL wall-clock one riding loop may consume —
// sleeps AND re-runs. The attempt cap alone under-counts the real cost of a
// `git commit` retry: a "cannot lock ref" failure happens AFTER the pre-commit
// hooks already ran, so every retry re-pays the whole hook suite inside the fak
// advisory lock, and four hook-heavy attempts could otherwise eat most of the
// 10s window (DefaultLockTimeout) that queued fak peers are polling against.
// Once the budget is spent the loop stops retrying and the caller surfaces
// LOCK_BUSY — the peers' window stays protected no matter how slow the hooks.
const lockContentionBudget = 3 * time.Second

// contentionNow is time.Now, injectable so tests exercise the time cap without
// real waits.
var contentionNow = time.Now

// contentionSleep is time.Sleep, injectable so tests exercise the retry loop
// without real waits.
var contentionSleep = time.Sleep

// lockContentionWait is the pre-attempt backoff before retry `attempt` (1-based):
// attempt²×150ms equal-jittered to [base/2, base] — the same shape as the
// upstream-transport schedule in internal/agent, scaled down to lockfile
// latencies (150ms, 600ms, 1.35s pre-jitter). The jitter matters under high
// concurrency: several fak writers queued behind one raw-git holder would
// otherwise re-collide in lockstep the instant it clears.
func lockContentionWait(attempt int) time.Duration {
	base := time.Duration(attempt*attempt) * 150 * time.Millisecond
	half := int64(base / 2)
	return time.Duration(half) + time.Duration(rand.Int63n(half+1))
}

// runRidingLockContention runs one mutating git op, re-running it up to
// lockContentionAttempts times — and within lockContentionBudget of total
// wall-clock — while each failure is lock contention. It returns the FINAL
// (out, code, err): a success, the first non-contention failure (a hook refusal
// is never retried), or the last contention failure once either bound is spent —
// the caller classifies that as LOCK_BUSY. A cancelled ctx stops the loop
// between attempts; the in-flight git run already honors ctx via the Runner
// itself.
func runRidingLockContention(ctx context.Context, run Runner, dir string, args ...string) (string, int, error) {
	start := contentionNow()
	var out string
	var code int
	var err error
	for attempt := 1; ; attempt++ {
		out, code, err = run(ctx, dir, args...)
		if err != nil || code == 0 || !isGitLockContention(out) {
			return out, code, err
		}
		if attempt >= lockContentionAttempts || ctx.Err() != nil {
			return out, code, err
		}
		if contentionNow().Sub(start) >= lockContentionBudget {
			// Time budget spent: the re-runs themselves (hooks included) consumed
			// the window; stop riding so queued fak peers still reach the lock.
			return out, code, err
		}
		contentionSleep(lockContentionWait(attempt))
	}
}
