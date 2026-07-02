package safecommit

import (
	"context"
	"strings"
	"testing"
	"time"
)

// indexLockFatal is the exact message git emits when a concurrent process holds
// the index lock — the transient breakage class this file's retries ride out.
const indexLockFatal = "fatal: Unable to create 'C:/work/fak/.git/index.lock': File exists.\n\n" +
	"Another git process seems to be running in this repository, e.g.\n" +
	"an editor opened by 'git commit'. Please make sure all processes\n" +
	"are terminated then try again."

// swallowContentionSleep replaces the retry sleep for the duration of a test,
// recording each wait instead of actually sleeping.
func swallowContentionSleep(t *testing.T) *[]time.Duration {
	t.Helper()
	var waits []time.Duration
	prev := contentionSleep
	contentionSleep = func(d time.Duration) { waits = append(waits, d) }
	t.Cleanup(func() { contentionSleep = prev })
	return &waits
}

func TestIsGitLockContention(t *testing.T) {
	cases := []struct {
		name string
		out  string
		want bool
	}{
		{"index lock fatal", indexLockFatal, true},
		{"ref lock", "error: cannot lock ref 'refs/heads/main': Unable to create 'C:/work/fak/.git/refs/heads/main.lock': File exists.", true},
		{"packed refs", "fatal: Unable to create 'C:/work/fak/.git/packed-refs.lock': File exists", true},
		{"generic lockfile", "fatal: could not write config file: Unable to create 'C:/work/fak/.git/config.lock': File exists", true},
		{"hook refusal", "FILE_ADMISSION: docs/tmp-scratch.md is a one-off operational artifact", false},
		{"vet failure", "internal/foo/bar.go:10: undefined: baz", false},
		{"empty", "", false},
		// "cannot lock ref" also fires on PERMANENT ref corruption — the marker
		// must win so a broken ref halts (HOOK_REFUSED) instead of retrying.
		{"corrupt ref", "error: cannot lock ref 'refs/heads/main': unable to resolve reference 'refs/heads/main': reference broken", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isGitLockContention(tc.out); got != tc.want {
				t.Fatalf("isGitLockContention(%q) = %v, want %v", tc.out, got, tc.want)
			}
		})
	}
}

// A single lost index.lock race on `git add` self-heals in place: the retry
// lands the add and the commit completes cleanly — no reason, no caller retry.
func TestAddLockContention_retriesInPlaceAndCommits(t *testing.T) {
	waits := swallowContentionSleep(t)
	g := &fakeGit{reply: onTrunkBase()}
	addFailures := 1
	addCalls := 0
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if len(args) > 0 && args[0] == "add" {
			addCalls++
			if addFailures > 0 {
				addFailures--
				return indexLockFatal, 128, nil
			}
		}
		return g.run(ctx, dir, args...)
	}

	res, err := CommitWith(context.Background(), run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" || !res.Committed || !res.Verified {
		t.Fatalf("transient index.lock should self-heal, got %+v", res)
	}
	if addCalls != 2 {
		t.Fatalf("add attempts = %d, want 2 (one contention, one success)", addCalls)
	}
	if len(*waits) != 1 || (*waits)[0] <= 0 {
		t.Fatalf("waits = %v, want exactly one positive backoff between attempts", *waits)
	}
}

// Contention that outlives every retry is LOCK_BUSY — the retryable exit-3
// class — never HOOK_REFUSED (the exit-1 halt that told callers to stop
// retrying the one failure a retry fixes).
func TestAddLockContention_persistentIsLockBusyNotHookRefused(t *testing.T) {
	swallowContentionSleep(t)
	g := &fakeGit{reply: onTrunkBase()}
	addCalls := 0
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if len(args) > 0 && args[0] == "add" {
			addCalls++
			return indexLockFatal, 128, nil
		}
		return g.run(ctx, dir, args...)
	}

	res, err := CommitWith(context.Background(), run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonLockBusy {
		t.Fatalf("Reason = %q, want %q (persistent contention is transient, not a hook refusal)", res.Reason, ReasonLockBusy)
	}
	if res.Committed {
		t.Fatalf("nothing should be committed under persistent contention: %+v", res)
	}
	if addCalls != lockContentionAttempts {
		t.Fatalf("add attempts = %d, want %d", addCalls, lockContentionAttempts)
	}
	if !strings.Contains(res.Detail, "index.lock") {
		t.Fatalf("Detail should carry the raw lock message, got %q", res.Detail)
	}
}

// A lost ref-lock race on `git commit` classifies the same way as the add step.
func TestCommitRefLockContention_persistentIsLockBusy(t *testing.T) {
	swallowContentionSleep(t)
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["commit"] = reply{
		out:  "error: cannot lock ref 'refs/heads/main': Unable to create 'C:/work/fak/.git/refs/heads/main.lock': File exists.",
		code: 128,
	}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonLockBusy {
		t.Fatalf("Reason = %q, want %q", res.Reason, ReasonLockBusy)
	}
	commits := 0
	for _, c := range g.calls {
		if len(c) > 0 && c[0] == "commit" {
			commits++
		}
	}
	if commits != lockContentionAttempts {
		t.Fatalf("commit attempts = %d, want %d", commits, lockContentionAttempts)
	}
}

// A genuine hook refusal keeps its exact historical behavior: HOOK_REFUSED,
// single attempt, no retry — a permanent refusal must never be re-hammered.
func TestHookRefusal_notRetriedStillHookRefused(t *testing.T) {
	waits := swallowContentionSleep(t)
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["commit"] = reply{out: "FILE_ADMISSION: docs/tmp-scratch.md is private-only content", code: 1}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonHookRefused {
		t.Fatalf("Reason = %q, want %q", res.Reason, ReasonHookRefused)
	}
	commits := 0
	for _, c := range g.calls {
		if len(c) > 0 && c[0] == "commit" {
			commits++
		}
	}
	if commits != 1 {
		t.Fatalf("commit attempts = %d, want 1 (a permanent refusal is never retried)", commits)
	}
	if len(*waits) != 0 {
		t.Fatalf("no backoff should fire for a permanent refusal, got %v", *waits)
	}
}

// Slow re-runs (a hook-heavy `git commit` retried on a ref-lock race) are
// bounded by wall-clock, not just attempt count: once lockContentionBudget is
// spent the loop stops riding — protecting the 10s window queued fak peers
// poll against — and the caller still classifies the failure LOCK_BUSY.
func TestLockContention_timeBudgetStopsRiding(t *testing.T) {
	waits := swallowContentionSleep(t)
	prevNow := contentionNow
	now := time.Unix(1_750_000_000, 0)
	contentionNow = func() time.Time { return now }
	t.Cleanup(func() { contentionNow = prevNow })

	g := &fakeGit{reply: onTrunkBase()}
	commits := 0
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if len(args) > 0 && args[0] == "commit" {
			commits++
			// Each re-run models a slow hook suite: the clock advances past the
			// whole budget before the retry decision is made.
			now = now.Add(lockContentionBudget + time.Second)
			return "error: cannot lock ref 'refs/heads/main': Unable to create 'C:/work/fak/.git/refs/heads/main.lock': File exists.", 128, nil
		}
		return g.run(ctx, dir, args...)
	}

	res, err := CommitWith(context.Background(), run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonLockBusy {
		t.Fatalf("Reason = %q, want %q", res.Reason, ReasonLockBusy)
	}
	if commits != 1 {
		t.Fatalf("commit attempts = %d, want 1 (time budget spent after the first slow re-run)", commits)
	}
	if len(*waits) != 0 {
		t.Fatalf("no backoff should fire once the budget is spent, got %v", *waits)
	}
}

// The backoff schedule is positive, grows with the attempt, and is jittered
// into (0, base] so queued writers do not re-collide in lockstep.
func TestLockContentionWait_boundedAndPositive(t *testing.T) {
	for attempt := 1; attempt < lockContentionAttempts; attempt++ {
		base := time.Duration(attempt*attempt) * 150 * time.Millisecond
		for i := 0; i < 50; i++ {
			w := lockContentionWait(attempt)
			if w < base/2 || w > base {
				t.Fatalf("attempt %d: wait %v outside [base/2, base] = [%v, %v]", attempt, w, base/2, base)
			}
		}
	}
}
