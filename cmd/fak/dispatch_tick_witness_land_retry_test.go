package main

// #3613/#12449: the dispatch land seam must CONSUME a refused optimistic land
// instead of discarding it. A LAND_READBACK_MISMATCH race refusal retries the land
// (bounded), but every terminal refusal retains the worktree (the only copy of the
// diff). Only a durable successful land may reach reap.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// withLandRetrySeams stubs the #3613 land/reap/backoff seams and returns an
// ordered call log ("land" per attempt, "reap:<wtPath>"). The stubbed land
// replays results in order; the last result repeats if the seam is called again
// (an always-refused land).
func withLandRetrySeams(t *testing.T, results []workerworktree.Result) *[]string {
	t.Helper()
	oldLand, oldReap, oldSleep := dispatchLandWorktreeOnce, dispatchReapWorktree, dispatchLandRetrySleep
	t.Cleanup(func() {
		dispatchLandWorktreeOnce, dispatchReapWorktree, dispatchLandRetrySleep = oldLand, oldReap, oldSleep
	})
	log := &[]string{}
	calls := 0
	dispatchLandWorktreeOnce = func(root, wtPath, base string, tree []string) workerworktree.Result {
		*log = append(*log, "land")
		i := calls
		if i >= len(results) {
			i = len(results) - 1
		}
		calls++
		return results[i]
	}
	dispatchReapWorktree = func(root, wtPath string) workerworktree.Result {
		*log = append(*log, "reap:"+wtPath)
		return workerworktree.Result{OK: true, Path: wtPath, Removed: true}
	}
	dispatchLandRetrySleep = func(int) {} // sequencing, not wall-clock jitter
	return log
}

func raceRefusedResult() workerworktree.Result {
	return workerworktree.Result{OK: false, Code: workerworktree.LandResultReconciliationRequired,
		Applied: true, Committed: true, Preserved: true,
		Reason: workerworktree.LandReadbackMismatchToken +
			": trunk HEAD abcdef123456 does not carry intended path(s) cmd/x.go after commit — shared-index race, land not trusted (#3547)"}
}

// TestRefusedLandRetriesThenSucceedsBeforeReap is the retry-then-succeed witness:
// one race refusal, then a clean land — the seam must land TWICE (the retry
// consumed the refusal) and only then reap, so the diff survives the race.
func TestRefusedLandRetriesThenSucceedsBeforeReap(t *testing.T) {
	log := withLandRetrySeams(t, []workerworktree.Result{
		raceRefusedResult(),
		{OK: true, Applied: true, Committed: true},
	})
	landAndReapWorkerWorktreeDefault("/root", "/wt/fak-worker-wt-cmd-abc", "base", []string{"cmd"})
	want := []string{"land", "land", "reap:/wt/fak-worker-wt-cmd-abc"}
	assertLandRetryLog(t, *log, want)
}

// TestRefusedLandExhaustionRetainsWorker pins the bound: an always-race-refused
// land is attempted exactly dispatchLandRefusedAttempts times, then the worktree
// remains available for explicit reconciliation. The retry is bounded without
// turning budget exhaustion into implicit discard authority.
func TestRefusedLandExhaustionRetainsWorker(t *testing.T) {
	if dispatchLandRefusedAttempts < 2 {
		t.Fatalf("the #3613 bound must allow at least one retry, got %d", dispatchLandRefusedAttempts)
	}
	log := withLandRetrySeams(t, []workerworktree.Result{raceRefusedResult()})
	landAndReapWorkerWorktreeDefault("/root", "/wt/fak-worker-wt-cmd-abc", "base", []string{"cmd"})
	want := []string{}
	for i := 0; i < dispatchLandRefusedAttempts; i++ {
		want = append(want, "land")
	}
	assertLandRetryLog(t, *log, want)
}

// TestDeterministicRefusalRetainsWorker pins the guard rail: a fail-closed
// reconciliation result is deterministic, so the seam lands exactly once and
// retains the worker rather than replaying or reaping it.
func TestDeterministicRefusalRetainsWorker(t *testing.T) {
	log := withLandRetrySeams(t, []workerworktree.Result{
		{OK: false, Code: workerworktree.LandResultReconciliationRequired,
			Path: "/wt/fak-worker-wt-cmd-abc", Preserved: true,
			Reason: "isolated land requires reconciliation: worktree verify failed"},
	})
	landAndReapWorkerWorktreeDefault("/root", "/wt/fak-worker-wt-cmd-abc", "base", []string{"cmd"})
	assertLandRetryLog(t, *log, []string{"land"})
}

// TestCleanLandSingleAttemptThenReap pins the happy path untouched: a first-try
// OK land makes exactly one attempt, then reaps.
func TestCleanLandSingleAttemptThenReap(t *testing.T) {
	log := withLandRetrySeams(t, []workerworktree.Result{
		{OK: true, Applied: true, Committed: true},
	})
	landAndReapWorkerWorktreeDefault("/root", "/wt/fak-worker-wt-cmd-abc", "base", []string{"cmd"})
	assertLandRetryLog(t, *log, []string{"land", "reap:/wt/fak-worker-wt-cmd-abc"})
}

func assertLandRetryLog(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("land/reap sequence mismatch: got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("land/reap sequence mismatch at %d: got %v want %v", i, got, want)
		}
	}
}
