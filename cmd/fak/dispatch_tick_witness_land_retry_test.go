package main

// #3613: the dispatch land seam must CONSUME a refused optimistic land instead of
// discarding it — a LAND_READBACK_MISMATCH race refusal retries the land (bounded)
// BEFORE the reap destroys the worktree (the only copy of the diff), while a
// deterministic refusal (red verify) goes straight to the reap, and an exhausted
// bound still reaps (the pre-#3613 fail-open final resort).

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
	return workerworktree.Result{OK: false, Applied: true, Committed: true,
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

// TestRefusedLandGivesUpAfterBoundThenReaps pins the bound: an always-race-refused
// land is attempted exactly dispatchLandRefusedAttempts times, then the reap still
// runs — bounded retry, never an unbounded loop, never a leaked worktree.
func TestRefusedLandGivesUpAfterBoundThenReaps(t *testing.T) {
	if dispatchLandRefusedAttempts < 2 {
		t.Fatalf("the #3613 bound must allow at least one retry, got %d", dispatchLandRefusedAttempts)
	}
	log := withLandRetrySeams(t, []workerworktree.Result{raceRefusedResult()})
	landAndReapWorkerWorktreeDefault("/root", "/wt/fak-worker-wt-cmd-abc", "base", []string{"cmd"})
	want := []string{}
	for i := 0; i < dispatchLandRefusedAttempts; i++ {
		want = append(want, "land")
	}
	want = append(want, "reap:/wt/fak-worker-wt-cmd-abc")
	assertLandRetryLog(t, *log, want)
}

// TestDeterministicRefusalNeverRetries pins the guard rail: a red-verify refusal
// is deterministic — replaying it cannot change the verdict — so the seam must
// land exactly ONCE and reap, never burn retries on it.
func TestDeterministicRefusalNeverRetries(t *testing.T) {
	log := withLandRetrySeams(t, []workerworktree.Result{
		{OK: false, Reason: "worktree verify failed, refusing to land: go build ./... failed: boom"},
	})
	landAndReapWorkerWorktreeDefault("/root", "/wt/fak-worker-wt-cmd-abc", "base", []string{"cmd"})
	assertLandRetryLog(t, *log, []string{"land", "reap:/wt/fak-worker-wt-cmd-abc"})
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
