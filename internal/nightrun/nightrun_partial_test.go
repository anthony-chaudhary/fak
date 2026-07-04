package nightrun

import (
	"context"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// nightrun_partial_test.go covers the PARTIAL freshness tier (#2383): a timed-out
// task whose captured partial output still parses a real headline number is
// banked as OutcomePartial instead of discarding the parse the way a plain
// OutcomeTimeout does. A PARTIAL datum is admitted as weak-but-real evidence: it
// suppresses an immediate re-pick (so a chronically-slow benchmark does not burn a
// fresh full budget every single night) but — unlike a clean collect — it ages out
// on a much shorter horizon so a genuine re-measure still happens eventually.

// TestExecTaskTimeoutWithParsableOutputBanksPartial pins the parse-on-timeout path:
// when the wall-clock budget fires but the partial output already contains a
// parseable headline number, execTask reports OutcomePartial with that number —
// never a bare OutcomeTimeout with an empty number.
func TestExecTaskTimeoutWithParsableOutputBanksPartial(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses sh/sleep; runs on the Linux test path (native windows go test is blocked anyway)")
	}
	art := filepath.Join(t.TempDir(), "out.log")
	task := Task{ID: "slow-partial", Run: "echo 'decode: 12.5 tok/s so far'; sleep 30", TimeoutSec: 1}
	outcome, number, _, err := DefaultExecutor(context.Background(), task, art)
	if outcome != OutcomePartial {
		t.Fatalf("outcome = %q, want %q (err=%v)", outcome, OutcomePartial, err)
	}
	if number != "12.5 tok/s" {
		t.Errorf("number = %q, want the parsed partial number %q", number, "12.5 tok/s")
	}
	if err == nil {
		t.Error("a timed-out run must still report an error naming the budget, even when a partial number was banked")
	}
}

// TestExecTaskTimeoutWithoutParsableOutputStaysTimeout pins the negative: a
// timed-out run whose partial output has no recognizable headline number stays a
// plain OutcomeTimeout with an empty number — parse-on-timeout never fabricates a
// number that was not actually in the captured output.
func TestExecTaskTimeoutWithoutParsableOutputStaysTimeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses sh/sleep; runs on the Linux test path (native windows go test is blocked anyway)")
	}
	art := filepath.Join(t.TempDir(), "out.log")
	task := Task{ID: "slow-empty", Run: "echo started; sleep 30", TimeoutSec: 1}
	outcome, number, _, err := DefaultExecutor(context.Background(), task, art)
	if outcome != OutcomeTimeout {
		t.Fatalf("outcome = %q, want %q (err=%v)", outcome, OutcomeTimeout, err)
	}
	if number != "" {
		t.Errorf("number = %q, want empty — no headline number was ever in the output", number)
	}
}

// TestOutcomePartialInClosedVocabulary pins the enum admission and the
// CollectedOutcome boundary: OutcomePartial is a valid ledger outcome, but it must
// NEVER count as a clean collect (that would let a partial masquerade as a real
// collection and permanently suppress a re-measure).
func TestOutcomePartialInClosedVocabulary(t *testing.T) {
	if !IsValidOutcome(OutcomePartial) {
		t.Error("OutcomePartial must be a member of the closed outcome vocabulary")
	}
	if CollectedOutcome(OutcomePartial) {
		t.Error("CollectedOutcome(OutcomePartial) must be false — a partial must not masquerade as a clean collect")
	}
}

// partialLedger builds a PARTIAL row for taskID on box, generated `ageDays` before
// now, with a non-empty parsed number.
func partialLedger(taskID, box string, now time.Time, ageDaysAgo float64) []CollectRow {
	gen := now.Add(-time.Duration(ageDaysAgo * float64(24*time.Hour)))
	return []CollectRow{{
		Schema:      CollectSchema,
		Date:        gen.UTC().Format("2006-01-02"),
		Box:         box,
		TaskID:      taskID,
		Outcome:     string(OutcomePartial),
		Number:      "12.5 tok/s",
		GeneratedAt: gen.UTC().Format(time.RFC3339),
	}}
}

// TestScoredPartialSuppressesSameNightRepick pins the same-night (very-next-pick)
// suppression: a task whose most recent row is a fresh PARTIAL (well inside its
// short partialRecheckDays horizon) must be Saturated — Rank does not re-run it
// from zero on the very next attempt, banking the work the timed-out run already
// did.
func TestScoredPartialSuppressesSameNightRepick(t *testing.T) {
	box := "ci"
	caps := Capabilities{Box: box, GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-28T00:00:00Z")
	task := Task{ID: "bench-partial", Value: ValueCoverage, Run: "echo a", RecheckDays: 14}

	ledger := partialLedger("bench-partial", box, now, 0) // generated right now
	ranked := Rank([]Task{task}, caps, ledger, now)
	if len(ranked) != 1 {
		t.Fatalf("want 1 scored task, got %d", len(ranked))
	}
	s := ranked[0]
	if !s.Saturated {
		t.Errorf("a fresh PARTIAL row must suppress the very next re-pick (Saturated), got %+v", s)
	}
	if s.LastCollected != "" {
		t.Errorf("a PARTIAL row must never populate LastCollected (that is the clean-collect field), got %q", s.LastCollected)
	}

	// The run loop itself must not re-fire the task while the partial is fresh.
	runs := 0
	_, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: caps, Tasks: []Task{task}, Now: now,
		Apply: true, Loop: true,
		ReadLedger: func() []CollectRow { return ledger },
		AppendRow:  func(r CollectRow) error { ledger = append(ledger, r); return nil },
		Executor: func(_ context.Context, _ Task, _ string) (Outcome, string, time.Duration, error) {
			runs++
			return OutcomeCollected, "", time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if runs != 0 {
		t.Errorf("a fresh PARTIAL row must suppress the run loop from re-running the task, ran %d time(s)", runs)
	}
}

// TestScoredPartialRepickedOnLongerHorizon pins the negative: unlike a clean
// collect, a PARTIAL row must NOT permanently suppress a re-measure — once its
// (short) partialRecheckDays horizon has elapsed, the task is un-saturated again
// and the run loop picks it up.
func TestScoredPartialRepickedOnLongerHorizon(t *testing.T) {
	box := "ci"
	caps := Capabilities{Box: box, GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-28T00:00:00Z")
	task := Task{ID: "bench-partial-old", Value: ValueCoverage, Run: "echo a", RecheckDays: 14}
	// partialRecheckDays() for RecheckDays=14 is 14/4=3 days; 10 days ago is well past it.
	ledger := partialLedger("bench-partial-old", box, now, 10)

	ranked := Rank([]Task{task}, caps, ledger, now)
	if len(ranked) != 1 {
		t.Fatalf("want 1 scored task, got %d", len(ranked))
	}
	if ranked[0].Saturated {
		t.Errorf("a PARTIAL row past its short recheck horizon must NOT be Saturated (it must be re-picked), got %+v", ranked[0])
	}

	runs := 0
	summary, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: caps, Tasks: []Task{task}, Now: now,
		Apply: true, Loop: true,
		ReadLedger: func() []CollectRow { return ledger },
		AppendRow:  func(r CollectRow) error { ledger = append(ledger, r); return nil },
		Executor: func(_ context.Context, _ Task, _ string) (Outcome, string, time.Duration, error) {
			runs++
			return OutcomeCollected, "17 tok/s", time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if runs != 1 {
		t.Errorf("a stale PARTIAL row must be re-picked and re-run, ran %d time(s)", runs)
	}
	if len(summary.Runs) != 1 || summary.Runs[0].Outcome != OutcomeCollected {
		t.Errorf("want exactly one collected run, got %+v", summary.Runs)
	}
}
