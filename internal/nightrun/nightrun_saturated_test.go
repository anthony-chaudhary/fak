package nightrun

import (
	"context"
	"strings"
	"testing"
	"time"
)

// nightrun_saturated_test.go covers the first-class SATURATED selector verdict (#1138):
// a feasible, auto-runnable datum that is already collected and still fresh is marked
// Saturated, and when EVERY feasible task is saturated the run loop stops with a
// "saturated — … next datum blocked on: <names>" reason built from the infeasible tasks'
// why-strings, instead of re-firing a settled measurement.

// freshLedger builds a collected row for taskID on box at `now`, so the task scores fresh
// (Staleness==0) and therefore Saturated.
func freshLedger(taskID, box string, now time.Time) []CollectRow {
	return []CollectRow{{
		Schema:      CollectSchema,
		Date:        now.UTC().Format("2006-01-02"),
		Box:         box,
		TaskID:      taskID,
		Outcome:     string(OutcomeCollected),
		GeneratedAt: now.UTC().Format(time.RFC3339),
	}}
}

// TestScoredSaturatedFlag pins the per-task Saturated verdict: a feasible auto-runnable
// datum collected today is saturated; a never-collected one is not; an overdue one is
// not; a Manual recipe is never saturated; an infeasible one is never saturated.
func TestScoredSaturatedFlag(t *testing.T) {
	box := "ci"
	caps := Capabilities{Box: box, GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-28T00:00:00Z")

	fresh := Task{ID: "bench-fresh", Value: ValueSmoke, Run: "echo a", RecheckDays: 14}
	never := Task{ID: "bench-never", Value: ValueSmoke, Run: "echo b"}
	overdue := Task{ID: "bench-overdue", Value: ValueSmoke, Run: "echo c", RecheckDays: 1}
	manual := Task{ID: "witness-manual", Value: ValueFrontier, Requires: []Requirement{ReqCUDA, ReqWeights}, Run: "run.sh <model>", Manual: true}
	infeasible := Task{ID: "witness-metal", Value: ValueWitness, Requires: []Requirement{ReqMetal}, Run: "echo m"}

	ledger := freshLedger("bench-fresh", box, now)
	// An overdue collection: 30 days old against a 1-day recheck → Staleness clamps to 1.
	old := mustTime(t, "2026-05-28T00:00:00Z")
	ledger = append(ledger, CollectRow{
		Schema: CollectSchema, Date: old.Format("2006-01-02"), Box: box,
		TaskID: "bench-overdue", Outcome: string(OutcomeCollected), GeneratedAt: old.Format(time.RFC3339),
	})

	ranked := Rank([]Task{fresh, never, overdue, manual, infeasible}, caps, ledger, now)
	byID := map[string]Scored{}
	for _, s := range ranked {
		byID[s.Task.ID] = s
	}
	if !byID["bench-fresh"].Saturated {
		t.Errorf("a feasible auto-runnable datum collected today must be Saturated, got %+v", byID["bench-fresh"])
	}
	if byID["bench-never"].Saturated {
		t.Error("a never-collected task must NOT be Saturated (it is the most novel datum)")
	}
	if byID["bench-overdue"].Saturated {
		t.Errorf("an overdue task (Staleness=%.2f) must NOT be Saturated", byID["bench-overdue"].Staleness)
	}
	if byID["witness-manual"].Saturated {
		t.Error("a Manual recipe is never auto-collectable, so it must never be Saturated")
	}
	if byID["witness-metal"].Saturated {
		t.Error("an infeasible task must never be Saturated")
	}
}

// TestStopReasonSaturated pins the loop-level SATURATED verdict: when every feasible
// auto-runnable datum is already fresh, the loop stops with a saturated reason that
// names the external conditions the next datum waits on (the infeasible tasks' reasons),
// instead of the generic "attempted the whole queue" / "nothing to collect" messages.
func TestStopReasonSaturated(t *testing.T) {
	box := "cuda-box"
	caps := Capabilities{Box: box, GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-28T00:00:00Z")

	tasks := []Task{
		{ID: "bench-offline", Value: ValueSmoke, Run: "echo ok", RecheckDays: 14},                    // feasible, will be fresh
		{ID: "witness-metal", Value: ValueWitness, Requires: []Requirement{ReqMetal}, Run: "echo m"}, // infeasible here
		{ID: "witness-dataset", Value: ValueWitness, Requires: []Requirement{ReqDataset}, Run: "echo d"},
	}
	ranked := Rank(tasks, caps, freshLedger("bench-offline", box, now), now)

	reason := stopReason(ranked, 0, true)
	if !strings.HasPrefix(reason, "saturated") {
		t.Fatalf("stop reason must be the SATURATED verdict, got %q", reason)
	}
	if !strings.Contains(reason, "blocked on:") {
		t.Errorf("the saturated reason must name the blockers, got %q", reason)
	}
	// The blocker text comes from the infeasible tasks' Satisfies why-strings, with the
	// "not feasible here — " prefix stripped.
	if !strings.Contains(reason, "Apple GPU") || !strings.Contains(reason, "dataset") {
		t.Errorf("the saturated reason must list the missing-capability blockers (metal, dataset), got %q", reason)
	}
	if strings.Contains(reason, "not feasible here") {
		t.Errorf("the blocker list must strip the per-task prefix, got %q", reason)
	}
}

// TestStopReasonNotSaturatedWhenWorkRemains pins the negative: while a feasible
// auto-runnable datum is still gatherable (never-collected here), the loop must NOT claim
// saturation — it keeps collecting.
func TestStopReasonNotSaturatedWhenWorkRemains(t *testing.T) {
	box := "cuda-box"
	caps := Capabilities{Box: box, GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-28T00:00:00Z")
	tasks := []Task{
		{ID: "bench-fresh", Value: ValueSmoke, Run: "echo a", RecheckDays: 14},
		{ID: "bench-never", Value: ValueSmoke, Run: "echo b"}, // never collected → still gatherable
	}
	ranked := Rank(tasks, caps, freshLedger("bench-fresh", box, now), now)
	if reason := stopReason(ranked, 1, true); strings.HasPrefix(reason, "saturated") {
		t.Errorf("must not report saturated while a never-collected datum remains, got %q", reason)
	}
}

// TestRunLoopReportsSaturatedAfterDraining drives the whole loop: one feasible offline
// task, an infeasible HW-gated witness. After the loop collects the offline task once,
// the next iteration finds it fresh (saturated) and stops with the SATURATED verdict —
// the loop does not re-fire the settled measurement.
func TestRunLoopReportsSaturatedAfterDraining(t *testing.T) {
	box := "cuda-box"
	caps := Capabilities{Box: box, GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-28T00:00:00Z")
	tasks := []Task{
		{ID: "bench-offline", Value: ValueCoverage, Run: "echo ok", RecheckDays: 14},
		{ID: "witness-metal", Value: ValueFrontier, Requires: []Requirement{ReqMetal}, Run: "echo m"},
	}
	var ledger []CollectRow
	runs := 0
	summary, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: caps, Tasks: tasks, Now: now,
		Apply: true, Loop: true, Max: 0,
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
	if runs != 1 {
		t.Errorf("the offline datum must be collected exactly once (not re-fired once fresh), ran %d", runs)
	}
	if !strings.HasPrefix(summary.StopReason, "saturated") {
		t.Errorf("after draining the feasible queue the loop must stop SATURATED, got %q", summary.StopReason)
	}
}
