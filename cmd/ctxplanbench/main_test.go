package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// TestReplayInvariants drives the full pipeline (ingest → recall reload → replay) over a
// small synthetic transcript of known shape and asserts the four invariants the measurement
// rests on: the O(1) bound holds, exact recall holds every turn, planned resident never
// exceeds the full transcript, and every forecast miss is SERVED (recoverable) — never lost.
// It is the test-form twin of the -selfcheck gate, so `go test ./cmd/ctxplanbench` (under
// WSL on this host) covers what an operator gets from `go run ./cmd/ctxplanbench -selfcheck`.
func TestReplayInvariants(t *testing.T) {
	tpath, cleanup := writeSelfcheckTranscript(t)
	defer cleanup()

	r, err := measureFile(context.Background(), tpath, 20, 6, ctxplan.ProbeOptions{}) // tight budget forces elision
	if err != nil {
		t.Fatalf("measureFile: %v", err)
	}
	if r.Turns < 3 {
		t.Fatalf("too few turns replayed (%d) — ingest/reload lost spans", r.Turns)
	}
	if r.PeakPlannedTok > r.Budget && r.OverBudgetTurns == 0 {
		t.Errorf("planned resident (%d) exceeded the budget (%d) without any pin-overrun turn — the O(1) bound broke", r.PeakPlannedTok, r.Budget)
	}
	if r.PlannedFaithfulTurns != r.Turns {
		t.Errorf("planned faithful %d/%d turns — exact recall did not hold every turn", r.PlannedFaithfulTurns, r.Turns)
	}
	if r.PlannedCumTok > r.LinearCumTok {
		t.Errorf("planned cum (%d) exceeded linear cum (%d) — planned resident never exceeds the full transcript", r.PlannedCumTok, r.LinearCumTok)
	}
	if r.CompactionLossTurns == 0 {
		t.Errorf("compaction never destroyed a fact — the tight budget did not force elision, so the contrast is vacuous")
	}
	// The recoverability contract: every fault must be served (a page fault, never a lost
	// fact). A refused fault is the gate holding on a sealed re-screen — not a lost fact,
	// but it must not happen on the benign synthetic transcript.
	if r.Served != r.Faults {
		t.Errorf("served (%d) != faults (%d) — a forecast miss was not recoverable", r.Served, r.Faults)
	}
	if r.FactsRecovered != r.Served {
		t.Errorf("factsRecovered (%d) != served (%d) — the quality tally drifted from the fault tally", r.FactsRecovered, r.Served)
	}
	if r.References == 0 {
		t.Errorf("zero references — the lexical back-reference signal did not fire on a transcript built to trigger it")
	}
	// Planning cost: this transcript is short (N < the default cap 128), so the bounded probe
	// sees every span and must score no more than the full scan, never overrun its cap, and
	// reach the IDENTICAL resident plan every turn at the identical cost (the bounded probe
	// changes the planner's COST, not its answer, whenever it sees the whole candidate set).
	if r.PeakProbe > r.MaxCandidates {
		t.Errorf("peak probe (%d) exceeded cap (%d) — the per-turn planner-compute bound broke", r.PeakProbe, r.MaxCandidates)
	}
	if r.ProbeCandCum > r.FullScanCandCum {
		t.Errorf("bounded probe scored more candidates (%d) than the full scan (%d) — impossible", r.ProbeCandCum, r.FullScanCandCum)
	}
	if r.PlanAgreeTurns != r.Turns {
		t.Errorf("bounded plan diverged on %d/%d turns at N<cap — the probe must reach the same plan when it sees every span", r.Turns-r.PlanAgreeTurns, r.Turns)
	}
	if r.ProbePlannedCum != r.FullScanPlannedCum {
		t.Errorf("bounded resident tokens (%d) != full-scan (%d) on identical plans — the cost win was not free", r.ProbePlannedCum, r.FullScanPlannedCum)
	}
}

// TestPlanningCostFlattenWhenBounded drives a transcript LONGER than a deliberately tiny probe
// cap, so the cap actually prunes candidates on the later turns — the regime the persistent
// index exists for. It asserts the three load-bearing properties of the cost flatten: the
// per-turn probe never overruns the cap (the Θ(c·N) bound holds), the cap genuinely bit on
// some turns (the test is not vacuous), and the cumulative probe work is strictly below the
// full-scan work (the Θ(N²)→Θ(c·N) flatten is real, not a no-op). Recoverability under the
// aggressive pruning is the existing served==faults contract: a pruned span the turn needs is
// a demand-page, never a lost fact.
func TestPlanningCostFlattenWhenBounded(t *testing.T) {
	tpath, cleanup := writeLongTranscript(t, 24)
	defer cleanup()

	// A tiny cap (4 candidates, 2-span recency tail) forces the bound to bite well before the
	// 24-turn transcript ends; a tight budget keeps only a couple spans resident so the probe's
	// recent+pinned set usually still covers the selection.
	probe := ctxplan.ProbeOptions{MaxCandidates: 4, RecencyWindow: 2}
	r, err := measureFile(context.Background(), tpath, 40, 4, probe)
	if err != nil {
		t.Fatalf("measureFile: %v", err)
	}
	if r.Turns < 8 {
		t.Fatalf("too few turns replayed (%d) — need a transcript longer than the cap to exercise the bound", r.Turns)
	}
	if r.MaxCandidates != 4 {
		t.Fatalf("effective cap = %d, want 4 (the explicit ProbeOptions did not take)", r.MaxCandidates)
	}
	if r.PeakProbe > r.MaxCandidates {
		t.Errorf("peak probe (%d) exceeded the cap (%d) — the per-turn planner-compute bound broke", r.PeakProbe, r.MaxCandidates)
	}
	if r.BoundedTurns == 0 {
		t.Errorf("the cap never bit (0 bounded turns) on a %d-turn transcript with cap 4 — the flatten test is vacuous", r.Turns)
	}
	if r.ProbeCandCum >= r.FullScanCandCum {
		t.Errorf("probe cum (%d) not below full-scan cum (%d) — the Θ(N²)→Θ(c·N) flatten did not happen", r.ProbeCandCum, r.FullScanCandCum)
	}
	if max := int64(r.MaxCandidates) * int64(r.Turns); r.ProbeCandCum > max {
		t.Errorf("probe cum (%d) exceeded the analytic Θ(c·N) ceiling c·N=%d — the bound is not actually linear", r.ProbeCandCum, max)
	}
	if r.BoundedAgreeTurns > r.BoundedTurns || r.PlanAgreeTurns > r.Turns {
		t.Errorf("agreement tallies inconsistent: bounded %d/%d, total %d/%d", r.BoundedAgreeTurns, r.BoundedTurns, r.PlanAgreeTurns, r.Turns)
	}
	// The honesty fence under aggressive pruning: a forecast/probe miss is served, never lost.
	if r.Served != r.Faults {
		t.Errorf("served (%d) != faults (%d) — a pruned span a turn referenced was not recoverable", r.Served, r.Faults)
	}
}

// writeLongTranscript materializes an n-line benign transcript with varied, partly
// back-referencing content — enough turns to push N past a small probe cap. Half the lines
// re-use an earlier "auth token rotation" phrase so the lexical reference signal fires (forcing
// real demand-page faults once the cap prunes the older spans), the rest are distinct noise.
func writeLongTranscript(t *testing.T, n int) (string, func()) {
	t.Helper()
	dir, err := os.MkdirTemp("", "ctxplanbench-long-*")
	if err != nil {
		t.Fatal(err)
	}
	lines := make([]string, 0, n)
	// A durable preference (pinned) anchors turn 0.
	lines = append(lines, `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"u0","content":"preference: always deploy from the release branch, never main"}]}}`)
	for i := 1; i < n; i++ {
		var content string
		if i%2 == 0 {
			content = fmt.Sprintf("auth token rotation runbook step %d mint roll revoke billing service", i)
		} else {
			content = fmt.Sprintf("unrelated build log compiled %d files quietly on iteration %d nothing relevant", i*7, i)
		}
		lines = append(lines, fmt.Sprintf(
			`{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u%d","name":"Bash"},{"type":"tool_result","tool_use_id":"u%d","content":%q}]}}`,
			i, i, content))
	}
	tpath := filepath.Join(dir, "long.jsonl")
	if err := os.WriteFile(tpath, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		os.RemoveAll(dir)
		t.Fatal(err)
	}
	return tpath, func() { os.RemoveAll(dir) }
}

// TestReplayLooseBudgetHoldsEverything checks the other end of the regime: a budget large
// enough to hold the whole transcript should make planned == linear (nothing elided) and
// produce zero faults — the degenerate "O(1) never engaged" case, which must still be
// faithful and never over-count a turn the workload did not contain.
func TestReplayLooseBudgetHoldsEverything(t *testing.T) {
	tpath, cleanup := writeSelfcheckTranscript(t)
	defer cleanup()

	r, err := measureFile(context.Background(), tpath, 100_000, 6, ctxplan.ProbeOptions{})
	if err != nil {
		t.Fatalf("measureFile: %v", err)
	}
	if r.PlannedFaithfulTurns != r.Turns {
		t.Errorf("planned faithful %d/%d turns at a loose budget", r.PlannedFaithfulTurns, r.Turns)
	}
	if r.Faults != 0 {
		t.Errorf("loose budget still faulted (%d) — everything should have been resident", r.Faults)
	}
	if r.CompactionLossTurns != 0 {
		t.Errorf("loose budget still elided (%d turns) — nothing should have been cold", r.CompactionLossTurns)
	}
}

// writeSelfcheckTranscript materializes the synthetic known-shape transcript the selfcheck
// uses, so the test and the -selfcheck gate cannot drift apart. It returns the path and a
// cleanup.
func writeSelfcheckTranscript(t *testing.T) (string, func()) {
	t.Helper()
	const (
		durable = `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"u0","content":"preference: always deploy from the release branch, never main"}]}}`
		read    = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u1","name":"Read"},{"type":"tool_result","tool_use_id":"u1","content":"auth token rotation runbook step one mint roll revoke"}]}}`
		noiseA  = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u2","name":"Bash"},{"type":"tool_result","tool_use_id":"u2","content":"build log compiled 412 files quietly on tuesday afternoon"}]}}`
		noiseB  = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u3","name":"Bash"},{"type":"tool_result","tool_use_id":"u3","content":"weather sunny light wind from the west nothing relevant"}]}}`
		edit    = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u4","name":"Edit"},{"type":"tool_result","tool_use_id":"u4","content":"applied the auth token rotation runbook edit to the billing service"}]}}`
	)
	dir, err := os.MkdirTemp("", "ctxplanbench-test-*")
	if err != nil {
		t.Fatal(err)
	}
	tpath := filepath.Join(dir, "selfcheck.jsonl")
	if err := os.WriteFile(tpath, []byte(strings.Join([]string{durable, read, noiseA, noiseB, edit}, "\n")), 0o644); err != nil {
		os.RemoveAll(dir)
		t.Fatal(err)
	}
	return tpath, func() { os.RemoveAll(dir) }
}
