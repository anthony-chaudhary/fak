package dojocal

import (
	"strings"
	"testing"
	"time"
)

func TestRankCandidatesNoThrash(t *testing.T) {
	now := mustSelectTime(t, "2026-06-29T00:00:00Z")
	a := Recal{Lever: "resume-posture", Metric: "cold_write_share", Kind: RecalibrateKind, CalibErr: 0.80}
	b := Recal{Lever: "compaction", Metric: "token_shed_ratio", Kind: HarvestKind, CalibErr: 0.40}
	rows := []JournalRow{{
		Schema: JournalSchema, Tick: 1, Lever: a.Lever, Metric: a.Metric,
		Kind: a.Kind, Decision: "KEEP", GeneratedAt: "2026-06-28T00:00:00Z", Date: "2026-06-28",
	}}

	ranked := RankCandidates([]Recal{a, b}, rows, SelectOptions{Now: now, RecheckDays: 7})
	if len(ranked) != 2 {
		t.Fatalf("ranked = %d, want 2", len(ranked))
	}
	if ranked[0].Candidate.Lever != b.Lever {
		t.Fatalf("never-touched candidate should outrank fresh touched cell, got %+v", ranked[0])
	}
	if !ranked[1].Saturated {
		t.Fatalf("fresh touched cell must be saturated to prevent thrash: %+v", ranked[1])
	}
	next, ok := NextCandidate(ranked)
	if !ok || next.Candidate.Lever != b.Lever {
		t.Fatalf("NextCandidate = %+v ok=%v, want compaction", next, ok)
	}

	rows = append(rows, JournalRow{
		Schema: JournalSchema, Tick: 2, Lever: b.Lever, Metric: b.Metric,
		Kind: b.Kind, Decision: "REVERT", GeneratedAt: now.Format(time.RFC3339), Date: "2026-06-29",
	})
	ranked = RankCandidates([]Recal{a, b}, rows, SelectOptions{Now: now, RecheckDays: 7})
	if _, ok := NextCandidate(ranked); ok {
		t.Fatalf("all cells were touched inside the recheck window; next must stop saturated: %+v", ranked)
	}
	wake := ScheduleWakeup(ranked, now)
	if wake.Pending || wake.DelayS <= 0 || !strings.Contains(wake.Reason, "fresh") {
		t.Fatalf("wake = %+v, want delayed saturated wake", wake)
	}
}

// TestRankCandidatesUndateableTouchStaysEligible pins the freshness boundary that
// keeps the calibration loop from going stale: a cell whose only journal touch
// carries NO parseable timestamp (empty/corrupt generated_at AND date) cannot be
// proven fresh, so it must stay eligible (re-checked), never be parked as
// "saturated/fresh" forever. Before the fix, an undateable touch left Staleness at
// 0, tripped the Staleness<1 saturation branch, and froze the cell as fresh — the
// loop silently stopped recalibrating it. A NEVER-touched cell of the same kind is
// the control: it must still be runnable too.
func TestRankCandidatesUndateableTouchStaysEligible(t *testing.T) {
	now := mustSelectTime(t, "2026-06-29T00:00:00Z")
	undateable := Recal{Lever: "resume-posture", Metric: "cold_write_share", Kind: RecalibrateKind, CalibErr: 0.80}
	control := Recal{Lever: "compaction", Metric: "token_shed_ratio", Kind: RecalibrateKind, CalibErr: 0.40}
	// The only touch for `undateable` has neither a generated_at nor a date that
	// parseStamp can read, so its age is unknowable.
	rows := []JournalRow{{
		Schema: JournalSchema, Tick: 1, Lever: undateable.Lever, Metric: undateable.Metric,
		Kind: undateable.Kind, Decision: "KEEP", GeneratedAt: "", Date: "not-a-date",
	}}

	ranked := RankCandidates([]Recal{undateable, control}, rows, SelectOptions{Now: now, RecheckDays: 7})
	byCell := map[string]ScoredCell{}
	for _, s := range ranked {
		byCell[s.Candidate.Lever+"/"+s.Candidate.Metric] = s
	}

	und := byCell["resume-posture/cold_write_share"]
	if und.Saturated {
		t.Fatalf("undateable touch must NOT saturate the cell (cannot prove freshness): %+v", und)
	}
	if und.Staleness != 1 {
		t.Fatalf("undateable touch must fail toward stale (Staleness=1), got %v: %+v", und.Staleness, und)
	}
	if und.AgeDays >= 0 {
		t.Fatalf("undateable touch must leave AgeDays<0 (no provable age), got %v", und.AgeDays)
	}
	if !strings.Contains(und.Reason, "undateable") {
		t.Fatalf("reason should explain the undateable touch, got %q", und.Reason)
	}

	// The loop must still have runnable work — the undateable cell is eligible.
	next, ok := NextCandidate(ranked)
	if !ok {
		t.Fatalf("an undateable-touch cell must keep the loop runnable, got no next candidate: %+v", ranked)
	}
	// Both candidates carry real value; whichever ranks first, the undateable cell
	// must never be the saturated one that stops the loop.
	if next.Saturated {
		t.Fatalf("NextCandidate must be unsaturated, got %+v", next)
	}
}

func TestTreeChangedWithinDeclaredPaths(t *testing.T) {
	declared := []string{"internal/resume/", "cmd/fak/dojo.go"}
	if !TreeChangedWithin([]string{"internal/resume/backtest.go", "cmd/fak/dojo.go"}, declared) {
		t.Fatal("expected declared resume + dojo paths to pass")
	}
	if TreeChangedWithin([]string{"internal/dojo/claims.go"}, declared) {
		t.Fatal("claims.go must not pass a REPROJECT path allow-list")
	}
	if TreeChangedWithin(nil, declared) {
		t.Fatal("empty changed-path set must fail closed")
	}
	if TreeChangedWithin([]string{"internal/resume/backtest.go"}, nil) {
		t.Fatal("empty declared path set must fail closed")
	}
}

func TestJournalTrendCountsKeepRoutesAndFloorEscalate(t *testing.T) {
	rows := []JournalRow{
		{Schema: JournalSchema, Tick: 1, Lever: "resume-posture", Metric: "cold_write_share", Kind: RecalibrateKind, Decision: "KEEP", Kept: true},
		{Schema: JournalSchema, Tick: 2, Lever: "resume-posture", Metric: "posture_accuracy", Kind: ReprojectKind, Decision: "REVERT", AgentArm: true},
		{Schema: JournalSchema, Tick: 3, Lever: "vcache-warmth", Metric: "false_warm_rate", Kind: RouteFloor, Decision: "ESCALATE", AgentArm: true},
		{Schema: JournalSchema, Tick: 4, Lever: "compaction", Metric: "token_shed_ratio", Kind: HarvestKind, Decision: "REVERT", AgentArm: true},
	}
	tr := FoldTrend(rows, mustSelectTime(t, "2026-06-29T00:00:00Z"))
	if tr.Keep != 1 || tr.MechanicalKeep != 1 || tr.Revert != 2 || tr.Escalate != 1 {
		t.Fatalf("trend decision counts wrong: %+v", tr)
	}
	if tr.AgentRoutes != 3 || tr.ReprojectRoutes != 1 || tr.HarvestRoutes != 1 || tr.FloorEscalates != 1 {
		t.Fatalf("trend route counts wrong: %+v", tr)
	}
	if tr.Latest == nil || tr.Latest.Tick != 4 {
		t.Fatalf("latest row = %+v, want tick 4", tr.Latest)
	}
	if !strings.Contains(MarshalTrendText(tr), "KEEP 1") {
		t.Fatalf("trend text missing keep count: %s", MarshalTrendText(tr))
	}
}

func mustSelectTime(t *testing.T, s string) time.Time {
	t.Helper()
	tm, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse time %q: %v", s, err)
	}
	return tm.UTC()
}
