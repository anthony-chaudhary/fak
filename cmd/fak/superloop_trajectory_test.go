package main

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/superloop"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// trajSeededCollector builds a collector with the trajctl open-objective curves
// pre-folded, so the enumerator can be exercised without an on-disk ledger.
func trajSeededCollector(objs ...trajctl.ObjectiveCurve) *superloopCollector {
	return &superloopCollector{
		trajLoaded:        true,
		trajLedgerPresent: true, // simulate a real, readable ledger (present, some/zero open)
		trajCurve:         trajctl.CurveReport{Schema: trajctl.CurveSchema, Objectives: objs},
	}
}

// TestCollectTrajectoryEnumeratesWorstFirst is the trajectory-member adapter witness
// (issue #2563): the single improve-trajectory KindTrajectory member enumerates into one
// status per OPEN objective, each weighed by its curve-signal debt, and the walk folds
// them worst-first (DETOUR_OVERRUN ahead of STALL) while dropping the clean HEALTHY one.
func TestCollectTrajectoryEnumeratesWorstFirst(t *testing.T) {
	s, ok := superloop.Lookup("improve-trajectory")
	if !ok {
		t.Fatal("improve-trajectory not registered")
	}
	c := trajSeededCollector(
		trajctl.ObjectiveCurve{ObjectiveID: "stall-obj", Signal: trajctl.SignalStall, Status: trajctl.StatusActive},
		trajctl.ObjectiveCurve{ObjectiveID: "healthy-obj", Signal: trajctl.SignalHealthy, Status: trajctl.StatusActive},
		trajctl.ObjectiveCurve{ObjectiveID: "detour-obj", Signal: trajctl.SignalDetourOverrun, Status: trajctl.StatusPaused},
	)
	statuses := c.collect(s, map[string]bool{s.Name: true})
	if len(statuses) != 3 {
		t.Fatalf("trajectory member must enumerate one status per open objective, got %d", len(statuses))
	}
	byRef := map[string]superloop.MemberStatus{}
	for _, st := range statuses {
		if st.Member.Kind != superloop.KindTrajectory {
			t.Errorf("enumerated status kind = %q, want %q", st.Member.Kind, superloop.KindTrajectory)
		}
		if !st.Measured {
			t.Errorf("objective %q must be measured (its curve was folded)", st.Member.Ref)
		}
		byRef[st.Member.Ref] = st
	}
	if byRef["detour-obj"].Debt != 3 || byRef["stall-obj"].Debt != 1 || byRef["healthy-obj"].Debt != 0 {
		t.Errorf("debt must map to signal severity, got detour=%d stall=%d healthy=%d",
			byRef["detour-obj"].Debt, byRef["stall-obj"].Debt, byRef["healthy-obj"].Debt)
	}

	rep := superloop.Walk(s, statuses)
	if rep.Satisfied {
		t.Error("open off-course objectives must keep the trajectory intent unsatisfied")
	}
	if len(rep.Worklist) == 0 || rep.Worklist[0].Member.Ref != "detour-obj" {
		t.Errorf("worst-first head must be the DETOUR_OVERRUN objective, got %+v", rep.Worklist)
	}
	for _, w := range rep.Worklist {
		if w.Member.Ref == "healthy-obj" {
			t.Error("a HEALTHY objective must be dropped from the worklist")
		}
	}
}

// TestCollectTrajectoryEmptyIsSatisfied pins the clean case: no open objective (an empty
// ledger, or every objective closed) folds to a single measured, zero-debt status so the
// intent reads SATISFIED — an on-course fleet is nothing to enter, not an unmeasured gap.
func TestCollectTrajectoryEmptyIsSatisfied(t *testing.T) {
	s, _ := superloop.Lookup("improve-trajectory")
	c := trajSeededCollector() // no open objectives
	statuses := c.collect(s, map[string]bool{s.Name: true})
	if len(statuses) != 1 || !statuses[0].Measured || statuses[0].Debt != 0 {
		t.Fatalf("an empty trajectory must fold to one measured clean status, got %+v", statuses)
	}
	rep := superloop.Walk(s, statuses)
	if !rep.Satisfied || len(rep.Worklist) != 0 {
		t.Errorf("no open objectives → satisfied with empty worklist, got satisfied=%v worklist=%d", rep.Satisfied, len(rep.Worklist))
	}
}

// TestCollectTrajectoryNoLedgerUnmeasured pins the honesty fence: with NO trajctl
// ledger in the workspace, trajectory health cannot be read, so the member folds to a
// single UNMEASURED status — never silently clean — which keeps the intent unsatisfied.
func TestCollectTrajectoryNoLedgerUnmeasured(t *testing.T) {
	s, _ := superloop.Lookup("improve-trajectory")
	c := newSuperloopCollector(t.TempDir()) // empty workspace, no ledger
	statuses := c.collect(s, map[string]bool{s.Name: true})
	if len(statuses) != 1 || statuses[0].Measured {
		t.Fatalf("an absent ledger must fold to one UNMEASURED status, got %+v", statuses)
	}
	if rep := superloop.Walk(s, statuses); rep.Satisfied {
		t.Error("trajectory health cannot read satisfied with no ledger to measure")
	}
}

// TestCollectTrajectoryReadsLedgerEndToEnd exercises the whole shell wiring: a real
// trajctl ledger under DefaultLedgerRel is read, folded, and enumerated worst-first. The
// declining W3 progress curve makes the objective DRIFT, so the walk enters it.
func TestCollectTrajectoryReadsLedgerEndToEnd(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(root, filepath.FromSlash(trajctl.DefaultLedgerRel))
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(trajctl.Objective{
		ID: "drift-e2e", Statement: "ship the widget", Status: trajctl.StatusActive,
	})); err != nil {
		t.Fatalf("seed objective: %v", err)
	}
	// Two declining witnessed-commit-progress points → a DRIFT curve.
	for i, v := range []float64{0.6, 0.2} {
		if err := trajctl.Append(ledger, trajctl.ScoreRecord(trajctl.ScoreRow{
			ObjectiveID: "drift-e2e", Value: v, Method: trajctl.CommitScorerMethod,
			Version: "1", Witness: trajctl.W3, UnixMillis: int64(i + 1),
		})); err != nil {
			t.Fatalf("seed score %d: %v", i, err)
		}
	}

	s, _ := superloop.Lookup("improve-trajectory")
	c := newSuperloopCollector(root)
	statuses := c.collect(s, map[string]bool{s.Name: true})
	if len(statuses) != 1 {
		t.Fatalf("one open objective should enumerate to one status, got %d", len(statuses))
	}
	st := statuses[0]
	if st.Member.Ref != "drift-e2e" || !st.Measured {
		t.Fatalf("status should name the ledger objective and be measured, got %+v", st)
	}
	if st.Debt != trajctl.SignalDebt(trajctl.SignalDrift) {
		t.Errorf("a declining progress curve should read DRIFT (debt %d), got debt %d",
			trajctl.SignalDebt(trajctl.SignalDrift), st.Debt)
	}
	rep := superloop.Walk(s, statuses)
	if rep.Satisfied {
		t.Error("a drifting objective must keep the trajectory intent unsatisfied")
	}
}
