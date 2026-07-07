package main

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// writeProgressLedger drops a .dispatch-runs/progress.jsonl under root with one row per
// closed_now value, the same shape `fak dispatch progress` appends and
// dispatchProgressFoldClosedHistory sums.
func writeProgressLedger(t *testing.T, root string, closedPerRow ...int) {
	t.Helper()
	dir := filepath.Join(root, dispatchProgressRunsDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var b []byte
	for _, n := range closedPerRow {
		b = append(b, []byte(`{"closed_now": `+strconv.Itoa(n)+"}\n")...)
	}
	if err := os.WriteFile(filepath.Join(dir, dispatchProgressLogName), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestNightIssueProgressMeasuredBit pins the fail-closed measured bit: no ledger reads as
// UNMEASURED (surface-only), a present ledger reads as measured — even when it sums to a
// real zero, which must stay distinct from "not measured".
func TestNightIssueProgressMeasuredBit(t *testing.T) {
	t.Run("no ledger is unmeasured", func(t *testing.T) {
		count, measured := nightIssueProgress(t.TempDir())
		if measured {
			t.Error("with no dispatch ledger the walk must be UNMEASURED (surface-only)")
		}
		if count != 0 {
			t.Errorf("unmeasured count must be 0, got %d", count)
		}
	})

	t.Run("present ledger folds closed_now", func(t *testing.T) {
		root := t.TempDir()
		writeProgressLedger(t, root, 12, 8, 5) // 25 closed across three rows
		count, measured := nightIssueProgress(root)
		if !measured {
			t.Fatal("a present ledger must read as measured")
		}
		if count != 25 {
			t.Errorf("closed fold: want 25 (12+8+5), got %d", count)
		}
	})

	t.Run("present-but-zero is a measured zero", func(t *testing.T) {
		root := t.TempDir()
		writeProgressLedger(t, root, 0, 0)
		count, measured := nightIssueProgress(root)
		if !measured {
			t.Error("a present ledger summing to zero is a REAL measured zero, not unmeasured")
		}
		if count != 0 {
			t.Errorf("want 0, got %d", count)
		}
	})
}

// TestIssueProgressWalkOptsGate pins the shell seam end-to-end: the opts only bind for a
// declared-target intent with a present ledger, and feeding them through superloop.Walk
// reproduces the pure gate (shortfall reds an otherwise-clean walk; a met target clears).
func TestIssueProgressWalkOptsGate(t *testing.T) {
	night := superloop.Super{Name: "run-the-night", Title: "t", Floor: 0, IssueTarget: 200,
		Members: []superloop.Member{{Kind: superloop.KindScorecard, Ref: "a"}}}
	cleanStatus := []superloop.MemberStatus{{Member: night.Members[0], Debt: 0, Measured: true}}

	t.Run("no target: no opts regardless of ledger", func(t *testing.T) {
		root := t.TempDir()
		writeProgressLedger(t, root, 50)
		noTarget := superloop.Super{Name: "t", Members: night.Members}
		if opts := issueProgressWalkOpts(root, noTarget); opts != nil {
			t.Errorf("an intent with no declared target must get nil opts, got %d", len(opts))
		}
	})

	t.Run("target but no ledger: surface-only (nil opts)", func(t *testing.T) {
		if opts := issueProgressWalkOpts(t.TempDir(), night); opts != nil {
			t.Errorf("no ledger must yield nil opts (surface-only), got %d", len(opts))
		}
		// And a Walk with those nil opts must not gate.
		rep := superloop.Walk(night, cleanStatus, issueProgressWalkOpts(t.TempDir(), night)...)
		if !rep.Satisfied || rep.IssueProgressMeasured {
			t.Errorf("surface-only walk must be satisfied+unmeasured, got satisfied=%v measured=%v", rep.Satisfied, rep.IssueProgressMeasured)
		}
	})

	t.Run("target + short ledger reds the walk", func(t *testing.T) {
		root := t.TempDir()
		writeProgressLedger(t, root, 120) // 120 < 200
		rep := superloop.Walk(night, cleanStatus, issueProgressWalkOpts(root, night)...)
		if rep.Satisfied {
			t.Error("a short ledger must keep the declared-target walk unsatisfied")
		}
		if rep.IssueProgressed != 120 || rep.IssueShortfall != 80 {
			t.Errorf("want progressed=120 shortfall=80, got %d/%d", rep.IssueProgressed, rep.IssueShortfall)
		}
		if rep.Finding != "superloop_issue_shortfall" {
			t.Errorf("want superloop_issue_shortfall, got %q", rep.Finding)
		}
	})

	t.Run("target met clears the gate", func(t *testing.T) {
		root := t.TempDir()
		writeProgressLedger(t, root, 150, 60) // 210 >= 200
		rep := superloop.Walk(night, cleanStatus, issueProgressWalkOpts(root, night)...)
		if !rep.Satisfied {
			t.Errorf("a ledger meeting the headline must clear the gate; reason=%q", rep.Reason)
		}
		if rep.IssueShortfall != 0 {
			t.Errorf("want shortfall 0, got %d", rep.IssueShortfall)
		}
	})
}
