package sessionrecovery

import (
	"strings"
	"testing"
	"time"
)

func observedSession(id, turnStatus, startedAt string, guardedTrees int) Session {
	trees := make([]ProcessTree, guardedTrees)
	for i := range trees {
		trees[i] = ProcessTree{RootPID: i + 10, HasGuard: true}
	}
	return Session{
		Thread:       &Thread{ID: id, Source: "interactive_tui", CWD: `C:\work\fak`},
		LatestTurn:   &Turn{Status: turnStatus, StartedAt: startedAt},
		GuardReceipt: &GuardReceipt{RecordedAt: "2026-08-18T01:01:00Z"},
		ProcessTrees: trees,
	}
}

func TestObserveEnforcesExactlyOneGuardedTree(t *testing.T) {
	before := InventoryReport{Sessions: []Session{observedSession("t1", "inProgress", "2026-08-18T01:00:00Z", 0)}}
	prior := Result{ThreadID: "t1", Status: "launched_unproven"}
	for _, tc := range []struct {
		name  string
		trees int
		want  string
	}{
		{name: "none", trees: 0, want: "launched_unproven"},
		{name: "one", trees: 1, want: "productive"},
		{name: "duplicate", trees: 2, want: "cardinality_failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			after := InventoryReport{Sessions: []Session{observedSession("t1", "inProgress", "2026-08-18T02:00:00Z", tc.trees)}}
			got := Observe(before, after, prior)
			if got.Status != tc.want || got.GuardedProcessTrees != tc.trees {
				t.Fatalf("result=%+v", got)
			}
		})
	}
}

func TestObserveRecognizesCompletedFreshTurn(t *testing.T) {
	before := InventoryReport{Sessions: []Session{observedSession("t1", "inProgress", "2026-08-18T01:00:00Z", 0)}}
	after := InventoryReport{Sessions: []Session{observedSession("t1", "completed", "2026-08-18T02:00:00Z", 0)}}
	got := Observe(before, after, Result{ThreadID: "t1", Status: "launched_unproven"})
	if got.Status != "completed" {
		t.Fatalf("result=%+v", got)
	}
}

func TestObserveTreatsFailedFreshTurnAsVerificationFailure(t *testing.T) {
	before := InventoryReport{Sessions: []Session{observedSession("t1", "inProgress", "2026-08-18T01:00:00Z", 0)}}
	after := InventoryReport{Sessions: []Session{observedSession("t1", "failed", "2026-08-18T02:00:00Z", 1)}}
	got := Observe(before, after, Result{ThreadID: "t1", Status: "launched_unproven"})
	if got.Status != "verification_failed" || got.Remediation == "" {
		t.Fatalf("result=%+v", got)
	}
}

func TestSummaryCountsAndRemediation(t *testing.T) {
	report := InventoryReport{Sessions: []Session{observedSession("active", "inProgress", "2026-08-18T01:00:00Z", 1)}}
	requests := []Request{
		{ThreadID: "candidate", Status: "candidate", CWD: `C:\work\fak`},
		{ThreadID: "receipt", Status: "already_receipted", ReceiptPath: `C:\r.json`},
	}
	got := NewSummary("preview", report, requests, time.Unix(1, 0))
	if got.Schema != SummarySchema || got.Counts.Discovered != 1 || got.Counts.Selected != 1 || got.Counts.AlreadyActive != 1 || got.Counts.AlreadyReceipted != 1 {
		t.Fatalf("summary=%+v", got)
	}
	if got.Results[0].Remediation == "" || !strings.Contains(got.Results[0].Remediation, "--apply") {
		t.Fatalf("candidate=%+v", got.Results[0])
	}
}
