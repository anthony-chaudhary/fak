package sessionrecovery

import (
	"encoding/json"
	"os"
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
		{name: "none", trees: 0, want: "productive"},
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

func TestObserveIdleShellNeverClaimsProgress(t *testing.T) {
	before := InventoryReport{Sessions: []Session{observedSession("t1", "inProgress", "2026-08-18T01:00:00Z", 0)}}
	after := InventoryReport{Sessions: []Session{observedSession("t1", "inProgress", "2026-08-18T01:00:00Z", 1)}}
	got := Observe(before, after, Result{ThreadID: "t1", Provider: ProviderCodex, Status: "launched_unproven", LaunchedAt: "2026-08-18T01:30:00Z"})
	if got.Status != "launched_unproven" || got.Advanced || got.ProgressEvidence != "" {
		t.Fatalf("idle shell falsely completed recovery: %+v", got)
	}
}

func TestObserveClaudeRequiresRealAssistantAdvancement(t *testing.T) {
	before := InventoryReport{Sessions: []Session{{Thread: &Thread{ID: "c"}, Provider: ProviderClaude, Cursor: "a1", CursorAt: "2026-08-18T01:00:00Z"}}}
	after := InventoryReport{Sessions: []Session{{Thread: &Thread{ID: "c"}, Provider: ProviderClaude, Cursor: "a2", CursorAt: "2026-08-18T02:00:00Z"}}}
	got := Observe(before, after, Result{ThreadID: "c", Provider: ProviderClaude, Status: "launched_unproven", LaunchedAt: "2026-08-18T01:30:00Z"})
	if got.Status != "productive" || !got.Advanced || got.ProgressEvidence != "claude_assistant_transcript_advanced" {
		t.Fatalf("claude progress=%+v", got)
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

func TestObserveRecognizesPostLaunchCompletionOfSameCodexTurn(t *testing.T) {
	before := observedSession("t1", "inProgress", "2026-08-18T01:00:00Z", 0)
	before.LatestTurn.ID = "turn-1"
	after := observedSession("t1", "completed", "2026-08-18T01:00:00Z", 0)
	after.LatestTurn.ID = "turn-1"
	after.LatestTurn.CompletedAt = "2026-08-18T02:00:00Z"
	got := Observe(
		InventoryReport{Sessions: []Session{before}},
		InventoryReport{Sessions: []Session{after}},
		Result{ThreadID: "t1", Provider: ProviderCodex, Status: "launched_unproven", LaunchedAt: "2026-08-18T01:30:00Z"},
	)
	if got.Status != "completed" || !got.Advanced || got.PostAt != after.LatestTurn.CompletedAt {
		t.Fatalf("same-turn completion was not witnessed: %+v", got)
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
	if got.Results[0].Remediation == "" || !strings.Contains(got.Results[0].Remediation, "--live") {
		t.Fatalf("candidate=%+v", got.Results[0])
	}
}

func TestWriteSummaryPersistsVersionedRunWitnessAtomically(t *testing.T) {
	dir := t.TempDir()
	started := time.Date(2026, 8, 24, 1, 2, 3, 4, time.UTC)
	path := SummaryPath(dir, started)
	summary := Summary{Schema: SummarySchema, Mode: "preview", StartedAt: started.Format(time.RFC3339Nano), WitnessPath: path,
		Results: []Result{{ThreadID: "t1", Provider: ProviderCodex, Category: CategorySubstantive, Status: "candidate", BaselineCursor: "turn-1"}}}
	if err := WriteSummary(path, summary); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got Summary
	if err := json.Unmarshal(raw, &got); err != nil || got.Schema != SummarySchema || got.WitnessPath != path || len(got.Results) != 1 {
		t.Fatalf("witness err=%v summary=%+v", err, got)
	}
}
