package trajctlhook

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// BenchmarkHookDispatch measures the core turn-end hook dispatch and evaluation
// pipeline over active objectives using the cheap scorer set (W3 commit progress
// and W2 stall divergence) and evidence window assembly.
func BenchmarkHookDispatch(b *testing.B) {
	state := trajctl.State{
		Objectives: map[string]trajctl.Objective{
			"obj-active": {
				ID:        "obj-active",
				Statement: "ship trajectory control hook",
				Plan: []trajctl.PlanPhase{
					{ID: "phase-1", Title: "hook assembly"},
					{ID: "phase-2", Title: "benchmark witness"},
				},
				Status: trajctl.StatusActive,
			},
			"obj-paused": {
				ID:        "obj-paused",
				Statement: "secondary objective",
				Plan:      []trajctl.PlanPhase{{ID: "phase-3"}},
				Status:    trajctl.StatusPaused,
			},
			"obj-done": {
				ID:        "obj-done",
				Statement: "completed task",
				Plan:      []trajctl.PlanPhase{{ID: "phase-0"}},
				Status:    trajctl.StatusMet,
			},
		},
		Scores: []trajctl.ScoreRow{
			{
				ObjectiveID: "obj-active",
				Method:      trajctl.CommitScorerMethod,
				Value:       0.5,
				Witness:     trajctl.W3,
				UnixMillis:  500,
			},
		},
	}
	in := WindowInput{
		PhaseCommits: map[string][]string{
			"phase-1": {"a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"},
			"phase-2": {"b2c3d4e5f60718293a4b5c6d7e8f90123456789a"},
		},
		Resolve: func(ref trajctl.EvidenceRef) trajctl.EvidenceStatus {
			return trajctl.EvidenceVerified
		},
		UnixMillis: 1000,
	}
	stamp := trajctl.Stamp{SessionID: "sess-bench", RunID: "run-bench"}
	scorers := CheapScorers()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		win := BuildWindow(state, in)
		sample := trajctl.Sample(state.Objectives, scorers, win, stamp)
		if len(sample.Rows) == 0 {
			b.Fatal("sample produced zero rows")
		}
	}
}

// BenchmarkBuildWindow measures pure evidence window construction from folded
// ledger state, prior score history, and session inputs.
func BenchmarkBuildWindow(b *testing.B) {
	state := trajctl.State{
		Scores: []trajctl.ScoreRow{
			{ObjectiveID: "o1", Value: 0.25, Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3},
			{ObjectiveID: "o1", Value: 0.50, Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3},
			{ObjectiveID: "o2", Value: 1.00, Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3},
		},
	}
	in := WindowInput{
		PhaseCommits: map[string][]string{
			"p1": {"deadbeef1"},
			"p2": {"deadbeef2"},
		},
		Resolve: func(ref trajctl.EvidenceRef) trajctl.EvidenceStatus {
			return trajctl.EvidenceVerified
		},
		UnixMillis: 9999,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		win := BuildWindow(state, in)
		if len(win.PriorScores) != 3 {
			b.Fatal("unexpected prior scores count")
		}
	}
}

// BenchmarkRunTurnEnd measures end-to-end Stop-hook execution including ledger
// state folding, evidence window assembly, cheap scorer evaluation, and sample
// ledger append.
func BenchmarkRunTurnEnd(b *testing.B) {
	dir := b.TempDir()
	ledger := filepath.Join(dir, "trajctl.jsonl")
	obj := trajctl.Objective{
		ID:        "obj-bench",
		Statement: "benchmark turn end",
		Plan:      []trajctl.PlanPhase{{ID: "p1"}},
		Status:    trajctl.StatusActive,
	}
	seed, err := json.Marshal(trajctl.ObjectiveRecord(obj))
	if err != nil {
		b.Fatalf("marshal objective: %v", err)
	}
	seed = append(seed, '\n')
	if err := os.WriteFile(ledger, seed, 0o644); err != nil {
		b.Fatalf("seed ledger: %v", err)
	}
	in := WindowInput{
		PhaseCommits: map[string][]string{"p1": {"a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"}},
		Resolve: func(ref trajctl.EvidenceRef) trajctl.EvidenceStatus {
			return trajctl.EvidenceVerified
		},
		UnixMillis: 1000,
	}
	stamp := trajctl.Stamp{SessionID: "sess-bench", RunID: "run-bench"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := os.WriteFile(ledger, seed, 0o644); err != nil {
			b.Fatalf("reset ledger: %v", err)
		}
		res := RunTurnEnd(ledger, in, stamp)
		if res.Err != nil {
			b.Fatalf("RunTurnEnd: %v", res.Err)
		}
		if res.Appended != 1 {
			b.Fatalf("appended = %d, want 1", res.Appended)
		}
	}
}

// BenchmarkRunCompaction measures PreCompact hook dispatch creating W0 context
// boundary markers across open objectives and appending them to the ledger.
func BenchmarkRunCompaction(b *testing.B) {
	dir := b.TempDir()
	ledger := filepath.Join(dir, "trajctl.jsonl")
	obj := trajctl.Objective{
		ID:        "obj-compact",
		Statement: "benchmark compaction",
		Status:    trajctl.StatusActive,
	}
	seed, err := json.Marshal(trajctl.ObjectiveRecord(obj))
	if err != nil {
		b.Fatalf("marshal objective: %v", err)
	}
	seed = append(seed, '\n')
	if err := os.WriteFile(ledger, seed, 0o644); err != nil {
		b.Fatalf("seed ledger: %v", err)
	}
	stamp := trajctl.Stamp{SessionID: "sess-compact", RunID: "run-compact"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := os.WriteFile(ledger, seed, 0o644); err != nil {
			b.Fatalf("reset ledger: %v", err)
		}
		res := RunCompaction(ledger, 2000, stamp)
		if res.Err != nil {
			b.Fatalf("RunCompaction: %v", res.Err)
		}
		if res.Appended != 1 {
			b.Fatalf("appended = %d, want 1", res.Appended)
		}
	}
}
