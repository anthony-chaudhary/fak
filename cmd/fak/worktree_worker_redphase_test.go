package main

import (
	"testing"
)

func TestWorkerRedPhase(t *testing.T) {
	t.Run("InitBugIssue", func(t *testing.T) {
		for _, kind := range []string{"bug", "kind:bug", "BUG", " kind:bug "} {
			st := InitWorkerPhase(kind)
			if st.IssueKind != kind {
				t.Fatalf("expected IssueKind %q, got %q", kind, st.IssueKind)
			}
			if st.Phase != WorkerPhaseRedReproduction {
				t.Fatalf("expected Phase %q, got %q", WorkerPhaseRedReproduction, st.Phase)
			}
			if st.ImplementationUnlocked {
				t.Fatalf("expected ImplementationUnlocked to be false for %q", kind)
			}
			if st.HasFailingTestProof {
				t.Fatalf("expected HasFailingTestProof to be false for %q", kind)
			}
			const wantReason = "Bug fix requires reproduction test proving failure before code tree is unlocked."
			if st.Reason != wantReason {
				t.Fatalf("expected Reason %q, got %q", wantReason, st.Reason)
			}
		}
	})

	t.Run("InitNonBugIssue", func(t *testing.T) {
		for _, kind := range []string{"feat", "docs", "refactor", "chore", "enhancement", ""} {
			st := InitWorkerPhase(kind)
			if st.IssueKind != kind {
				t.Fatalf("expected IssueKind %q, got %q", kind, st.IssueKind)
			}
			if st.Phase != WorkerPhaseUnlocked {
				t.Fatalf("expected Phase %q, got %q", WorkerPhaseUnlocked, st.Phase)
			}
			if !st.ImplementationUnlocked {
				t.Fatalf("expected ImplementationUnlocked to be true for %q", kind)
			}
			if st.HasFailingTestProof {
				t.Fatalf("expected HasFailingTestProof to be false for %q", kind)
			}
			const wantReason = "Non-bug issue does not require red reproduction phase."
			if st.Reason != wantReason {
				t.Fatalf("expected Reason %q, got %q", wantReason, st.Reason)
			}
		}
	})

	t.Run("RedPhaseNoTestRan", func(t *testing.T) {
		st := InitWorkerPhase("bug")
		unlocked, reason := ValidateWorkerRedPhase(&st, 0, false)
		if unlocked {
			t.Fatal("expected unlocked to be false when test did not run")
		}
		const wantReason = "No reproduction test executed."
		if reason != wantReason {
			t.Fatalf("expected reason %q, got %q", wantReason, reason)
		}
		if st.ImplementationUnlocked {
			t.Fatal("expected state.ImplementationUnlocked to be false")
		}
		if st.Reason != wantReason {
			t.Fatalf("expected state.Reason %q, got %q", wantReason, st.Reason)
		}
		if st.Phase != WorkerPhaseRedReproduction {
			t.Fatalf("expected Phase %q, got %q", WorkerPhaseRedReproduction, st.Phase)
		}
		if st.HasFailingTestProof {
			t.Fatal("expected HasFailingTestProof to be false")
		}
	})

	t.Run("RedPhaseTautologicalPassingTest", func(t *testing.T) {
		st := InitWorkerPhase("kind:bug")
		unlocked, reason := ValidateWorkerRedPhase(&st, 0, true)
		if unlocked {
			t.Fatal("expected unlocked to be false when test passed without fix")
		}
		const wantReason = "Test passed without fix; reproduction test must fail on unfixed codebase."
		if reason != wantReason {
			t.Fatalf("expected reason %q, got %q", wantReason, reason)
		}
		if st.ImplementationUnlocked {
			t.Fatal("expected state.ImplementationUnlocked to be false")
		}
		if st.Reason != wantReason {
			t.Fatalf("expected state.Reason %q, got %q", wantReason, st.Reason)
		}
		if st.Phase != WorkerPhaseRedReproduction {
			t.Fatalf("expected Phase %q, got %q", WorkerPhaseRedReproduction, st.Phase)
		}
		if st.HasFailingTestProof {
			t.Fatal("expected HasFailingTestProof to be false")
		}
	})

	t.Run("RedPhaseFailingTestEstablishesProof", func(t *testing.T) {
		st := InitWorkerPhase("bug")
		unlocked, reason := ValidateWorkerRedPhase(&st, 1, true)
		if !unlocked {
			t.Fatalf("expected unlocked to be true on failing test, got false with reason %q", reason)
		}
		if reason != "" {
			t.Fatalf("expected empty reason on successful unlock, got %q", reason)
		}
		if !st.ImplementationUnlocked {
			t.Fatal("expected state.ImplementationUnlocked to be true")
		}
		if !st.HasFailingTestProof {
			t.Fatal("expected state.HasFailingTestProof to be true")
		}
		if st.Phase != WorkerPhaseGreenImplementation {
			t.Fatalf("expected Phase %q, got %q", WorkerPhaseGreenImplementation, st.Phase)
		}
		if st.Reason != "" {
			t.Fatalf("expected state.Reason to be empty, got %q", st.Reason)
		}

		// Subsequent validation calls when already in green phase must remain unlocked
		subsequentUnlocked, subsequentReason := ValidateWorkerRedPhase(&st, 0, false)
		if !subsequentUnlocked || subsequentReason != "" {
			t.Fatalf("expected already green state to remain unlocked, got unlocked=%v reason=%q", subsequentUnlocked, subsequentReason)
		}
	})

	t.Run("NonBugDirectlyUnlocked", func(t *testing.T) {
		st := InitWorkerPhase("feat")
		unlocked, reason := ValidateWorkerRedPhase(&st, 0, false)
		if !unlocked {
			t.Fatal("expected unlocked to be true for non-bug issue")
		}
		if reason != "" {
			t.Fatalf("expected empty reason for non-bug issue, got %q", reason)
		}
	})

	t.Run("NilStateSafe", func(t *testing.T) {
		unlocked, reason := ValidateWorkerRedPhase(nil, 0, false)
		if unlocked {
			t.Fatal("expected unlocked to be false for nil state")
		}
		if reason == "" {
			t.Fatal("expected non-empty reason for nil state")
		}
	})
}
