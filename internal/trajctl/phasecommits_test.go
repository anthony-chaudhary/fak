package trajctl

import (
	"reflect"
	"testing"
)

// TestPhaseCommitsFromTrailers_BindsNamedPhases proves the fold reads the
// Trajctl-Phase trailer out of a commit body and binds the phase to the commit's
// SHA, in commit input order.
func TestPhaseCommitsFromTrailers_BindsNamedPhases(t *testing.T) {
	commits := []TrailerCommit{
		{SHA: "aaa111", Message: "feat(trajctl): land phase one (fak trajctl)\n\nbody\n\nTrajctl-Phase: phase-1\nSigned-off-by: x <x@y>"},
		{SHA: "bbb222", Message: "feat(trajctl): land phase two (fak trajctl)\n\nTrajctl-Phase: phase-2"},
	}
	got := PhaseCommitsFromTrailers(commits)
	want := map[string][]string{
		"phase-1": {"aaa111"},
		"phase-2": {"bbb222"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("PhaseCommitsFromTrailers = %+v, want %+v", got, want)
	}
}

// TestPhaseCommitsFromTrailers_CaseInsensitiveKey proves the trailer key matches
// case-insensitively (a hand-typed `trajctl-phase:` still binds).
func TestPhaseCommitsFromTrailers_CaseInsensitiveKey(t *testing.T) {
	got := PhaseCommitsFromTrailers([]TrailerCommit{
		{SHA: "c0ffee", Message: "fix: thing\n\ntrajctl-phase:  phase-x  "},
	})
	if got["phase-x"] == nil || got["phase-x"][0] != "c0ffee" {
		t.Fatalf("case-insensitive key did not bind: %+v", got)
	}
}

// TestPhaseCommitsFromTrailers_MultipleCommitsSamePhase proves every candidate
// SHA is kept for a phase named by more than one commit (the W3 scorer credits
// the phase once ANY of them resolves), in append order.
func TestPhaseCommitsFromTrailers_MultipleCommitsSamePhase(t *testing.T) {
	got := PhaseCommitsFromTrailers([]TrailerCommit{
		{SHA: "first", Message: "Trajctl-Phase: phase-1"},
		{SHA: "second", Message: "Trajctl-Phase: phase-1"},
	})
	if want := []string{"first", "second"}; !reflect.DeepEqual(got["phase-1"], want) {
		t.Fatalf("same-phase candidates = %+v, want %+v", got["phase-1"], want)
	}
}

// TestPhaseCommitsFromTrailers_MultiPhaseCommit proves a bundled ship naming more
// than one phase binds each named phase.
func TestPhaseCommitsFromTrailers_MultiPhaseCommit(t *testing.T) {
	got := PhaseCommitsFromTrailers([]TrailerCommit{
		{SHA: "bundle", Message: "feat: bundle\n\nTrajctl-Phase: phase-1\nTrajctl-Phase: phase-2"},
	})
	if got["phase-1"][0] != "bundle" || got["phase-2"][0] != "bundle" {
		t.Fatalf("multi-phase commit did not bind both: %+v", got)
	}
}

// TestPhaseCommitsFromTrailers_DedupsWithinOneCommit proves a commit naming the
// same phase twice binds it once (no duplicate SHA in the candidate list).
func TestPhaseCommitsFromTrailers_DedupsWithinOneCommit(t *testing.T) {
	got := PhaseCommitsFromTrailers([]TrailerCommit{
		{SHA: "dup", Message: "Trajctl-Phase: phase-1\nTrajctl-Phase: phase-1"},
	})
	if want := []string{"dup"}; !reflect.DeepEqual(got["phase-1"], want) {
		t.Fatalf("intra-commit dedup failed: %+v", got["phase-1"])
	}
}

// TestPhaseCommitsFromTrailers_IgnoresNoise proves commits without the trailer,
// with an empty SHA, or with an empty phase id contribute nothing — and that a
// window with no bindings is a nil map (the fail-closed rung: no bindings scores
// 0, never credits unverified work).
func TestPhaseCommitsFromTrailers_IgnoresNoise(t *testing.T) {
	got := PhaseCommitsFromTrailers([]TrailerCommit{
		{SHA: "noref", Message: "fix: no phase trailer here\n\nSigned-off-by: x <x@y>"},
		{SHA: "", Message: "Trajctl-Phase: orphan"},           // empty SHA
		{SHA: "empty", Message: "Trajctl-Phase:   "},          // empty phase id
		{SHA: "colonless", Message: "Trajctl-Phase no colon"}, // not a trailer line
	})
	if got != nil {
		t.Fatalf("noise-only commits produced bindings, want nil: %+v", got)
	}
}

// TestPhaseCommitsFromTrailers_Empty proves an empty input is a nil map.
func TestPhaseCommitsFromTrailers_Empty(t *testing.T) {
	if got := PhaseCommitsFromTrailers(nil); got != nil {
		t.Fatalf("PhaseCommitsFromTrailers(nil) = %+v, want nil", got)
	}
}

// TestPhaseCommitsFeedCommitScorer is the load-bearing integration: the fold's
// output, wired into an EvidenceWindow with a resolver that verifies the bound
// SHA, drives the real W3 CommitProgressScorer to a witnessed progress row. This
// is the exact path the live producer takes — trailer → PhaseCommits → W3 score.
func TestPhaseCommitsFeedCommitScorer(t *testing.T) {
	obj := Objective{
		ID:        "o1",
		Statement: "ship two phases",
		Plan:      []PlanPhase{{ID: "phase-1"}, {ID: "phase-2"}},
		Status:    StatusActive,
	}
	pc := PhaseCommitsFromTrailers([]TrailerCommit{
		{SHA: "sha-1", Message: "Trajctl-Phase: phase-1"},
		// phase-2 has no commit yet — half the plan is witnessed.
	})
	win := EvidenceWindow{
		PhaseCommits: pc,
		Resolve: func(ref EvidenceRef) EvidenceStatus {
			if ref.Ref == "sha-1" {
				return EvidenceVerified
			}
			return EvidenceDangling
		},
	}
	rows := CommitProgressScorer{}.Score(obj, win)
	if len(rows) != 1 {
		t.Fatalf("scorer produced %d rows, want 1", len(rows))
	}
	if rows[0].Value != 0.5 || rows[0].Witness != W3 {
		t.Fatalf("row = value %v witness %v, want 0.5 W3 (one of two phases witnessed)", rows[0].Value, rows[0].Witness)
	}
}
