package relay

import "testing"

// Issue #1885 done condition: rotation is refused with RELAY_NOT_EXTERNALIZED when
// transcript-only state exists, and admitted when it does not. These are that witness
// (run: `go test ./internal/relay -run Externalize`).

// TestExternalizeGateRefusesTranscriptOnly asserts the fail-closed refusal: a leg holding
// any unbacked load-bearing fact is refused the rotate, with the token and the culprits.
func TestExternalizeGateRefusesTranscriptOnly(t *testing.T) {
	facts := []LoadBearingFact{
		{Label: "the fix commit", Backing: Artifact{Kind: string(ArtifactCommit), Ref: "abc123"}},
		{Label: "a decision only in the transcript", Backing: Artifact{}}, // transcript-only
	}
	gate := CheckExternalizeGate(facts)
	if gate.Admit {
		t.Fatalf("rotate must be refused while transcript-only state exists: %+v", gate)
	}
	if gate.Reason != ReasonNotExternalized {
		t.Errorf("reason = %q, want %s", gate.Reason, ReasonNotExternalized)
	}
	if len(gate.Culprits) != 1 || gate.Culprits[0].Label != "a decision only in the transcript" {
		t.Errorf("refusal must name the transcript-only culprit, got %+v", gate.Culprits)
	}
}

// TestExternalizeGateAdmitsFullyExternalized asserts a fully externalized state (and the
// empty state) admits the rotate with no reason token.
func TestExternalizeGateAdmitsFullyExternalized(t *testing.T) {
	facts := []LoadBearingFact{
		{Label: "commit", Backing: Artifact{Kind: string(ArtifactCommit), Ref: "deadbeef"}},
		{Label: "issue", Backing: Artifact{Kind: string(ArtifactIssue), Ref: "#1885"}},
	}
	if gate := CheckExternalizeGate(facts); !gate.Admit || gate.Reason != "" {
		t.Errorf("a fully externalized state must admit the rotate cleanly: %+v", gate)
	}
	if gate := CheckExternalizeGate(nil); !gate.Admit {
		t.Errorf("an empty fact set must admit the rotate: %+v", gate)
	}
}
