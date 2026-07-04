package relay

import "testing"

// Issue #1887 (Rung F4) done condition: an ADVERSARIAL witness — not just the #1884 detector
// and #1885 gate unit coverage — that plants a genuinely load-bearing fact ONLY in the
// transcript (the anti-pattern the externalize gate exists to kill) and proves a rotation is
// refused with RELAY_NOT_EXTERNALIZED. Per the rung's scope it drives the gate in isolation
// (no driver dependency). The gate (F2) is what makes this pass; before F2 there was nothing
// to refuse the rotate, so the assertion below would have failed.
// Run: `go test ./internal/relay -run ExternalizeGateAdversarial`.

// TestExternalizeGateAdversarialTranscriptOnlyDecision models a leg about to rotate after real
// work. It has properly externalized its code (a fix commit) and its tracking (a filed issue) —
// but the load-bearing DECISION that shaped the work ("chose the lock-free ring after the mutex
// path deadlocked under the K-arm replay") lives ONLY in the conversation transcript, backed by
// nothing durable. That is exactly the state a rotation would silently drop, leaving the
// successor to re-derive — or unknowingly contradict — the decision. The gate must refuse, and
// must name exactly the planted decision. Then, once the leg acts on the refusal and externalizes
// the decision behind a durable pointer, the SAME rotate must admit cleanly — proving the gate is
// satisfiable by the correct action, not a permanent block.
func TestExternalizeGateAdversarialTranscriptOnlyDecision(t *testing.T) {
	const decision = "chose the lock-free ring after the mutex path deadlocked under K-arm replay"
	facts := []LoadBearingFact{
		{Label: "the fix commit", Backing: Artifact{Kind: string(ArtifactCommit), Ref: "b40dd384"}},
		{Label: "the tracking issue", Backing: Artifact{Kind: string(ArtifactIssue), Ref: "#1887"}},
		// Planted: a real, load-bearing rationale that only the transcript holds.
		{Label: decision, Backing: Artifact{}},
	}

	gate := CheckExternalizeGate(facts)
	if gate.Admit {
		t.Fatalf("adversarial: a rotate that would drop a transcript-only decision must be refused, got admit: %+v", gate)
	}
	if gate.Reason != ReasonNotExternalized {
		t.Fatalf("adversarial: refusal reason = %q, want %s", gate.Reason, ReasonNotExternalized)
	}
	// Exactly the planted decision — nothing more (the commit and issue are externalized),
	// nothing less (the decision must not slip through the gate).
	if len(gate.Culprits) != 1 || gate.Culprits[0].Label != decision {
		t.Fatalf("adversarial: the gate must name exactly the transcript-only decision, got %+v", gate.Culprits)
	}

	// The remedy the gate demands: externalize the decision behind a durable pointer (here an
	// agent-memory slug). The same leg, having acted on the refusal, now rotates cleanly.
	facts[2].Backing = Artifact{Kind: string(ArtifactMemory), Ref: "relay-lockfree-ring-decision"}
	if gate := CheckExternalizeGate(facts); !gate.Admit || gate.Reason != "" || len(gate.Culprits) != 0 {
		t.Fatalf("adversarial: once the decision is externalized the rotate must admit cleanly, got %+v", gate)
	}
}

// TestExternalizeGateAdversarialShamPointerStillRefuses hardens the witness against the
// tempting evasion: dressing the transcript-only fact in a pointer that is NOT a durable,
// re-derivable artifact. A pointer with an empty ref, or a ref under a kind outside the closed
// ArtifactKind vocabulary (a chat permalink, a scratch note), does not externalize the fact —
// the state is still transcript-only and the gate must still refuse. This is the adversary that
// would otherwise route around the gate by attaching a plausible-looking but unresolvable
// backing.
func TestExternalizeGateAdversarialShamPointerStillRefuses(t *testing.T) {
	for _, sham := range []struct {
		name    string
		backing Artifact
	}{
		{"empty ref under a real kind", Artifact{Kind: string(ArtifactMemory), Ref: ""}},
		{"non-vocabulary kind", Artifact{Kind: "slack-thread", Ref: "T123/p456"}},
		{"bare label, no backing at all", Artifact{}},
	} {
		t.Run(sham.name, func(t *testing.T) {
			facts := []LoadBearingFact{
				{Label: "the committed fix", Backing: Artifact{Kind: string(ArtifactCommit), Ref: "deadbeef"}},
				{Label: "a load-bearing fact behind a sham pointer", Backing: sham.backing},
			}
			gate := CheckExternalizeGate(facts)
			if gate.Admit {
				t.Fatalf("adversarial: a sham backing must not externalize the fact — rotate must be refused, got %+v", gate)
			}
			if gate.Reason != ReasonNotExternalized {
				t.Fatalf("adversarial: refusal reason = %q, want %s", gate.Reason, ReasonNotExternalized)
			}
			if len(gate.Culprits) != 1 || gate.Culprits[0].Label != "a load-bearing fact behind a sham pointer" {
				t.Fatalf("adversarial: the gate must name the sham-backed fact as the culprit, got %+v", gate.Culprits)
			}
		})
	}
}
