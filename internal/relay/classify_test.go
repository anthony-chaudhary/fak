package relay

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// Issue #1886 done condition: a test asserts EPHEMERAL state does not trip the gate and
// DURABLE-worthy state does (witness: `go test ./internal/relay -run Classify`).

// TestClassifyEphemeralDoesNotTripGate plants a turn-scoped fact that lives ONLY in the
// transcript (no durable backing). The raw F2 gate would refuse the rotate on it; the F3
// classifier recognizes it as ephemeral, so the bounded gate ADMITS — an ephemeral culprit is
// exactly the false positive the classifier exists to suppress.
func TestClassifyEphemeralDoesNotTripGate(t *testing.T) {
	c := Candidate{
		LoadBearingFact: LoadBearingFact{Label: "the wall clock reads 3pm", Backing: Artifact{}},
		Durability:      ctxplan.DurabilityTurn,
	}
	if class, lb := Classify(c); lb {
		t.Fatalf("a turn-scoped candidate must classify ephemeral, got class=%q loadBearing=%v", class, lb)
	}
	// Precondition: unbacked, so the RAW F2 gate would refuse it...
	if raw := CheckExternalizeGate([]LoadBearingFact{c.LoadBearingFact}); raw.Admit {
		t.Fatalf("precondition: the raw F2 gate should refuse an unbacked fact, got %+v", raw)
	}
	// ...but the F3-bounded gate must ADMIT, because the fact is ephemeral.
	if gate := CheckExternalizeGateClassified([]Candidate{c}); !gate.Admit {
		t.Errorf("ephemeral transcript-only state must NOT trip the bounded gate: %+v", gate)
	}
}

// TestClassifyDurableTripsGate plants a durable-worthy fact that lives ONLY in the transcript.
// It IS load-bearing — a rotation would silently drop a decision the successor needs — so the
// bounded gate must still refuse with RELAY_NOT_EXTERNALIZED and name the culprit. Once the
// same fact is backed by a durable pointer, though still load-bearing, the gate admits.
func TestClassifyDurableTripsGate(t *testing.T) {
	c := Candidate{
		LoadBearingFact: LoadBearingFact{Label: "chose the lock-free ring over a mutex", Backing: Artifact{}},
		Durability:      ctxplan.DurabilityDurable,
	}
	class, lb := Classify(c)
	if !lb {
		t.Fatalf("a durable candidate must classify load-bearing, got class=%q", class)
	}
	gate := CheckExternalizeGateClassified([]Candidate{c})
	if gate.Admit {
		t.Fatalf("durable-worthy transcript-only state must trip the bounded gate: %+v", gate)
	}
	if gate.Reason != ReasonNotExternalized {
		t.Errorf("reason = %q, want %s", gate.Reason, ReasonNotExternalized)
	}
	if len(gate.Culprits) != 1 || gate.Culprits[0].Label != c.Label {
		t.Errorf("refusal must name the load-bearing culprit, got %+v", gate.Culprits)
	}
	// Externalize the durable fact: the bounded gate now admits the rotate.
	c.Backing = Artifact{Kind: string(ArtifactMemory), Ref: "relay-notes"}
	if gate := CheckExternalizeGateClassified([]Candidate{c}); !gate.Admit {
		t.Errorf("an externalized load-bearing fact must admit: %+v", gate)
	}
}

// TestClassifyUnknownClassIsLoadBearing pins the fail-closed direction: an unclassified or
// unknown durability class is load-bearing (NOT ephemeral), so unknown state can never route
// around the gate. Only an explicit turn class is ephemeral.
func TestClassifyUnknownClassIsLoadBearing(t *testing.T) {
	for _, class := range []string{"", "bogus", ctxplan.DurabilitySession, ctxplan.DurabilityBounded, ctxplan.DurabilityDurable} {
		c := Candidate{
			LoadBearingFact: LoadBearingFact{Label: "unbacked " + class, Backing: Artifact{}},
			Durability:      class,
		}
		if _, lb := Classify(c); !lb {
			t.Errorf("class %q must be treated as load-bearing (fail closed), got ephemeral", class)
		}
		if gate := CheckExternalizeGateClassified([]Candidate{c}); gate.Admit {
			t.Errorf("class %q: unbacked load-bearing state must trip the gate, got admit", class)
		}
	}
}

// TestClassifyMixedKeepsOnlyLoadBearingCulprits mixes ephemeral and durable unbacked state and
// asserts the bounded gate refuses naming ONLY the load-bearing culprit — the ephemeral one is
// filtered out before the gate sees it.
func TestClassifyMixedKeepsOnlyLoadBearingCulprits(t *testing.T) {
	candidates := []Candidate{
		{LoadBearingFact: LoadBearingFact{Label: "the clock reads 3pm", Backing: Artifact{}}, Durability: ctxplan.DurabilityTurn},
		{LoadBearingFact: LoadBearingFact{Label: "picked the lock-free ring", Backing: Artifact{}}, Durability: ctxplan.DurabilityDurable},
	}
	gate := CheckExternalizeGateClassified(candidates)
	if gate.Admit {
		t.Fatalf("a mix with a load-bearing transcript-only fact must refuse: %+v", gate)
	}
	if len(gate.Culprits) != 1 || gate.Culprits[0].Label != "picked the lock-free ring" {
		t.Errorf("only the load-bearing culprit may be named, got %+v", gate.Culprits)
	}
}
