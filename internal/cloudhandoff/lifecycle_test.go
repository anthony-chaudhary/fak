package cloudhandoff

import (
	"testing"
)

// Invariant: Cloud handoff must preserve operational context and respect strict policy gates.
// Guard: Handoff blocks unauthorized cloud routing when consent or triggers fail policy predicates.

func TestCloudHandoffLifecycle(t *testing.T) {
	t.Parallel()

	p := Policy{
		Eligible:        true,
		Consent:         ConsentPreapproved,
		Destinations:    []string{"vendor"},
		AllowedTriggers: []Trigger{TriggerUnsupported, TriggerFault},
	}
	r := Request{
		OperationID:      "op-lifecycle-test",
		Trigger:          TriggerFault,
		Data:             []DataClass{{Name: "prompt"}},
		DestinationClass: "vendor",
		Consequence:      "provider billing",
		Alternatives:     []string{"retry local", "cancel"},
		Payload:          []byte("payload-data"),
	}

	b := New()
	sent := false
	receipt, err := b.Handoff(p, r, nil, func(pkg Package) error {
		sent = true
		return nil
	}, Attempt{Engine: "fak-native", Location: "local", Outcome: "failed"})

	if err != nil {
		t.Fatalf("unexpected handoff error: %v", err)
	}
	if !sent || !receipt.RemoteCompleted {
		t.Fatalf("handoff failed to complete remotely: %+v", receipt)
	}
}
