package microscaleeval

import (
	"testing"
)

// Invariant: Microscale evaluation must correctly identify native vs delegated execution profiles.
// Guard: Evaluate refuses requests with mismatched runtime architectures or unverified formats.

func TestMicroscaleEvalLifecycle(t *testing.T) {
	t.Parallel()

	op := Operand{"e2m1", "none"}
	req := Request{
		Descriptor:   Descriptor{Schema: SchemaV1, Family: "ocp-mx-v1", BlockSize: 32, ScaleFormat: "e8m0", Weights: op, Activations: op},
		Capabilities: RuntimeCapabilities{Profiles: []NativeProfile{{Family: "ocp-mx-v1", BlockSize: 32, ScaleFormat: "e8m0", Weights: op, Activations: op}}},
		Provenance:   pinned(),
		Evidence:     modeled(),
	}

	result := Evaluate(req)
	if result.Outcome != Native {
		t.Fatalf("expected Native outcome, got %s: %s", result.Outcome, result.Reason)
	}
}
