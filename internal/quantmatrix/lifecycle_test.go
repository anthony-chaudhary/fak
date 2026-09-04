package quantmatrix

import (
	"testing"
)

// Invariant: Quantization capability matrix must correctly adjudicate support across registered execution envelopes.
// Guard: Adjudicate refuses unknown versions or unsupported runtime combinations.

func TestQuantMatrixLifecycle(t *testing.T) {
	t.Parallel()

	req := Request{ID: EntryGGUFQ4KCPU, ArtifactVersion: "gguf-v3", Runtime: "fak-native-cpu"}
	dec := Adjudicate(req)
	if dec.Outcome != OutcomeAllow {
		t.Fatalf("expected OutcomeAllow, got %s: %s", dec.Outcome, dec.Reason)
	}
}
