package residualquant

import (
	"testing"
)

// Invariant: Residual quantization adjudication must verify tier bits and research paper descriptors.
// Guard: Adjudicate refuses invalid contracts, unsupported operations, or unpinned weights.

func TestResidualQuantLifecycle(t *testing.T) {
	t.Parallel()

	d := PinnedPaperDescriptor()
	req := Request{Descriptor: d, Operation: "inspect", TierBits: 6}
	dec := Adjudicate(req)
	if dec.Verdict != CaseSupported {
		t.Fatalf("expected CaseSupported, got %s: %s", dec.Verdict, dec.Reason)
	}
}
