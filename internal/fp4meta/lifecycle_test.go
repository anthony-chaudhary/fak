package fp4meta

import (
	"testing"
)

// Invariant: FP4 metadata parsing must strictly validate block sizes, scale encodings, and accumulator types.
// Guard: Parse rejects unsupported variant definitions and unverified hardware targets.

func TestFP4MetaLifecycle(t *testing.T) {
	t.Parallel()

	caps := allCapabilities()
	d := goldenDescriptor(t, "nvfp4")
	res := Adjudicate(d, caps)
	if res.Outcome != OutcomeAccept {
		t.Fatalf("expected accept outcome for golden nvfp4 descriptor, got %s: %s", res.Outcome, res.Reason)
	}
}
