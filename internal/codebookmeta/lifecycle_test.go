package codebookmeta

import (
	"testing"
)

// Invariant: Codebook metadata parsing must strictly validate packing and runtime capability requirements.
// Guard: Adjudicate rejects descriptors whose packing ID or runtime is missing from capabilities.

func TestCodebookMetaLifecycle(t *testing.T) {
	t.Parallel()

	caps := Capability{
		PackingIDs:     []string{"nibble-lsb@1"},
		DecodeFeatures: []string{"per-block-scale"},
		RoutedRuntimes: []string{"lab-runtime@2.4.1"},
	}

	d := readFixture(t, "nf4.json")
	res := Adjudicate(d, caps)
	if res.Outcome != OutcomeSupported {
		t.Fatalf("expected supported outcome, got %s: %s", res.Outcome, res.Reason)
	}
}
