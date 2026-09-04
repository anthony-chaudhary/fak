package fastintent

import (
	"testing"
)

// Invariant: Fast intent joining must preserve model provider receipts and downgrade transparency.
// Guard: Join refuses plans with missing quality floors or unreplayable evaluation bundles.

func TestFastIntentLifecycle(t *testing.T) {
	t.Parallel()

	plan, providers, evaluation := fixture(t)
	receipt, err := Join(plan, providers, evaluation)
	if err != nil {
		t.Fatalf("Join failed: %v", err)
	}
	if receipt.EvidenceDigest == "" {
		t.Fatal("expected non-empty EvidenceDigest")
	}
	if receipt.Verdict != evaluation.Verdict {
		t.Fatalf("expected verdict %s, got %s", evaluation.Verdict, receipt.Verdict)
	}
}
