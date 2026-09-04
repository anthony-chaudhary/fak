package kvvectoreval

import (
	"testing"
)

// Invariant: KV vector evaluation must preserve exact paper artifact verification and delegate runtime checks.
// Guard: Evaluate rejects requests with mismatched artifact digests or unregistered recipe IDs.

func TestKVVectorEvalLifecycle(t *testing.T) {
	t.Parallel()

	req := validRequest()
	res := Evaluate(req)
	if res.Outcome != OutcomeSupported {
		t.Fatalf("expected OutcomeSupported, got %s: %s", res.Outcome, res.Reason)
	}
}
