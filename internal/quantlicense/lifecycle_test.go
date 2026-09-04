package quantlicense

import (
	"testing"
)

// Invariant: Quantization license compatibility evaluation must distinguish artifact and recipe license chains.
// Guard: Evaluate refuses incompatible redistribution terms or unverified licenses.

func TestQuantLicenseLifecycle(t *testing.T) {
	t.Parallel()

	m := readFixture(t, "compatible.json")
	res := Evaluate(m)
	if res.Outcome != OutcomeAllow {
		t.Fatalf("expected OutcomeAllow for compatible fixture, got %s: %s", res.Outcome, res.Reason)
	}
}
