package ultracodedogfood

import (
	"os"
	"testing"
)

// Invariant: Ultracode lifecycle dogfood sessions must evaluate cell boundaries and cache recovery without false passes.
// Guard: EvaluateLifecycleSession refuses sessions with unequal outcomes or missing provider cache reads.

func TestUltracodeDogfoodLifecycle(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("testdata/issue8678-lifecycle-session.json")
	if err != nil {
		t.Fatalf("failed reading lifecycle session: %v", err)
	}

	report, err := EvaluateLifecycleSession(data)
	if err != nil {
		t.Fatalf("EvaluateLifecycleSession failed: %v", err)
	}
	if report.Verdict != "PASS" {
		t.Fatalf("expected PASS verdict, got %s: %s", report.Verdict, report.Reason)
	}
}
