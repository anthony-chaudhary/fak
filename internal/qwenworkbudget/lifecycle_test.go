package qwenworkbudget

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// Invariant: Qwen work budget evaluation must correctly attribute token usage and identify amplification breaches.
// Guard: Evaluate holds or alerts on amplification breaches and refuses non-native engines.

func TestQwenWorkBudgetLifecycle(t *testing.T) {
	t.Parallel()

	audit := auditWith(transcript("claude", "q1", "Qwen3", 40, 10, 50, 2))
	policy := Policy{MaxInputPerOutput: 2}
	packet := Packet{Boundary: BoundaryLaunch, Engine: "fak-native/qwen3", Audit: &audit}

	receipt := policy.Evaluate(packet)
	if !receipt.Eligible || receipt.Action != trajectory.QwenAmplificationObserve {
		t.Fatalf("expected eligible observe action, got eligible=%v, action=%s", receipt.Eligible, receipt.Action)
	}
}
