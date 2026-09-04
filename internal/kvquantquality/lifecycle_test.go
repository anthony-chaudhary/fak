package kvquantquality

import (
	"testing"
)

// Invariant: Long-context KV quantization quality evaluations must compare candidates against unquantized baselines.
// Guard: Evaluate refuses evaluations with quantized baselines or unpinned artifacts.

func TestKVQuantQualityLifecycle(t *testing.T) {
	t.Parallel()

	req := validRequest()
	report := Evaluate(req)
	if report.Outcome != OutcomeSupported {
		t.Fatalf("expected supported outcome for valid request, got %s: %s", report.Outcome, report.Reason)
	}
}
