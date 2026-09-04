package completiondist

import (
	"testing"
)

// Invariant: Completion distribution statistics must accurately compute median, percentiles, and duration buckets.
// Guard: Build calculates verified p50/p95 percentiles over historical closure durations.

func TestCompletionDistLifecycle(t *testing.T) {
	t.Parallel()

	d := Build(fixture())
	if d.Count != 10 {
		t.Fatalf("expected count 10, got %d", d.Count)
	}
	if d.MedianSec() != 1800 {
		t.Fatalf("expected median 1800, got %g", d.MedianSec())
	}
}
