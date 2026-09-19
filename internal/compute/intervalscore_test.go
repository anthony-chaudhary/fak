package compute

import (
	"math"
	"testing"
)

// TestIntervalScoringPrefixEquivalence is the kernel witness: for a fixed inside
// logit sequence, every (start, end) interval scored from the single O(L) prefix
// equals the explicit per-interval summation within the fp32 reassociation bound,
// and the largest intermediate the prefix build allocates is L+1 (never L*L).
func TestIntervalScoringPrefixEquivalence(t *testing.T) {
	inside := []float32{0.5, -1.25, 2.0, 0.75, -0.5, 1.5, -2.25, 0.25, 3.0, -0.75}
	prefix := BuildIntervalPrefix(inside)
	if len(prefix.Prefix) != len(inside)+1 {
		t.Fatalf("prefix len = %d, want %d", len(prefix.Prefix), len(inside)+1)
	}
	if prefix.Prefix[0] != 0 {
		t.Fatalf("prefix[0] = %v, want 0", prefix.Prefix[0])
	}
	tol := intervalTolerance(inside)
	for start := 0; start <= len(inside); start++ {
		for end := start; end <= len(inside); end++ {
			want, ok := IntervalScoreExplicit(inside, start, end)
			if !ok {
				t.Fatalf("explicit [%d,%d) unexpectedly out of range", start, end)
			}
			got, ok := prefix.IntervalScore(inside, start, end)
			if !ok {
				t.Fatalf("prefix [%d,%d) unexpectedly out of range", start, end)
			}
			if d := float32(math.Abs(float64(got - want))); d > tol {
				t.Fatalf("interval [%d,%d): prefix = %v, explicit = %v, |delta| = %v > tol %v", start, end, got, want, d, tol)
			}
		}
	}
}

// TestIntervalScoringMemoryBound proves the kernel stays linear in memory: the
// largest intermediate for L inside logits is L+1, not L*L, so the [L, L] score
// matrix the kernel replaces is never materialized.
func TestIntervalScoringMemoryBound(t *testing.T) {
	const l = 4096
	inside := make([]float32, l)
	for i := range inside {
		inside[i] = float32((i%7)-3) * 0.25
	}
	prefix := BuildIntervalPrefix(inside)
	if got := len(prefix.Prefix); got != MaxIntervalIntermediateLen(l) {
		t.Fatalf("largest intermediate = %d, want %d (L+1, not L*L=%d)", got, MaxIntervalIntermediateLen(l), l*l)
	}
	if len(prefix.Prefix) >= l*l {
		t.Fatalf("prefix len %d is not linear in L=%d", len(prefix.Prefix), l)
	}
}

// TestIntervalScoringBatchReusesPrefix pins that many spans score from one
// prefix build, and a malformed span is dropped rather than silently clamped.
func TestIntervalScoringBatchReusesPrefix(t *testing.T) {
	inside := []float32{1, 2, 3, 4, 5}
	prefix := BuildIntervalPrefix(inside)
	spans := [][2]int{{0, 5}, {1, 3}, {2, 2}, {-1, 2}, {3, 9}}
	dst := make([]float32, len(spans))
	n := prefix.ScoreIntervals(inside, spans, dst)
	if n != 3 {
		t.Fatalf("scored %d spans, want 3 (the two out-of-range spans must be dropped)", n)
	}
	want := []float32{15, 5, 0}
	tol := intervalTolerance(inside)
	for i, w := range want {
		if d := float32(math.Abs(float64(dst[i] - w))); d > tol {
			t.Fatalf("span %v = %v, want %v (|delta| %v > tol %v)", spans[i], dst[i], w, d, tol)
		}
	}
}

// TestIntervalScoringRangeRefusals pins the fail-closed edges: an out-of-range
// interval is refused (ok=false), never a silently clamped or partial score.
func TestIntervalScoringRangeRefusals(t *testing.T) {
	inside := []float32{1, 2, 3}
	prefix := BuildIntervalPrefix(inside)
	for _, c := range []struct{ s, e int }{{-1, 2}, {0, 4}, {2, 1}, {4, 5}} {
		if got, ok := prefix.IntervalScore(inside, c.s, c.e); ok {
			t.Fatalf("IntervalScore(%d,%d) = (%v, true), want refusal", c.s, c.e, got)
		}
		if got, ok := IntervalScoreExplicit(inside, c.s, c.e); ok {
			t.Fatalf("IntervalScoreExplicit(%d,%d) = (%v, true), want refusal", c.s, c.e, got)
		}
	}
	if _, ok := prefix.IntervalScore(inside, 0, 3); !ok {
		t.Fatal("full-span interval was refused, want it scored")
	}
}

// TestIntervalScoringEmptyAndSingle pins the degenerate shapes: an empty input
// builds a length-1 zero prefix (no panic), and a single token's score is the
// token itself.
func TestIntervalScoringEmptyAndSingle(t *testing.T) {
	empty := BuildIntervalPrefix(nil)
	if len(empty.Prefix) != 1 || empty.Prefix[0] != 0 {
		t.Fatalf("empty prefix = %v, want a single 0", empty.Prefix)
	}
	single := BuildIntervalPrefix([]float32{2.5})
	got, ok := single.IntervalScore([]float32{2.5}, 0, 1)
	if !ok || got != 2.5 {
		t.Fatalf("single-token score = (%v, %v), want (2.5, true)", got, ok)
	}
}
