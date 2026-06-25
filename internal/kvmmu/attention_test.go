package kvmmu_test

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// attention_test.go — acceptance for issue #853 (span attention attribution).
//
// These tests exercise the From/Len attribution partition directly through the public
// Context API (Append lays out spans; AttributeRow routes weights; Quarantine/Compact
// evict). They use a real synthetic model.Session for the cache backend (same synthCfg
// the bridge witnesses use) so Append/evict renumbering is the real machinery, but they
// feed a KNOWN weight stream to AttributeRow rather than running a forward pass — the
// attribution math is what is under test, not the softmax numerics (those are #852).

// newAttnCtx builds a kvmmu.Context over a fresh synthetic-model session (the same
// synthCfg the other bridge tests use). The cache mechanics (Append layout, evict
// renumber) are real; the tests feed a known weight stream to AttributeRow rather than
// running a forward pass, since the attribution partition is what is under test.
func newAttnCtx(t *testing.T) *kvmmu.Context {
	t.Helper()
	m := model.NewSynthetic(synthCfg())
	return kvmmu.New(m.NewSession())
}

// attendedByID maps each segment's id to its accumulated Attended mass for assertions.
func attendedByID(c *kvmmu.Context) map[string]float64 {
	m := make(map[string]float64)
	for _, s := range c.Segments() {
		m[s.ID] = s.Attended
	}
	return m
}

// fullRow builds a contiguous attention row over [0, n) with the given per-position
// weights — the shape the rung-1 observer emits (keyPositions[i] = i).
func fullRow(weights []float32) ([]int, []float32) {
	kp := make([]int, len(weights))
	for i := range kp {
		kp[i] = i
	}
	return kp, weights
}

// TestAttributeRowConservation: a single softmax row (sums to 1.0) spread over two
// disjoint spans must distribute its full mass across the spans with zero residual —
// no weight lost, none double-counted across the From/Len partition.
func TestAttributeRowConservation(t *testing.T) {
	c := newAttnCtx(t)

	c.Append("A", "toolA", []int{1, 2, 3}) // positions 0,1,2
	c.Append("B", "toolB", []int{4, 5})    // positions 3,4

	// A normalized (sums to 1.0) row over all 5 positions: 0.6 mass on A, 0.4 on B.
	weights := []float32{0.2, 0.2, 0.2, 0.25, 0.15}
	kp, w := fullRow(weights)

	residual := c.AttributeRow(kp, w)
	if residual != 0 {
		t.Fatalf("unattributed residual = %v, want 0 (spans partition all positions)", residual)
	}

	got := attendedByID(c)
	if d := math.Abs(got["A"] - 0.6); d > 1e-6 {
		t.Errorf("A attended = %v, want 0.6 (Δ=%v)", got["A"], d)
	}
	if d := math.Abs(got["B"] - 0.4); d > 1e-6 {
		t.Errorf("B attended = %v, want 0.4 (Δ=%v)", got["B"], d)
	}

	// Conservation: total attributed mass == total emitted mass (1.0).
	var emitted float64
	for _, x := range weights {
		emitted += float64(x)
	}
	if d := math.Abs(c.AttendedMass() - emitted); d > 1e-6 {
		t.Errorf("AttendedMass = %v, want emitted %v (Δ=%v) — conservation violated", c.AttendedMass(), emitted, d)
	}
}

// TestAttributeRowCorrectSpan: a weight in span A's range lands in A, not B, on a
// multi-span sequence — attribution respects the From/Len boundary exactly.
func TestAttributeRowCorrectSpan(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2, 3}) // 0,1,2
	c.Append("B", "toolB", []int{4, 5, 6}) // 3,4,5

	// All mass on position 1 (inside A) and position 4 (inside B).
	kp := []int{1, 4}
	w := []float32{0.7, 0.3}
	if r := c.AttributeRow(kp, w); r != 0 {
		t.Fatalf("residual = %v, want 0", r)
	}
	got := attendedByID(c)
	if math.Abs(got["A"]-0.7) > 1e-6 {
		t.Errorf("A = %v, want 0.7", got["A"])
	}
	if math.Abs(got["B"]-0.3) > 1e-6 {
		t.Errorf("B = %v, want 0.3", got["B"])
	}
}

// TestAttributeRowUnattendedSpanIsZero: a span no row ever names accumulates 0.
func TestAttributeRowUnattendedSpanIsZero(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // 0,1
	c.Append("B", "toolB", []int{3, 4}) // 2,3  (never attended)

	// All mass inside A.
	c.AttributeRow([]int{0, 1}, []float32{0.5, 0.5})

	got := attendedByID(c)
	if got["B"] != 0 {
		t.Errorf("B attended = %v, want 0 (never attended)", got["B"])
	}
	if math.Abs(got["A"]-1.0) > 1e-6 {
		t.Errorf("A attended = %v, want 1.0", got["A"])
	}
}

// TestAttributeRowResidualOnHeldSpan: a weight on a position owned by no live segment
// (here, after a span is evicted) is returned as the unattributed residual, not
// silently dropped and not misattributed to a neighbor.
func TestAttributeRowResidualOnEvicted(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2, 3}) // 0,1,2
	c.Append("B", "toolB", []int{4, 5})    // 3,4

	// Evict A; B renumbers to From=0,Len=2 → live positions are now [0,2). A position
	// at index 3 (beyond the compacted cache) is owned by no live segment.
	if ev, ok := c.Quarantine("A"); !ok || ev == 0 {
		t.Fatalf("Quarantine(A) = (%d,%v), want evicted>0,true", ev, ok)
	}
	residual := c.AttributeRow([]int{0, 1, 3}, []float32{0.4, 0.4, 0.2})
	if math.Abs(residual-0.2) > 1e-6 {
		t.Errorf("residual = %v, want 0.2 (the weight on the out-of-range position)", residual)
	}
	got := attendedByID(c)
	if math.Abs(got["B"]-0.8) > 1e-6 {
		t.Errorf("B = %v, want 0.8 (the two in-range weights)", got["B"])
	}
}

// TestEvictDropsMassSurvivorsKeep: eviction zeroes the evicted span's mass while every
// survivor keeps exactly the mass it accumulated (mass is not renumbered with From).
func TestEvictDropsMassSurvivorsKeep(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // 0,1
	c.Append("B", "toolB", []int{3, 4}) // 2,3
	c.Append("C", "toolC", []int{5, 6}) // 4,5

	// Attend each span: A=0.2, B=0.5, C=0.3.
	c.AttributeRow([]int{0, 1, 2, 3, 4, 5}, []float32{0.1, 0.1, 0.25, 0.25, 0.15, 0.15})
	before := attendedByID(c)
	if math.Abs(before["A"]-0.2) > 1e-6 || math.Abs(before["B"]-0.5) > 1e-6 || math.Abs(before["C"]-0.3) > 1e-6 {
		t.Fatalf("setup masses wrong: A=%v B=%v C=%v", before["A"], before["B"], before["C"])
	}

	// Evict the middle span B. A and C must keep their mass; B drops to 0.
	if ev, ok := c.Quarantine("B"); !ok || ev == 0 {
		t.Fatalf("Quarantine(B) = (%d,%v)", ev, ok)
	}
	after := attendedByID(c)
	if after["B"] != 0 {
		t.Errorf("B attended after evict = %v, want 0", after["B"])
	}
	if math.Abs(after["A"]-0.2) > 1e-6 {
		t.Errorf("A attended after evict = %v, want 0.2 (survivor mass preserved)", after["A"])
	}
	if math.Abs(after["C"]-0.3) > 1e-6 {
		t.Errorf("C attended after evict = %v, want 0.3 (survivor mass preserved)", after["C"])
	}

	// AttendedMass now excludes the evicted span: 0.2 + 0.3 = 0.5.
	if d := math.Abs(c.AttendedMass() - 0.5); d > 1e-6 {
		t.Errorf("AttendedMass after evict = %v, want 0.5 (Δ=%v)", c.AttendedMass(), d)
	}
}

// TestAttentionObserverAttributes: the model.AttnObserver adapter routes emitted rows
// onto the ledger exactly as AttributeRow does (the wiring the live forward pass uses).
func TestAttentionObserverAttributes(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // 0,1
	c.Append("B", "toolB", []int{3, 4}) // 2,3

	obs := c.AttentionObserver()
	// Emit two rows (e.g. two query positions / heads); masses accumulate.
	obs(0, 0, 0, []int{0, 1, 2, 3}, []float32{0.25, 0.25, 0.25, 0.25})
	obs(0, 1, 0, []int{0, 1, 2, 3}, []float32{0.1, 0.1, 0.4, 0.4})

	got := attendedByID(c)
	// A got 0.5 + 0.2 = 0.7; B got 0.5 + 0.8 = 1.3.
	if math.Abs(got["A"]-0.7) > 1e-6 {
		t.Errorf("A = %v, want 0.7", got["A"])
	}
	if math.Abs(got["B"]-1.3) > 1e-6 {
		t.Errorf("B = %v, want 1.3", got["B"])
	}
}
