package compute

import (
	"math"
	"testing"
)

func TestShortConvRingOracle(t *testing.T) {
	t.Parallel()
	// Zero-width no-op.
	zw, err := NewWave32ShortConvRing(1, 3, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out, err := zw.Step(0, 0, nil, nil, true); err != nil || len(out) != 0 {
		t.Fatalf("zero-width Step = %v, %v", out, err)
	}

	// K=3, maxRollback=2 -> ringLen = 4.
	r, err := NewWave32ShortConvRing(1, 3, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if r.RingLen != 4 {
		t.Fatalf("RingLen = %d, want 4", r.RingLen)
	}
	w := []float32{1, 2, 3, 0.5, 1, 1.5}
	// Commit pos 0 and pos 1.
	if _, err := r.Step(0, 0, []float32{1, 2}, w, true); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Step(0, 1, []float32{2, 1}, w, true); err != nil {
		t.Fatal(err)
	}
	// Commit=false verify at pos 2 does not mutate ring[2 % 4]; a subsequent call at pos 2
	// with the same input produces the exact same output, and rollback to pos 2 after a
	// rejected speculative commit at pos 2 still sees pos 0 and pos 1 intact.
	v1, err := r.Step(0, 2, []float32{3, 4}, w, false)
	if err != nil {
		t.Fatal(err)
	}
	v2, err := r.Step(0, 2, []float32{3, 4}, w, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := range v1 {
		if math.Abs(float64(v1[i]-v2[i])) > 1e-6 {
			t.Fatalf("commit=false vs true mismatch at %d: %v != %v", i, v1[i], v2[i])
		}
	}
	// Speculative pos 3 committed, then rejected (rolled back to pos 3 again with new token):
	// since ringLen=4, pos 1 and pos 2 are still preserved in ring[1] and ring[2].
	if _, err := r.Step(0, 3, []float32{99, 99}, w, true); err != nil {
		t.Fatal(err)
	}
	rb, err := r.Step(0, 3, []float32{1, 1}, w, true)
	if err != nil {
		t.Fatal(err)
	}
	// Expected at pos 3 for ch0: x0 + w0*ring[1] + w1*ring[2] + w2*x0 = 1 + 1*2 + 2*3 + 3*1 = 12.
	if math.Abs(float64(rb[0]-12.0)) > 1e-6 {
		t.Fatalf("rollback output ch0 = %v, want 12.0", rb[0])
	}
}
