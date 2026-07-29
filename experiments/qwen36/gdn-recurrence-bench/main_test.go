package main

import (
	"math"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/experiments/qwen36/gdn"
)

// The FLOP counts ARE the published result of this command: the ~0.5%-of-prefill claim in
// the #65 decision is read straight off them, and they are exact integers, not measurements.
// Pin all three, and pin the ratio the verdict quotes, so a slipped dimension cannot rewrite
// the decision silently.
func TestFlopCounts(t *testing.T) {
	proj, conv, recur := flopCounts()

	// Five projections at the real shapes: convDim*H + valDim*H + 2*nV*H + H*valDim.
	const wantProj = int64(convDim)*H + int64(valDim)*H + 2*int64(nV)*H + int64(H)*valDim
	if wantProj != 115834880 {
		t.Fatalf("the shapes themselves drifted: projection MACs from dims = %d, want 115834880", wantProj)
	}
	if proj != wantProj {
		t.Errorf("projMACs = %d, want %d", proj, wantProj)
	}
	if want := int64(convDim) * K; conv != want || want != 40960 {
		t.Errorf("convMACs = %d, want %d (=40960)", conv, want)
	}
	// Per v-head: decay + kv_mem + delta + (state update & readout), plus the q/k L2-norms.
	if want := int64(4*kHd*vHd+vHd)*int64(nV) + int64(2*keyDim); recur != want || want != 3155968 {
		t.Errorf("recurMACs = %d, want %d (=3155968)", recur, want)
	}

	// The load-bearing ratio: the recurrence is a low-single-digit % of the projections.
	frac := float64(recur) / float64(proj)
	if frac <= 0.01 || frac >= 0.05 {
		t.Fatalf("recurrence/projection ratio = %.4f; the decision text claims a low-single-digit %%", frac)
	}
	// The conv term is asserted to be negligible in the header.
	if float64(conv)/float64(proj) > 0.001 {
		t.Errorf("conv is no longer a negligible term: %.4f of projections", float64(conv)/float64(proj))
	}
}

// l2normInto here is deliberately the MEAN-of-squares variant, not the shared
// gdn.L2NormInto (which reproduces qwen35.go's SUM form). Pin the difference so nobody
// "fixes" the duplication by folding the two together: on a kHd-wide head the two differ by
// a factor of sqrt(kHd).
func TestL2NormIntoIsTheMeanVariant(t *testing.T) {
	src := make([]float32, kHd)
	for i := range src {
		src[i] = 1
	}
	dst := make([]float32, kHd)
	l2normInto(dst, src, 1e-6)

	if math.Abs(float64(dst[0])-1) > 1e-3 {
		t.Fatalf("local l2normInto(ones) = %v, want ~1 (mean-of-squares)", dst[0])
	}
	shared := make([]float32, kHd)
	gdn.L2NormInto(shared, src, 1e-6)
	if ratio := float64(dst[0]) / float64(shared[0]); math.Abs(ratio-math.Sqrt(kHd)) > 1e-3 {
		t.Fatalf("local/shared ratio = %v, want sqrt(kHd) = %v; the two forms are not interchangeable",
			ratio, math.Sqrt(kHd))
	}
}

// medianMs returns the upper-middle sample in milliseconds and must NOT mutate
// its input (it copies before the insertion sort).
func TestMedianMs(t *testing.T) {
	ds := []time.Duration{5 * time.Millisecond, time.Millisecond, 3 * time.Millisecond, 2 * time.Millisecond, 4 * time.Millisecond}
	if got := medianMs(ds); got != 3.0 {
		t.Errorf("medianMs = %v, want 3.0", got)
	}
	if ds[0] != 5*time.Millisecond {
		t.Errorf("medianMs mutated its input: ds[0] = %v, want 5ms", ds[0])
	}
	if got := medianMs([]time.Duration{7 * time.Millisecond}); got != 7.0 {
		t.Errorf("medianMs(single) = %v, want 7.0", got)
	}
}
