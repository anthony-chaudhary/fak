package main

import (
	"math"
	"testing"
	"time"
)

// The activation helpers are the pure numeric core of the bench. Pin their
// defining values and saturation limits so a typo (e.g. a flipped sign in the
// exponent) cannot pass silently.
func TestActivations(t *testing.T) {
	if got := silu(0); got != 0 {
		t.Errorf("silu(0) = %v, want 0", got)
	}
	if got := sigmoidf(0); got != 0.5 {
		t.Errorf("sigmoidf(0) = %v, want 0.5", got)
	}
	if got := softplus(0); math.Abs(float64(got)-math.Ln2) > 1e-6 {
		t.Errorf("softplus(0) = %v, want ln2 (%v)", got, math.Ln2)
	}
	// silu(x) -> x as x -> +inf (the gate saturates to 1); -> 0 as x -> -inf.
	if got := silu(20); math.Abs(float64(got)-20) > 1e-3 {
		t.Errorf("silu(20) = %v, want ~20", got)
	}
	if got := silu(-20); math.Abs(float64(got)) > 1e-6 {
		t.Errorf("silu(-20) = %v, want ~0", got)
	}
	// sigmoidf saturates monotonically toward 1 and 0.
	if got := sigmoidf(20); got < 0.999 {
		t.Errorf("sigmoidf(20) = %v, want ~1", got)
	}
	if got := sigmoidf(-20); got > 0.001 {
		t.Errorf("sigmoidf(-20) = %v, want ~0", got)
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
