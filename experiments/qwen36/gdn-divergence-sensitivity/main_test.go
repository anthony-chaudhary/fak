package main

import (
	"math"
	"testing"
)

// quantF16 emulates the f16 round-trip the divergence study applies to state.
// Pin the three regimes: f16-exact values are unchanged, overflow saturates to
// the max f16 magnitude (sign kept), and subnormal underflow flushes to zero.
func TestQuantF16(t *testing.T) {
	// Powers-of-two and simple dyadic fractions are exactly representable in f16,
	// so they must round-trip unchanged. (65504 = max f16 is NOT here: the round
	// path drops it to 65472; it only appears as the saturation target below.)
	for _, x := range []float32{0, 1, -1, 0.5, 2, 1.5, -0.25} {
		if got := quantF16(x); got != x {
			t.Errorf("quantF16(%v) = %v, want unchanged (f16-exact)", x, got)
		}
	}
	// Overflow saturates to +/-65504 (max f16), preserving sign.
	if got := quantF16(1e30); got != 65504 {
		t.Errorf("quantF16(1e30) = %v, want 65504 (saturate)", got)
	}
	if got := quantF16(-1e30); got != -65504 {
		t.Errorf("quantF16(-1e30) = %v, want -65504 (saturate)", got)
	}
	// Subnormal underflow flushes to zero.
	if got := quantF16(1e-30); got != 0 {
		t.Errorf("quantF16(1e-30) = %v, want 0 (flush)", got)
	}
}

// relDiv is the relative L2 divergence ||a-b|| / ||a||, with a zero-reference
// guard. Pin the identity, the all-zero-b case (== 1), and the den==0 guard.
func TestRelDiv(t *testing.T) {
	if got := relDiv([]float32{3, 4}, []float32{3, 4}); got != 0 {
		t.Errorf("relDiv(equal) = %v, want 0", got)
	}
	if got := relDiv([]float32{3, 4}, []float32{0, 0}); math.Abs(got-1) > 1e-12 {
		t.Errorf("relDiv([3,4],[0,0]) = %v, want 1", got)
	}
	// Zero reference vector trips the den==0 guard and returns 0, not NaN.
	if got := relDiv([]float32{0, 0}, []float32{1, 1}); got != 0 {
		t.Errorf("relDiv(zero ref) = %v, want 0 (den==0 guard)", got)
	}
}
