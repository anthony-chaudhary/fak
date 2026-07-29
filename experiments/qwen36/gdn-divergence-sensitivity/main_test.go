package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/experiments/qwen36/gdn"
)

// The kernel primitives this command used to own now live in the shared gdn package and are
// tested there (gdn_test.go). What is left here is the one thing this command adds: the
// lockstep two-run residual stack whose divergence curve IS the published result. These tests
// exercise it at a small hidden size — the head/state dims (nV, kHd, vHd, K) and the layer
// count stay real, only the projection fan-in shrinks.
const (
	testHidden = 64
	testTokens = 4
	testLayers = 3
)

// TestForwardVsForwardIsExactlyZero is the load-bearing control. The whole experiment claims
// that the ONLY difference between its reference and test runs is the scan's numerics. If the
// test run is given the same mode as the reference, every reported rho must therefore be
// EXACTLY zero — not merely small. A nonzero value here would mean the harness leaks some
// other difference (weights, input, scheduling) into the measurement and that every published
// rho is partly noise of unknown size.
func TestForwardVsForwardIsExactlyZero(t *testing.T) {
	perLayer, final := runStackTraced(testHidden, testTokens, testLayers, gdn.ModeForward)
	if len(perLayer) != testLayers {
		t.Fatalf("got %d per-layer samples, want one per layer (%d)", len(perLayer), testLayers)
	}
	for i, rho := range perLayer {
		if rho != 0 {
			t.Fatalf("layer %d: rho = %v, want exactly 0 — reference and test runs differ by something "+
				"other than the scan mode", i+1, rho)
		}
	}
	if final != 0 {
		t.Fatalf("final rho = %v, want exactly 0", final)
	}
}

// TestReorderPerturbsAndCompounds pins the shape of the published claim: reversing the scan's
// reduction order is pure f32 rounding, so it must move the hidden state (strictly > 0) while
// staying orders of magnitude below the rho* ~ 0.0875 needed to flip the observed near-tie —
// and, because the scan is the only stateful op, the divergence must GROW with depth rather
// than stay put.
func TestReorderPerturbsAndCompounds(t *testing.T) {
	perLayer, final := runStackTraced(testHidden, testTokens, testLayers, gdn.ModeReverse)
	t.Logf("reorder per-layer rho: %v (final %v)", perLayer, final)

	if perLayer[0] <= 0 {
		t.Fatalf("layer 1 rho = %v, want > 0: a reversed f32 reduction must round differently", perLayer[0])
	}
	if final <= perLayer[0] {
		t.Fatalf("final rho %v did not grow past layer 1's %v; the residual stack must compound the "+
			"per-layer difference", final, perLayer[0])
	}
	const rhoStar = 1.75 / 20.0 // the near-tie flip threshold the command reports against
	if final >= rhoStar {
		t.Fatalf("final rho = %v reached the near-tie threshold %v at a 3-layer smoke scale; pure "+
			"reduction-order rounding cannot plausibly be that large", final, rhoStar)
	}
}

// TestF16StateDivergesMoreThanReorder pins the bracket ordering the command's verdict reads
// off: the f16-state mode drops 13 mantissa bits from the recurrent accumulator every step, so
// it must diverge strictly more than a reduction-order-only change. If this inverted, the
// "strongest bracket" the verdict is taken from would be the weakest one.
func TestF16StateDivergesMoreThanReorder(t *testing.T) {
	_, reorder := runStackTraced(testHidden, testTokens, testLayers, gdn.ModeReverse)
	_, f16 := runStackTraced(testHidden, testTokens, testLayers, gdn.ModeF16State)
	t.Logf("reorder final rho %v vs f16state final rho %v", reorder, f16)
	if f16 <= reorder {
		t.Fatalf("f16-state rho %v must exceed reduction-order rho %v", f16, reorder)
	}
}

// TestRunStackTracedHonoursHidden guards the parameter that replaced the baked-in H: a
// different hidden size must actually change the measurement rather than being ignored.
func TestRunStackTracedHonoursHidden(t *testing.T) {
	_, small := runStackTraced(64, testTokens, 1, gdn.ModeF16State)
	_, large := runStackTraced(128, testTokens, 1, gdn.ModeF16State)
	if small == large {
		t.Fatalf("hidden=64 and hidden=128 produced the identical rho %v; the parameter is not "+
			"reaching the stack", small)
	}
	if small <= 0 || large <= 0 {
		t.Fatalf("both runs must report a positive f16-state divergence; got %v and %v", small, large)
	}
}
