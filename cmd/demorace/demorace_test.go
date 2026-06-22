// Tests for the pure, deterministic helpers in the demorace command: the
// exact (timing-free) prefill-token accounting, the prefill-cost
// interpolation/extrapolation model, and the tokens/sec formatter. None of
// these touch a model file, the network, or any goroutine — they are plain
// arithmetic and string formatting, so the expected values below are computed
// by hand from the session structure documented in main.go.
package main

import "testing"

func TestPrefillTokens(t *testing.T) {
	tests := []struct {
		name                string
		P, T, C, D, R       int
		wantA, wantB, wantC int
	}{
		{
			// Default workload (handleLadder): a = C·Σ_{t<T}(P + t·(D+R)),
			// b = C·(P + (T-1)·R), c = P + C·(T-1)·R.
			// D+R = 48; Σ_{t=0..4}(512 + 48t) = 2560 + 48·10 = 3040; a = 5·3040 = 15200.
			// b = 5·(512 + 4·32) = 5·640 = 3200. c = 512 + 5·4·32 = 1152.
			name: "default", P: 512, T: 5, C: 5, D: 16, R: 32,
			wantA: 15200, wantB: 3200, wantC: 1152,
		},
		{
			// Single turn (T=1): the t-loop runs once with t=0, and the (T-1)
			// terms vanish. a = C·P, b = C·P, c = P.
			name: "single_turn", P: 100, T: 1, C: 4, D: 8, R: 16,
			wantA: 400, wantB: 400, wantC: 100,
		},
		{
			// Single agent (C=1): a and b lose their ×C factor.
			// D+R = 9; Σ_{t=0..2}(10 + 9t) = (10)+(19)+(28) = 57; a = 1·57 = 57.
			// b = 1·(10 + 2·5) = 20. c = 10 + 1·2·5 = 20.
			name: "single_agent", P: 10, T: 3, C: 1, D: 4, R: 5,
			wantA: 57, wantB: 20, wantC: 20,
		},
		{
			// curve default workload (handleCurve): P=128 T=5 C=3 D=16 R=16.
			// D+R = 32; Σ_{t=0..4}(128 + 32t) = 640 + 32·10 = 960; a = 3·960 = 2880.
			// b = 3·(128 + 4·16) = 3·192 = 576. c = 128 + 3·4·16 = 320.
			name: "curve_default", P: 128, T: 5, C: 3, D: 16, R: 16,
			wantA: 2880, wantB: 576, wantC: 320,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, b, c := prefillTokens(tc.P, tc.T, tc.C, tc.D, tc.R)
			if a != tc.wantA || b != tc.wantB || c != tc.wantC {
				t.Fatalf("prefillTokens(%d,%d,%d,%d,%d) = (a=%d, b=%d, c=%d); want (a=%d, b=%d, c=%d)",
					tc.P, tc.T, tc.C, tc.D, tc.R, a, b, c, tc.wantA, tc.wantB, tc.wantC)
			}
			// Invariant the demo relies on: the cold loop never does less prefill
			// work than the fak arm, and the warm arm sits between them.
			if !(a >= b && b >= c) {
				t.Fatalf("expected a >= b >= c, got a=%d b=%d c=%d", a, b, c)
			}
		})
	}
}

func TestPrefillModelCost(t *testing.T) {
	pm := prefillModel{
		Lens: []int{100, 200},
		MS:   []float64{10, 20},
	}
	tests := []struct {
		name string
		L    int
		want float64
	}{
		// L below the first sample: linear from the origin through (Lens[0], MS[0]).
		// 10 · 50/100 = 5.
		{"below_first", 50, 5},
		// L exactly at the first sample.
		{"at_first", 100, 10},
		// L between samples: 10 + 0.5·(20-10) = 15.
		{"interpolate", 150, 15},
		// L exactly at the last sample.
		{"at_last", 200, 20},
		// L above the last sample: extrapolate on the last segment's slope
		// (0.1 ms/tok): 20 + 0.1·(300-200) = 30.
		{"above_last", 300, 30},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := pm.cost(tc.L)
			if got != tc.want {
				t.Fatalf("pm.cost(%d) = %v; want %v", tc.L, got, tc.want)
			}
		})
	}

	// An empty model has no samples and must return zero cost for any length.
	var empty prefillModel
	if got := empty.cost(123); got != 0 {
		t.Fatalf("empty prefillModel cost = %v; want 0", got)
	}
}

func TestTokPerSec(t *testing.T) {
	tests := []struct {
		tps  float64
		want string
	}{
		{0, "0 tok/s"},
		{840, "840 tok/s"},
		{840.4, "840 tok/s"}, // rounds down to nearest whole tok/s
		{999, "999 tok/s"},   // just under the 1000 threshold stays in tok/s
		{1000, "1.0k tok/s"}, // threshold is inclusive (>= 1000)
		{1500, "1.5k tok/s"},
		{12340, "12.3k tok/s"},
	}
	for _, tc := range tests {
		got := tokPerSec(tc.tps)
		if got != tc.want {
			t.Errorf("tokPerSec(%v) = %q; want %q", tc.tps, got, tc.want)
		}
	}
}
