package main

import (
	"math"
	"testing"
)

// The value-stack's arm-A projection rests on two pieces of pure arithmetic: the prefill-cost
// interpolation (prefillModel.cost) and the exact per-turn context-length sum (computeAPrefill).
// These tests pin both so a refactor can't silently corrupt the headline number. (The kernel
// timings themselves are live; this is the bookkeeping around them.)

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestPrefillCostInterpolatesAndExtrapolates(t *testing.T) {
	pm := prefillModel{Lens: []int{256, 512, 1024}, MS: []float64{600, 2000, 8000}}
	cases := []struct {
		L    int
		want float64
	}{
		{256, 600},    // exact sample
		{512, 2000},   // exact sample
		{1024, 8000},  // exact sample
		{384, 1300},   // midpoint of [256,512]: 600 + 0.5*(2000-600)
		{768, 5000},   // midpoint of [512,1024]: 2000 + 0.5*(8000-2000)
		{128, 300},    // below the first sample: linear from origin, 600*128/256
		{1536, 14000}, // extrapolate beyond: slope of top two = (8000-2000)/512, +512 past 1024
	}
	for _, c := range cases {
		got := pm.cost(c.L)
		if !approx(got, c.want, 1e-6) {
			t.Errorf("cost(%d) = %.3f, want %.3f", c.L, got, c.want)
		}
	}
}

func TestComputeAPrefillMatchesClosedForm(t *testing.T) {
	// With a purely LINEAR prefill model (cost(L) = k*L), arm A's total prefill must equal
	// k * C * Σ_{t=0..T-1}(P + t*(D+R)) = k * C * (T*P + (D+R)*T*(T-1)/2) — the closed form the
	// quadratic re-prefill is built from. This pins the count loop against algebra.
	const k = 0.5
	pm := prefillModel{Lens: []int{1, 1_000_000}, MS: []float64{k, k * 1_000_000}} // cost(L)=k*L
	P, T, C, D, R := 2048, 50, 5, 32, 64
	want := k * float64(C) * (float64(T*P) + float64(D+R)*float64(T)*float64(T-1)/2)
	got := computeAPrefill(pm, P, T, C, D, R)
	if !approx(got, want, want*1e-9+1e-6) {
		t.Fatalf("computeAPrefill = %.3f, want %.3f", got, want)
	}
}

func TestComputeAPrefillIsQuadraticInTurns(t *testing.T) {
	// Sanity: doubling the turns more-than-doubles arm A's prefill (the O(T^2) re-prefill tax).
	pm := prefillModel{Lens: []int{1, 1_000_000}, MS: []float64{1, 1_000_000}} // cost(L)=L
	P, C, D, R := 1024, 4, 24, 48
	a16 := computeAPrefill(pm, P, 16, C, D, R)
	a32 := computeAPrefill(pm, P, 32, C, D, R)
	if a32 <= 2*a16 {
		t.Fatalf("expected super-linear growth: A(32)=%.0f should exceed 2*A(16)=%.0f", a32, 2*a16)
	}
}

func TestPrefillTokensClosedForm(t *testing.T) {
	// The contention-free work-elimination floor rests on EXACT token counts. Pin all three arms
	// against their closed forms, and assert the strict ordering A>B>C the value-stack claims.
	P, T, C, D, R := 2048, 50, 5, 32, 64
	a, b, c := prefillTokens(P, T, C, D, R)
	wantA := C * (T*P + (D+R)*T*(T-1)/2) // Σ_{t=0..T-1}(P + t·(D+R)) ×C
	wantB := C * (P + (T-1)*R)
	wantC := P + C*(T-1)*R
	if a != wantA || b != wantB || c != wantC {
		t.Fatalf("prefillTokens = (%d,%d,%d), want (%d,%d,%d)", a, b, c, wantA, wantB, wantC)
	}
	if !(a > b && b > c) {
		t.Fatalf("expected strict A>B>C prefill-token ordering, got %d,%d,%d", a, b, c)
	}
	// Sanity: at this 50×5 shape the naive arm re-prefills >50× the tokens fak does.
	if float64(a)/float64(c) < 50 {
		t.Errorf("A/C prefill-token ratio %.1f unexpectedly low at 50×5", float64(a)/float64(c))
	}
}

func TestSampleLensSpansRange(t *testing.T) {
	ls := sampleLens(2048, 6848)
	if len(ls) < 4 {
		t.Fatalf("want several samples, got %v", ls)
	}
	if ls[0] > 256 {
		t.Errorf("first sample %d should be <=256 to anchor short contexts", ls[0])
	}
	if ls[len(ls)-1] < 6848 {
		t.Errorf("last sample %d should reach maxCtx 6848", ls[len(ls)-1])
	}
	// includes the prefix length
	found := false
	for _, x := range ls {
		if x == 2048 {
			found = true
		}
	}
	if !found {
		t.Errorf("sampleLens should include the prefix length 2048: %v", ls)
	}
}

func TestParseInts(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []int
	}{
		{"50", []int{50}},
		{"5,8,16", []int{5, 8, 16}},
		{"1, 2 , 3", []int{1, 2, 3}},
		{"512,6,2,24,48", []int{512, 6, 2, 24, 48}},
	} {
		got := parseInts(c.in)
		if len(got) != len(c.want) {
			t.Fatalf("parseInts(%q) = %v, want %v", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("parseInts(%q) = %v, want %v", c.in, got, c.want)
			}
		}
	}
}
