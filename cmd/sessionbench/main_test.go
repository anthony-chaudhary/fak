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

func TestDeterministicReportKeepsExactTargetCell(t *testing.T) {
	report := deterministicReport("smollm2-135m [synthetic]", false, []int{50}, []int{5}, 2048, 32, 64)
	if report["model"] != "smollm2-135m [synthetic]" {
		t.Fatalf("model = %v", report["model"])
	}
	if report["timing_mode"] != "deterministic_prefill_token_counts_only" {
		t.Fatalf("timing_mode = %v", report["timing_mode"])
	}
	cells, ok := report["cells"].([]cell)
	if !ok || len(cells) != 1 {
		t.Fatalf("cells = %#v", report["cells"])
	}
	got := cells[0]
	if got.Turns != 50 || got.Agents != 5 || got.Prefix != 2048 || got.Decode != 32 || got.Result != 64 {
		t.Fatalf("unexpected target cell: %+v", got)
	}
	a, b, c := prefillTokens(2048, 50, 5, 32, 64)
	if got.PrefillTok.A != a || got.PrefillTok.B != b || got.PrefillTok.C != c {
		t.Fatalf("prefill tokens = %+v, want (%d,%d,%d)", got.PrefillTok, a, b, c)
	}
	if !approx(got.NetVsNaive, float64(a)/float64(c), 1e-9) {
		t.Fatalf("NetVsNaive = %.6f, want %.6f", got.NetVsNaive, float64(a)/float64(c))
	}
	if got.A.Live || got.B.Live || got.C.Live {
		t.Fatalf("counts-only arms must not be marked live: %+v %+v %+v", got.A, got.B, got.C)
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

// TestSyntheticShapeTiny pins the #967 CPU-tractable wiring shape: the live B/C
// arms run end-to-end in f32, so a 135M+ shape times out unattended nightrun;
// "tiny" returns a 64h/4L/vocab-256 config the kernel runs in seconds. Pure
// arithmetic over the shape map (no NewSynthetic / kernel / GPU), so CI runs it.
func TestSyntheticShapeTiny(t *testing.T) {
	cfg, ok := syntheticShape("tiny")
	if !ok {
		t.Fatal(`syntheticShape("tiny") returned ok=false; the tiny shape must resolve`)
	}
	if cfg.HiddenSize != 64 || cfg.NumLayers != 4 || cfg.VocabSize != 256 {
		t.Fatalf("tiny shape = %dh/%dL/vocab%d, want 64h/4L/vocab256", cfg.HiddenSize, cfg.NumLayers, cfg.VocabSize)
	}
	if cfg.NumHeads != 8 || cfg.NumKVHeads != 2 || cfg.HeadDim != 8 {
		t.Fatalf("tiny attn = %dq/%dkv/headdim%d, want 8q/2kv/headdim8", cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	}

	// The pre-existing shapes must still resolve (no regression from inserting tiny first).
	for _, name := range []string{"smollm2-135m", "135m", "qwen25-1.5b", "qwen25-7b"} {
		if _, ok := syntheticShape(name); !ok {
			t.Errorf("syntheticShape(%q) regressed to ok=false", name)
		}
	}
	// An unknown shape still returns ok=false.
	if _, ok := syntheticShape("bogus"); ok {
		t.Error(`syntheticShape("bogus") returned ok=true, want false`)
	}
}
