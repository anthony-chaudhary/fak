package compute

import (
	"math"
	"testing"
)

// prefill_test.go — the host-witnessable rungs for the issue #9 prefill scaffold. None of
// these need a CUDA device: they pin (1) the batched-GEMM skeleton's bit-identity to the
// shipped reference path, (2) the cost model's structural facts (the O(P²) attention
// crossover and the constant memory-bound intensity that justify the long-sequence and
// CUDA-graph work), and (3) the graph-capture seam's both-branches behavior on a non-CUDA
// build. The wall-clock "within 1.2× llama.cpp" acceptance is deferred to a CUDA node.

// TestPrefillGEMMBitExactToBatchedMatMul is the load-bearing rung: the tiled prefill
// kernel must produce byte-identical results to Backend.BatchedMatMul (itself per-row
// fdot) for every blocking, including tiles that do not divide out/P and P at the target
// scales. If a "fast" tiling ever reordered a reduction, the bytes would shift and this
// fires — so "bit-exact results unchanged" is witnessed, not asserted.
func TestPrefillGEMMBitExactToBatchedMatMul(t *testing.T) {
	c := cpu()
	var s lcg = 1234
	for _, dims := range []struct{ out, in, P int }{
		{5, 32, 3},    // tiny, exercises tail tiles
		{17, 48, 7},   // all three of out/in/P non-multiples of the default tile
		{64, 64, 256}, // a target prefill length
		{40, 96, 512}, // P=512 with a non-divisible out
		{8, 32, 1024}, // P=1024, the largest target
	} {
		out, in, P := dims.out, dims.in, dims.P
		w := randVec(&s, out*in)
		X := randVec(&s, P*in)
		want := c.Read(c.BatchedMatMul(NewF32(c, []int{out, in}, w), NewF32(c, []int{P, in}, X), P))
		for _, tile := range []struct{ o, t int }{
			{0, 0}, {1, 1}, {3, 5}, {64, 64}, {out, P}, {out + 9, P + 9}, {7, 256},
		} {
			got := PrefillGEMM(w, X, out, in, P, tile.o, tile.t)
			if len(got) != len(want) {
				t.Fatalf("dims %v tile %v: len %d != %d", dims, tile, len(got), len(want))
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("dims %v tile %v: cell %d drift got %v want %v (tiling must not move a byte)",
						dims, tile, i, got[i], want[i])
				}
			}
		}
	}
}

// TestPrefillGEMMEqualsFdot double-locks the skeleton against the raw reduction kernel (so
// the bit-identity does not merely chase a BatchedMatMul that itself drifted): every cell
// must equal a direct fdot of the weight row and the activation row.
func TestPrefillGEMMEqualsFdot(t *testing.T) {
	var s lcg = 99
	out, in, P := 13, 64, 20
	w := randVec(&s, out*in)
	X := randVec(&s, P*in)
	got := PrefillGEMM(w, X, out, in, P, 4, 8)
	for tk := 0; tk < P; tk++ {
		for o := 0; o < out; o++ {
			want := fdot(w[o*in:o*in+in], X[tk*in:tk*in+in])
			if math.Float32bits(got[tk*out+o]) != math.Float32bits(want) {
				t.Fatalf("PrefillGEMM[%d,%d] != fdot", tk, o)
			}
		}
	}
}

// llamaLikeGeom is a representative dense-7B-ish shape used by the cost-model rungs.
func llamaLikeGeom(P int) PrefillGeometry {
	return PrefillGeometry{
		DModel: 4096, NHeads: 32, NKVHeads: 8, HeadDim: 128,
		DFF: 11008, NLayers: 32, Vocab: 32000, P: P, WeightDtype: Q8_0,
	}
}

func stageByName(stages []StageCost, name string) (StageCost, bool) {
	for _, s := range stages {
		if s.Name == name {
			return s, true
		}
	}
	return StageCost{}, false
}

// TestPrefillCostModelStructure pins the facts that make this a real bottleneck profiler:
// the GEMM stages are exactly linear in P (double P => double FLOPs), attention is
// quadratic in P (double P => 4× FLOPs), so attention overtakes the GEMMs past a crossover
// length — the structural reason "optimize attention for long sequences" is its own scope
// bullet. All counts are exact, so these are equalities, not tolerances.
func TestPrefillCostModelStructure(t *testing.T) {
	g1 := llamaLikeGeom(512)
	g2 := llamaLikeGeom(1024) // 2× the prefill length

	s1 := PrefillCostModel(g1)
	s2 := PrefillCostModel(g2)

	// A projection GEMM is linear in P: FLOPs exactly double when P doubles.
	q1, ok1 := stageByName(s1, "q_proj")
	q2, ok2 := stageByName(s2, "q_proj")
	if !ok1 || !ok2 {
		t.Fatal("q_proj stage missing")
	}
	if q2.FLOPs != 2*q1.FLOPs {
		t.Fatalf("q_proj must be linear in P: %d != 2×%d", q2.FLOPs, q1.FLOPs)
	}

	// Attention is quadratic in P: FLOPs quadruple when P doubles.
	a1, _ := stageByName(s1, "attn")
	a2, _ := stageByName(s2, "attn")
	if a2.FLOPs != 4*a1.FLOPs {
		t.Fatalf("attn must be quadratic in P: %d != 4×%d", a2.FLOPs, a1.FLOPs)
	}

	// Naive attention is memory-bound at a P-independent ~0.5 FLOP/byte — the invariant
	// that motivates a fused/flash attention (intensity does not improve with length).
	if a1.WeightBytes != 0 {
		t.Fatalf("attention carries no weights, got WeightBytes=%d", a1.WeightBytes)
	}
	for _, a := range []StageCost{a1, a2} {
		if math.Abs(a.Intensity-0.5) > 1e-9 {
			t.Fatalf("naive attention intensity must be ~0.5 FLOP/byte, got %v", a.Intensity)
		}
	}
}

// TestPrefillAttentionCrossover witnesses that attention is NOT the bottleneck at a short
// prefill but BECOMES the dominant stage as P grows — the crossover a profiler must show
// so the optimization effort is aimed at the right stage for the right length.
func TestPrefillAttentionCrossover(t *testing.T) {
	// For this geometry the per-layer crossover is P ≈ DFF (11008): attention's
	// 2·NHeads·P²·HeadDim overtakes the FFN's 2·P·DFF·DModel once P passes
	// DFF·DModel/(NHeads·HeadDim) = DFF (since NHeads·HeadDim = DModel here). P=128 is well
	// below it (a GEMM dominates); P=32768 is well above it (attention dominates).
	short := Profile(llamaLikeGeom(128))
	long := Profile(llamaLikeGeom(32768))

	if short.Dominant.Name == "attn" {
		t.Fatalf("at P=128 a GEMM stage should dominate, not attn (got %q)", short.Dominant.Name)
	}
	if long.Dominant.Name != "attn" {
		t.Fatalf("at P=32768 attention should dominate the FLOPs, got %q", long.Dominant.Name)
	}
	// Totals must be strictly positive and the dominant stage must be a real member.
	if short.TotalFLOPs <= 0 || long.TotalFLOPs <= 0 {
		t.Fatal("total FLOPs must be positive")
	}
}

// TestStageBoundClassification checks the roofline classifier uses the caller's ridge and
// bakes in no hardware constants: the same stage flips compute/memory across the ridge.
func TestStageBoundClassification(t *testing.T) {
	s := StageCost{Intensity: 4.0}
	if s.Bound(2.0) != "compute" {
		t.Fatal("intensity 4 above ridge 2 must be compute-bound")
	}
	if s.Bound(8.0) != "memory" {
		t.Fatal("intensity 4 below ridge 8 must be memory-bound")
	}
}

// ---- CUDA-graph seam: both branches on a non-CUDA build --------------------------

// fakeGraphBE is a PrefillGraphCapturer built on the cpu-ref backend, so the captured
// branch of CapturePrefillGraph can be exercised with no CUDA device present. allow gates
// whether GraphBegin consents.
type fakeGraphBE struct {
	*cpuBackend
	allow        bool
	began, ended int
	reset        int
}

func (f *fakeGraphBE) GraphBegin() bool { f.began++; return f.allow }
func (f *fakeGraphBE) GraphEndLaunch()  { f.ended++ }
func (f *fakeGraphBE) GraphReset()      { f.reset++ }

// TestCapturePrefillGraphFallback: the CPU reference is not a capturer, so the body runs
// exactly once and the eager (false) path is reported — the non-CUDA default.
func TestCapturePrefillGraphFallback(t *testing.T) {
	ran := 0
	captured := CapturePrefillGraph(cpu(), func() { ran++ })
	if captured {
		t.Fatal("cpu-ref must not be a graph capturer (eager fallback expected)")
	}
	if ran != 1 {
		t.Fatalf("body must run exactly once on the fallback path, ran %d", ran)
	}
	ResetPrefillGraph(cpu()) // no-op on a non-capturer; must not panic
}

// TestCapturePrefillGraphCaptured: a consenting capturer takes the captured path —
// GraphBegin then body then GraphEndLaunch, reported true.
func TestCapturePrefillGraphCaptured(t *testing.T) {
	be := &fakeGraphBE{cpuBackend: cpu(), allow: true}
	ran := 0
	captured := CapturePrefillGraph(be, func() { ran++ })
	if !captured {
		t.Fatal("a consenting capturer must report captured=true")
	}
	if be.began != 1 || be.ended != 1 || ran != 1 {
		t.Fatalf("captured path: began=%d ended=%d ran=%d, want 1/1/1", be.began, be.ended, ran)
	}
	ResetPrefillGraph(be)
	if be.reset != 1 {
		t.Fatalf("ResetPrefillGraph must call GraphReset on a capturer, reset=%d", be.reset)
	}
}

// TestCapturePrefillGraphDeclined: a capturer that declines (GraphBegin=false) must fall
// back to eager execution without launching a graph.
func TestCapturePrefillGraphDeclined(t *testing.T) {
	be := &fakeGraphBE{cpuBackend: cpu(), allow: false}
	ran := 0
	captured := CapturePrefillGraph(be, func() { ran++ })
	if captured {
		t.Fatal("a declining capturer must report captured=false")
	}
	if be.began != 1 || be.ended != 0 || ran != 1 {
		t.Fatalf("declined path: began=%d ended=%d ran=%d, want 1/0/1", be.began, be.ended, ran)
	}
}
