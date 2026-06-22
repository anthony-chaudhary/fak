//go:build arm64

package model

import (
	"fmt"
	"math"
	"testing"
)

// quant_arm64_amort_test.go — the DEDICATED correctness gate for the issue #477
// amortized-FP-reduction decode kernel (qdot8amortNEON, quant_arm64_amort.s).
//
// This kernel deliberately CHANGES the float reduction order (per-block int32 dot kept as a
// 4-lane vector, converted with one SCVTF.4S, FMLA'd into four lane-parallel float
// accumulators, reduced once at the end — the llama.cpp/ggml deferred reduction), so it is
// NOT bit-identical to qdot8scalar and MUST NOT be pinned by TestQdot8NEONMatchesScalar (which
// stays the anchor for the bit-identical qdot8asm path and is untouched). Its correctness
// contract is the one the prefill GEMM and the decode unroll4 paths already use, and the one
// the M3 acceptance run applies against the HF oracle: the produced logit vector's ARGMAX is
// unchanged and its COSINE similarity to the scalar reference is ~1.
//
// The integer dot is provably identical (each 32-wide block sum ≤ 32·127·127 ≈ 5.2e5, exact in
// both int32 and float32), so the only divergence is float-combine rounding order — which is
// why argmax holds exactly and cosine sits at 1−ε across realistic decode shapes here.

// amortArgmaxCosineShapes covers the Qwen2.5-1.5B decode GEMV shapes (hidden 1536, intermediate
// 8960) plus a vocab-sized argmax target — the case where argmax-exactness matters most, since
// the LM head's argmax IS the greedy next token.
var amortArgmaxCosineShapes = []struct{ out, in int }{
	{1536, 1536},  // q/k/v/o projection
	{8960, 1536},  // gate/up projection
	{1536, 8960},  // down projection
	{32768, 1536}, // vocab-like LM-head argmax target
}

// gemvScalarRef computes the full decode GEMV y[o] = qdot8scalar(weight row o, qv) — the
// portable reference order qdot8asm is bit-identical to. The amortized kernel is compared
// against THIS (not qdot8asm), so the gate is independent of which arch ran the reference.
func gemvScalarRef(qt *q8Tensor, qv q8Vec) []float32 {
	y := make([]float32, qt.out)
	for o := 0; o < qt.out; o++ {
		y[o] = qdot8scalar(qt.q[o*qt.in:o*qt.in+qt.in], qt.d[o*qt.nblk:o*qt.nblk+qt.nblk], qv, qt.nblk)
	}
	return y
}

// gemvAmort computes the same GEMV via the issue #477 amortized kernel, one row at a time
// exactly as qMatRowsRange would under FAK_QKERNEL=amort.
func gemvAmort(qt *q8Tensor, qv q8Vec) []float32 {
	y := make([]float32, qt.out)
	for o := 0; o < qt.out; o++ {
		y[o] = qdot8amortNEON(&qt.q[o*qt.in], &qv.q[0], &qt.d[o*qt.nblk], &qv.d[0], qt.nblk)
	}
	return y
}

// TestQdot8AmortArgmaxAndCosine is the new gate (NOT a relaxation of TestQdot8NEONMatchesScalar,
// which it leaves alone). It runs the decode GEMV both ways over realistic shapes and several
// seeds and requires: (1) argmax of the amortized logit vector EQUALS the scalar reference's
// argmax (the greedy-decode-preserving property), and (2) cosine similarity ≥ 0.9999 (it is
// ~1−1e-6 in practice; the bound is the slack a deferred float reduction can introduce). Skips
// on an arm64 part without FEAT_DotProd, where the kernel is never dispatched.
func TestQdot8AmortArgmaxAndCosine(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — amortized NEON kernel inactive, scalar path only")
	}
	const minCosine = 0.9999
	worstCos := 1.0
	for _, s := range amortArgmaxCosineShapes {
		for trial := 0; trial < 4; trial++ {
			w := mkVec(s.out*s.in, uint64(s.out)*15485863+uint64(s.in)*2654435761+uint64(trial)*40503+1)
			x := mkVec(s.in, uint64(s.in)*2246822519+uint64(trial)*6364136223846793005+7)
			qt := quantizeQ8(w, s.out, s.in)
			qv := quantizeVecQ8(x)

			want := gemvScalarRef(qt, qv)
			got := gemvAmort(qt, qv)

			if a, b := argmax(got), argmax(want); a != b {
				t.Fatalf("out=%d in=%d trial=%d: argmax amort=%d (%v) != scalar=%d (%v) — greedy token would differ",
					s.out, s.in, trial, a, got[a], b, want[b])
			}
			c := cosine(got, want)
			if c < worstCos {
				worstCos = c
			}
			if c < minCosine {
				mx, at := amortMaxAbsDiff(got, want)
				t.Fatalf("out=%d in=%d trial=%d: cosine %.8f < %.4f (max|Δ|=%g at %d)",
					s.out, s.in, trial, c, minCosine, mx, at)
			}
		}
	}
	t.Logf("amortized kernel: argmax-exact + cosine ≥ %.4f across %d shapes (worst cosine %.8f); FEAT_DotProd=%v FEAT_I8MM=%v tier=%d",
		minCosine, len(amortArgmaxCosineShapes), worstCos, detectDotProd(), i8mmAvailable, qkernelTier)
}

func amortMaxAbsDiff(a, b []float32) (float64, int) {
	mx, at := 0.0, -1
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > mx {
			mx, at = d, i
		}
	}
	return mx, at
}

// BenchmarkDotKernelAmort A/Bs the SINGLE-CORE Q8 decode dot throughput of the three arm64
// kernels at decode GEMV shapes: the bit-identical per-block qdot8asm, the default unroll4, and
// the issue #477 amortized kernel. The MAC/ns metric is what the acceptance script reads to
// report the amort-vs-default decode delta on the M3 (the tok/s headline is the full-model
// modelbench run; this isolates the kernel).
func BenchmarkDotKernelAmort(b *testing.B) {
	if !detectDotProd() {
		b.Skip("FEAT_DotProd (asimddp) not available")
	}
	shapes := []struct{ in, out int }{{1536, 1536}, {8960, 1536}, {1536, 8960}}
	for _, s := range shapes {
		w := mkVec(s.out*s.in, uint64(s.out*s.in*131+7))
		qt := quantizeQ8(w, s.out, s.in)
		x := mkVec(s.in, uint64(s.in*977+3))
		qv := quantizeVecQ8(x)
		nblk := qt.nblk
		run := func(b *testing.B, kernel string) {
			var sink float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for o := 0; o < s.out; o++ {
					switch kernel {
					case "asm":
						sink += qdot8asm(&qt.q[o*s.in], &qv.q[0], &qt.d[o*nblk], &qv.d[0], nblk)
					case "unroll4":
						sink += qdot8unroll4NEON(&qt.q[o*s.in], &qv.q[0], &qt.d[o*nblk], &qv.d[0], nblk)
					case "amort":
						sink += qdot8amortNEON(&qt.q[o*s.in], &qv.q[0], &qt.d[o*nblk], &qv.d[0], nblk)
					}
				}
			}
			b.StopTimer()
			if sink == 0 {
				b.Fatal("zero")
			}
			b.ReportMetric(float64(s.out)*float64(s.in)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "MAC/ns")
		}
		for _, k := range []string{"asm", "unroll4", "amort"} {
			b.Run(fmt.Sprintf("%dx%d/%s", s.out, s.in, k), func(b *testing.B) { run(b, k) })
		}
	}
}
