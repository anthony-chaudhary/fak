//go:build amd64

package model

import (
	"math"
	"math/rand"
	"testing"
)

// TestQ4KInt8DecodeFaithfulAMD64 witnesses that the amd64 AVX2 int8 Q4_K decode GEMV
// (q4kMatRowsInto's SDOT path: q4kReduceRowAsmAVX2 + the shared float combine q4kCombineRow)
// stays within activation-quantization tolerance of the f32 dequant reference (q4kMatRowsRange)
// at a Qwen3.6-27B-shaped reduction dim. It is the amd64 counterpart to TestQ4KInt8DotMatchesF32,
// which SKIPS on amd64 because the int8 SDOT decode gate (FAK_KQ_INT8 / q4kSDOTEnabled) is off by
// default there. As a result the AVX2 reducer's INTEGER output is pinned bit-identical to scalar
// by TestQ4KReduceAsmMatchesScalar, but the END-TO-END int8 dot (integer reduction + float
// combine, i.e. what a real decode emits) had no numeric-faithfulness-vs-f32 witness on this arch.
// This closes that gap — the missing rung for trusting the ~8x AVX2 int8 decode path
// (BenchmarkQ4KInt8GEMV ~1.7ms vs BenchmarkQ4KF32GEMV ~14ms on an AVX2 box) as the resident-Q4_K
// decode for Qwen3.6-27B q4_k_m on a CPU server.
//
// The approximation under test is EXACTLY the activation Q8_0 quantization the shipping Q8 decode
// path already applies to every matmul; the weights stay exact Q4_K nibbles (no Q8 re-quant), so
// this path is strictly more faithful to the q4_k_m artifact than the current lean-Q8 default. A
// gross mismatch here (bad affine-min handling, sub-block/scale indexing, or activation-block
// alignment) diverges by orders of magnitude, not percent — which is what makes 5% a real gate.
func TestQ4KInt8DecodeFaithfulAMD64(t *testing.T) {
	if !detectAVX2() {
		t.Skip("AVX2 not available — amd64 int8 q4k reducer inactive")
	}
	if qtier < tierAVX2 {
		t.Skip("resolved kernel tier below AVX2 — no asm reducer to witness")
	}
	// Force the int8 SDOT decode gate ON so q4kMatRowsInto dispatches to the AVX2 reducer
	// (production-gated behind FAK_KQ_INT8 on amd64). Save/restore the exact prior value rather
	// than assuming a default, so the witness composes with whatever ambient force the suite runs.
	prev := q4kSDOTForce
	q4kSDOTForce = 1
	t.Cleanup(func() { q4kSDOTForce = prev })
	if !q4kSDOTEnabled() {
		t.Fatalf("forced int8 gate did not take (q4kSDOTForce=%d)", q4kSDOTForce)
	}

	const out, in = 512, 5120 // Qwen3.6-27B hidden reduction dim (5120 = 20 Q4_K super-blocks).
	rng := rand.New(rand.NewSource(20260709))
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			// Bounded d/min exponents keep the dequanted weights in the ~0.1 band a real
			// q4_k_m quantizer produces (same generator the native-parity gate uses), so the
			// measured error is the genuine activation-Q8 error, not catastrophic cancellation
			// from synthetic high-dynamic-range weights.
			randQ4KBlockBounded(rng, blk, 2, 6)
			off := (o*nblk + b) * q4kBlockBytes
			copy(raw[off:off+q4kBlockBytes], blk)
		}
	}
	qt := quantizeQ4KFromRaw(raw, out, in)

	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	yInt8 := make([]float32, out)
	q4kMatRowsInto(qt, x, yInt8) // AVX2 int8 SDOT path (gate forced on)
	yF32 := make([]float32, out)
	q4kMatRowsRange(qt, x, yF32, 0, out) // f32 scalar dequant reference

	var sumSq float64
	for o := 0; o < out; o++ {
		sumSq += float64(yF32[o]) * float64(yF32[o])
	}
	rms := math.Sqrt(sumSq / float64(out))
	if rms < 1e-9 {
		t.Fatalf("f32 reference RMS ~0; bad test data")
	}
	var maxRel float64
	for o := 0; o < out; o++ {
		if rel := math.Abs(float64(yInt8[o]-yF32[o])) / rms; rel > maxRel {
			maxRel = rel
		}
	}
	// Same activation-Q8 tolerance the arch-neutral sibling (TestQ4KInt8DotMatchesF32) uses: 0.05
	// ceiling, gaussian activations land well under 1% per dot. A real indexing/affine bug blows
	// this up by orders of magnitude.
	if maxRel > 0.05 {
		t.Fatalf("amd64 int8 q4k decode vs f32 max-abs/RMS %.4f exceeds 0.05 (tier=%d)", maxRel, qtier)
	}
	t.Logf("amd64 AVX2 int8 q4k decode vs f32: max-abs/RMS = %.4e (out=%d in=%d rms=%.4g tier=%d)", maxRel, out, in, rms, qtier)
}
