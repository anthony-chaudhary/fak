package compute

import (
	"math"
	"math/rand"
	"testing"
)

// w8a8_expert_gemm_test.go — the laptop-composable accuracy witness for the W8A8 int8
// expert-GEMM reference (w8a8_expert_gemm.go, #3087). It builds a small deterministic
// activation×weight case, computes the plain f32 reference (the existing F32
// BatchedMatMul — an 8-accumulator fdot, a DIFFERENT reduction than the int32 int8 dot,
// so this is not a tautology) and the W8A8 int8 reference, and asserts the two agree in
// shape, per-token argmax, and cosine ≥ the recorded gate. A fail-closed test proves the
// gate has teeth: an UNRELATED weight and a degenerate zero activation both land BELOW it.
//
// This is the CPU half of the sm_80 tensor-core kernel's contract, pinned before the GPU
// witness exists — the same order in which the Q4_K CPU decode kernel landed. cosineC and
// argmaxF32 are reused from the package (quant_q4k_cpuref_test.go / cpuref.go).

// randVecW8 fills v with deterministic uniform [-1,1) values from rng.
func randVecW8(rng *rand.Rand, v []float32) {
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
}

// f32ExpertGEMM is the plain-f32 reference: Y[t,o] = Σ_i W[o,i]·X[t,i], via the shipped
// F32 BatchedMatMul (the fdot reduction), returned as a flat [P,out] host slice.
func f32ExpertGEMM(be Backend, w []float32, out, in int, x []float32, P int) []float32 {
	W := NewF32(be, []int{out, in}, w)
	X := be.Upload(NewF32(be, []int{P, in}, x), F32)
	return be.Read(be.BatchedMatMul(W, X, P))
}

func TestW8A8ExpertGEMMCosineVsF32(t *testing.T) {
	const out, in, P = 16, 512, 4
	rng := rand.New(rand.NewSource(3087))
	w := make([]float32, out*in)
	x := make([]float32, P*in)
	randVecW8(rng, w)
	randVecW8(rng, x)

	be := Default()
	yF := f32ExpertGEMM(be, w, out, in, x, P)
	yW := W8A8ExpertGEMM(be, w, out, in, x, P)

	// Shape sanity: the int8 reference produces exactly the f32 reference's [P,out] shape.
	if len(yW) != len(yF) || len(yW) != P*out {
		t.Fatalf("W8A8 output len %d, f32 len %d, want %d (=P*out)", len(yW), len(yF), P*out)
	}

	// Per-token argmax sanity: the routed expert's top output channel must survive int8
	// quantization for every token.
	for tk := 0; tk < P; tk++ {
		aW := argmaxF32(yW[tk*out : (tk+1)*out])
		aF := argmaxF32(yF[tk*out : (tk+1)*out])
		if aW != aF {
			t.Fatalf("token %d: W8A8 argmax %d != f32 argmax %d", tk, aW, aF)
		}
	}

	// Cosine gate: the int8 expert GEMM must preserve the direction of the f32 output.
	c := cosineC(yW, yF)
	if c < w8a8ExpertCosineMin {
		t.Fatalf("W8A8 vs f32 cosine %.8f < gate %.8f", c, w8a8ExpertCosineMin)
	}
	if math.IsNaN(c) || c > 1.0000001 {
		t.Fatalf("W8A8 vs f32 cosine %.8f is not a valid [-1,1] cosine", c)
	}
	t.Logf("W8A8 int8 expert-GEMM vs f32 reference: cosine %.8f >= gate %.4f (out=%d in=%d P=%d)", c, w8a8ExpertCosineMin, out, in, P)
}

// TestW8A8ExpertGEMMFailClosed proves the gate is not vacuous: an int8 output computed for
// weight W does NOT clear the cosine gate when compared against the f32 output of an
// UNRELATED weight W2, and a degenerate all-zero activation yields a finite, NaN-free zero
// output that also lands below the gate. If either passed, the gate would have no teeth.
func TestW8A8ExpertGEMMFailClosed(t *testing.T) {
	const out, in, P = 16, 512, 4
	rng := rand.New(rand.NewSource(11))
	w := make([]float32, out*in)
	w2 := make([]float32, out*in)
	x := make([]float32, P*in)
	randVecW8(rng, w)
	randVecW8(rng, w2)
	randVecW8(rng, x)

	be := Default()

	// 1) Unrelated weight: cosine of the true int8 output against the wrong f32 output must
	//    fall BELOW the gate (two independent random weights → near-orthogonal outputs).
	yW := W8A8ExpertGEMM(be, w, out, in, x, P)
	yBad := f32ExpertGEMM(be, w2, out, in, x, P)
	if c := cosineC(yW, yBad); c >= w8a8ExpertCosineMin {
		t.Fatalf("fail-closed breach: unrelated-weight cosine %.8f >= gate %.8f (gate has no teeth)", c, w8a8ExpertCosineMin)
	}

	// 2) Degenerate zero activation: dynamic amax/127 scale is 0, so the output is exactly
	//    zero — finite, no NaN, and (correctly) below the gate against the true f32 output.
	zero := make([]float32, P*in)
	yZ := W8A8ExpertGEMM(be, w, out, in, zero, P)
	for i, v := range yZ {
		if v != 0 || math.IsNaN(float64(v)) {
			t.Fatalf("zero-activation output[%d] = %v, want a clean 0", i, v)
		}
	}
	yTrue := f32ExpertGEMM(be, w, out, in, x, P)
	if c := cosineC(yZ, yTrue); c >= w8a8ExpertCosineMin {
		t.Fatalf("fail-closed breach: zero-activation cosine %.8f >= gate %.8f", c, w8a8ExpertCosineMin)
	}
}
