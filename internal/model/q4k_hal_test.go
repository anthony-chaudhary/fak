package model

import (
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// q4k_hal_test.go — witnesses the Q4_K device-GEMM path for the GLM-DSA forward:
//  1. the cpu-ref backend's Q4_K MatMul reproduces the model's f32 Q4_K GEMV (q4kMatRowsRange)
//     BIT-FOR-BIT — so cpu-ref is the Reference the cuda k_q4k_gemm lane is the Approx peer of;
//  2. Session.glmDsaWeightHAL (the seam that previously PANICKED on a resident Q4_K weight —
//     "q4_k device GLM-DSA upload is a follow-up") now stages it onto the backend and its MatMul
//     equals the host q4kMatRows. That closes the residual: a memory-lean Q4_K GLM-5.2 can run its
//     dense projections on the device (k_q4k_gemm on the GPU server, cpu-ref here), not just q8/f32.

// buildRawQ4K returns a valid raw Q4_K weight [out,in] (in a multiple of 256): out*(in/256)
// random-but-finite super-blocks, using the shared randQ4KBlock generator (constrains the f16
// d/min so no NaN/Inf scale poisons the dot). Deterministic for a given seed.
func buildRawQ4K(t *testing.T, out, in, seed int) []byte {
	t.Helper()
	if in%qkK != 0 {
		t.Fatalf("in=%d not a multiple of %d", in, qkK)
	}
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	rng := rand.New(rand.NewSource(int64(seed)))
	for b := 0; b < out*nblk; b++ {
		randQ4KBlock(rng, raw[b*q4kBlockBytes:(b+1)*q4kBlockBytes])
	}
	return raw
}

// TestCPURefQ4KMatMulIsReference pins compute.cpuBackend.MatMul(Q4_K) == the model's f32 Q4_K GEMV
// (q4kMatRowsRange) with max|Δ|=0. The cpu-ref backend is the Reference, so this is the floor the
// cuda Q4_K kernel's recorded cosine gate (cudaQ4KCosineMin) sits above.
func TestCPURefQ4KMatMulIsReference(t *testing.T) {
	const out, in = 6, 512 // in = 2 super-blocks
	raw := buildRawQ4K(t, out, in, 1)
	qt := quantizeQ4KFromRaw(raw, out, in)

	x := make([]float32, in)
	rng := rand.New(rand.NewSource(99))
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	// Reference: the model's f32 Q4_K row dot.
	yRef := make([]float32, out)
	q4kMatRowsRange(qt, x, yRef, 0, out)

	// cpu-ref backend Q4_K MatMul over the SAME raw bytes.
	be := compute.Default()
	wt := compute.NewQ4K(be, []int{out, in}, raw)
	xt := be.Upload(compute.NewF32(be, []int{in}, x), compute.F32)
	yBE := be.Read(be.MatMul(wt, xt))

	if len(yBE) != out {
		t.Fatalf("cpu-ref Q4_K MatMul len %d, want %d", len(yBE), out)
	}
	for o := 0; o < out; o++ {
		if yBE[o] != yRef[o] {
			t.Fatalf("cpu-ref Q4_K MatMul not Reference at row %d: %v != model f32 %v", o, yBE[o], yRef[o])
		}
	}

	// BatchedMatMul Q4_K row t must equal the single-row MatMul (the q4kGemm contract).
	const P = 3
	X := make([]float32, P*in)
	for t2 := 0; t2 < P; t2++ {
		copy(X[t2*in:], x) // same row thrice — every batched row must reproduce yRef
	}
	Xt := be.Upload(compute.NewF32(be, []int{P, in}, X), compute.F32)
	YB := be.Read(be.BatchedMatMul(wt, Xt, P))
	for t2 := 0; t2 < P; t2++ {
		for o := 0; o < out; o++ {
			if YB[t2*out+o] != yRef[o] {
				t.Fatalf("cpu-ref Q4_K BatchedMatMul[%d,%d]=%v != MatMul %v", t2, o, YB[t2*out+o], yRef[o])
			}
		}
	}
}

// TestGLMDsaWeightHALServesQ4K is the residual-closure witness: glmDsaWeightHAL, given a resident
// Q4_K weight (m.q4kw), stages it onto the backend and the backend MatMul equals the host
// q4kMatRows — so the GLM-DSA forward's backendKernel can drive a Q4_K-resident dense projection
// instead of panicking. (Before this change glmDsaWeightHAL had no q4k branch and panicked.)
func TestGLMDsaWeightHALServesQ4K(t *testing.T) {
	const out, in = 8, 256
	raw := buildRawQ4K(t, out, in, 7)
	qt := quantizeQ4KFromRaw(raw, out, in)

	be := compute.Default()
	m := &Model{q4kw: map[string]*q4kTensor{"w": qt}}
	s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}

	x := make([]float32, in)
	rng := rand.New(rand.NewSource(123))
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	// The model's f32 Q4_K GEMV reference (q4kMatRowsRange — arch-independent, the path cpu-ref's
	// Q4_K MatMul reproduces bit-for-bit; q4kMatRows may take the arm64 int8-SDOT path).
	yHost := make([]float32, out)
	q4kMatRowsRange(qt, x, yHost, 0, out)

	// Route through the HAL seam: glmDsaWeightHAL must pick the q4k branch (no panic) and the
	// backend MatMul of the staged weight must match the host Q4_K GEMV.
	wt := s.glmDsaWeightHAL("w", out, in)
	if wt.Dtype != compute.Q4_K {
		t.Fatalf("glmDsaWeightHAL staged dtype %v, want Q4_K", wt.Dtype)
	}
	xt := be.Upload(compute.NewF32(be, []int{in}, x), compute.F32)
	yBE := be.Read(be.MatMul(wt, xt))
	if len(yBE) != out {
		t.Fatalf("staged Q4_K MatMul len %d, want %d", len(yBE), out)
	}
	for o := 0; o < out; o++ {
		if yBE[o] != yHost[o] {
			t.Fatalf("glmDsaWeightHAL Q4_K MatMul[%d]=%v != host q4kMatRows %v", o, yBE[o], yHost[o])
		}
	}

	// The weight is cached on the q4k key so a device session uploads it once, not per token.
	if _, ok := s.halW["q4k:w"]; !ok {
		t.Errorf("glmDsaWeightHAL did not cache the staged Q4_K weight under q4k:w")
	}
}
