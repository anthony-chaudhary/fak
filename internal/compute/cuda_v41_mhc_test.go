package compute

import "testing"

// TestV41MHCPostOracle pins the mHC post-mix contract adapted from modular/modular
// max/kernels/src/nn/mhc.mojo @88ab8913af (Apache-2.0 WITH LLVM-exception): the products and
// sums are FP32 and the result takes exactly ONE BF16 rounding, versus an early-round control
// that rounds each product to BF16 before summing. Host-only; no hardware or speed claim, and
// cuda_v41_mhc.cu's post path (k_mhc op==3) follows this same order (FP32 fadd/fmul, one
// mhc_bf16 on the sum) by source read; its device witness stays hardware-gated.
//
// fak-test:runtime fast est=1ms lane=default
func TestV41MHCPostOracle(t *testing.T) {
	t.Parallel()
	const rows, hidden = 1, 1
	streams := []float32{1, 1, 0, 0}
	coeffs := make([]float32, 24)
	p1 := float32(1) + float32(1)/512         // 1 + 2^-9: rounds to 1.0 in BF16
	p2 := float32(1)/512 + float32(1)/(1<<20) // 2^-9 + 2^-20: rounds to 2^-9 in BF16
	coeffs[8+0*4+0], coeffs[8+1*4+0] = p1, p2 // comb[src0,dst0], comb[src1,dst0]
	got := v41MHCRefPost([]float32{0}, streams, coeffs, rows, hidden)

	// Exact FP32 sum is 1 + 2^-8 + 2^-20, just above the BF16 tie, so one rounding gives 1+2^-7.
	if want := float32(1) + float32(1)/128; got[0] != want {
		t.Fatalf("single-round post = %v, want %v", got[0], want)
	}
	early := v41MHCRefBF16(v41MHCRefBF16(p1*streams[0]) + v41MHCRefBF16(p2*streams[1]))
	if early != 1 || early == got[0] {
		t.Fatalf("early-round control = %v, want 1 and different from the single-round %v", early, got[0])
	}
	for i := 1; i < 4*hidden; i++ {
		if got[i] != 0 {
			t.Fatalf("dst %d = %v, want 0 (no coefficients set)", i, got[i])
		}
	}
	// The x term enters through BF16(x) before the post multiply, not after.
	coeffs[4+1] = 1
	x := float32(1) + float32(1)/512 // BF16(x) == 1
	if out := v41MHCRefPost([]float32{x}, streams, coeffs, rows, hidden); out[1] != 1 {
		t.Fatalf("post[dst1]*BF16(x) = %v, want 1", out[1])
	}
}
