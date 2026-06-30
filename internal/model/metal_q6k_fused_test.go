//go:build darwin && arm64 && cgo

package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// randomQ6KTensor builds an [out,in] resident Q6_K kQuantTensor from deterministic pseudo-random
// super-block bytes. Any byte pattern is a valid Q6_K block (the dequant is total), so the CPU
// reference (kQuantMatRows / q6kDequantSuperBlock) and the GPU q6k_gemv interpret identical bytes
// — the comparison is pure kernel math, not a quantizer round-trip. The only constraint is that
// the per-super-block f16 scale d (last 2 bytes) stays finite, so the dot doesn't blow to Inf/NaN.
func randomQ6KTensor(out, in int, seed int64) *kQuantTensor {
	if in%qkK != 0 {
		panic("randomQ6KTensor: in not a multiple of 256")
	}
	nblk := in / qkK
	bb := q6kBlockBytes
	raw := make([]byte, out*nblk*bb)
	rng := rand.New(rand.NewSource(seed))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	// d (f16 at block end): a small positive half so the accumulation stays finite.
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := (o*nblk+b)*bb + bb - 2
			binary.LittleEndian.PutUint16(raw[base:], 0x2C00) // half ~0.0625
		}
	}
	return &kQuantTensor{out: out, in: in, nblk: nblk, kind: kindQ6K, raw: raw}
}

// TestMetalFusedMLPQ6DownMatchesCPU is the runtime correctness gate for the mixed-quant fused MLP:
// gate/up are Q4_K, down is Q6_K (the q4_k_m expert shape). It pins metalgemm.FusedMLPQ6Down — the
// one-command-buffer GPU path that runs the Q6_K down GEMV via q6k_gemv — against the exact CPU
// reference q4kFusedMLP falls back to: gate/up via q4kMatRows, silu(gate)*up, down via kQuantMatRows.
// The gate is cosine >= 0.9999 (the GPU's simd_sum reduction reorders the f32 accumulation vs the
// CPU's serial sum — the same Approx tolerance the Q4_K GEMV/MLP tests use), not bit-equality.
func TestMetalFusedMLPQ6DownMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("metalgemm backend not available (needs Apple Silicon + cgo + -tags fakmetal)")
	}
	t.Cleanup(func() { metalgemm.ResetQ4K() })

	const (
		H = 512  // hidden size = gate/up.In = down.Out
		I = 1536 // intermediate = gate/up.Out = down.In
	)
	gateT := randomQ4KTensor(I, H, 1) // [I,H]
	upT := randomQ4KTensor(I, H, 2)   // [I,H]
	downT := randomQ6KTensor(H, I, 3) // [H,I] Q6_K
	x := randomVecF(H, 4)

	// CPU reference: exactly the per-matmul path q4kFusedMLP returns to when the fused path declines.
	g := q4kMatRows(gateT, x) // [I]
	u := q4kMatRows(upT, x)   // [I]
	inter := make([]float32, I)
	for i := 0; i < I; i++ {
		inter[i] = silu(g[i]) * u[i]
	}
	cpu := kQuantMatRows(downT, inter) // [H], the resident Q6_K down GEMV

	// GPU fused path: upload gate/up (Q4_K) + down (Q6_K), run FusedMLPQ6Down.
	gw := metalgemm.UploadQ4K(gateT.raw, gateT.out, gateT.in)
	uw := metalgemm.UploadQ4K(upT.raw, upT.out, upT.in)
	dw := metalgemm.UploadQ6K(downT.raw, downT.out, downT.in)
	if gw == nil || uw == nil || dw == nil {
		t.Fatalf("upload failed: gate=%v up=%v down=%v", gw != nil, uw != nil, dw != nil)
	}
	gpu := make([]float32, H)
	if !metalgemm.FusedMLPQ6Down(gw, uw, dw, x, gpu) {
		t.Fatal("FusedMLPQ6Down returned false (shape mismatch) — gate/up In==down.Out==H, gate/up Out==down.In==I expected")
	}

	cos, maxRel := cosineAndMaxRel(cpu, gpu)
	if cos < 0.9999 || maxRel > 5e-3 || math.IsNaN(cos) {
		t.Errorf("FusedMLPQ6Down vs CPU: cosine=%.6f maxRel=%.4g (want cos>=0.9999, maxRel<=5e-3)\n  cpu[:4]=%v\n  gpu[:4]=%v",
			cos, maxRel, cpu[:4], gpu[:4])
	} else {
		t.Logf("FusedMLPQ6Down [H=%d,I=%d] vs CPU: cosine=%.6f maxRel=%.4g OK", H, I, cos, maxRel)
	}
}
