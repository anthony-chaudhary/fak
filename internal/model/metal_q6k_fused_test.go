//go:build darwin && arm64 && cgo

package model

// metal_q6k_fused_test.go — the correctness gate for the Stage B Q6_K-down fused MLP
// (internal/metalgemm FusedMLPQ6Down / q6k_gemv in q4k.m). On a q4_k_m GGUF the expert (and
// dense) down_proj quantizes to Q6_K, which loads into kqw (not q4kw); the original q4kFusedMLP
// required all three weights Q4_K-resident, so a Q6_K down made it decline and every such MLP fell
// to the per-matmul path — the ~7% cap on the fused-expert win. This test pins the lifted path:
// the GPU runs gate/up as Q4_K and down as Q6_K in ONE command buffer, and must match the CPU
// reference (q4kMatRowsRange gate/up → silu·up → kQuantMatRows Q6_K down) up to GPU float order.

import (
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// randomQ6KTensor builds an [out,in] resident Q6_K tensor from deterministic pseudo-random
// super-block bytes (210 B/block: ql[128] + qh[64] + scales[16,int8] + d[f16@208]). Any byte
// pattern is a valid Q6_K block (the dequant is total), so the CPU reference and the GPU kernel
// interpret identical bytes — the comparison is pure kernel math, not a quantizer round-trip. Only
// the f16 super-block scale d is clamped to a small finite magnitude so the dot stays finite.
func randomQ6KTensor(out, in int, seed int64) *kQuantTensor {
	if in%qkK != 0 {
		panic("randomQ6KTensor: in not a multiple of 256")
	}
	nblk := in / qkK
	raw := make([]byte, out*nblk*q6kBlockBytes)
	rng := rand.New(rand.NewSource(seed))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	// d (f16) is the last 2 bytes of every 210-B block. Clamp its high byte to a modest exponent
	// so a uniformly random 16-bit pattern can't be a huge/Inf half (sign 0, exp ~ 2^-5..2^-2).
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := (o*nblk+b)*q6kBlockBytes + (q6kBlockBytes - 1) // high byte of d
			raw[base] = 0x2C | (raw[base] & 0x03)
		}
	}
	return &kQuantTensor{out: out, in: in, nblk: nblk, kind: kindQ6K, raw: raw}
}

// TestMetalFusedMLPQ6DownMatchesCPU is the Stage B gate: gate/up Q4_K + down Q6_K, fused on the GPU
// in one command buffer, must match the CPU reference (q4kMatRowsRange → silu·up → kQuantMatRows).
func TestMetalFusedMLPQ6DownMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	cases := []struct{ H, I int }{
		{256, 512},
		{1024, 1024},
		{5120, 17408}, // real Qwen3.6-27B MLP shape (H hidden, I intermediate)
	}
	for _, c := range cases {
		H, I := c.H, c.I
		gateQ := randomQ4KTensor(I, H, 1)
		upQ := randomQ4KTensor(I, H, 2)
		downQ := randomQ6KTensor(H, I, 3) // Q6_K down: [H, I]
		x := randomVecF(H, 4)

		// CPU reference: gate/up Q4_K f32 GEMV → silu·up → Q6_K down GEMV.
		g := make([]float32, I)
		u := make([]float32, I)
		q4kMatRowsRange(gateQ, x, g, 0, I)
		q4kMatRowsRange(upQ, x, u, 0, I)
		inter := make([]float32, I)
		for j := 0; j < I; j++ {
			inter[j] = silu(g[j]) * u[j]
		}
		ref := kQuantMatRows(downQ, inter) // Q6_K resident GEMV (byte-identical to f32, pinned elsewhere)

		// GPU fused path: gate/up Q4_K, down Q6_K, one command buffer.
		gw := metalgemm.UploadQ4K(gateQ.raw, I, H)
		uw := metalgemm.UploadQ4K(upQ.raw, I, H)
		dw := metalgemm.UploadQ6K(downQ.raw, H, I)
		if gw == nil || uw == nil || dw == nil {
			t.Fatalf("upload returned nil (H=%d I=%d): gw=%v uw=%v dw=%v", H, I, gw != nil, uw != nil, dw != nil)
		}
		got := make([]float32, H)
		if !metalgemm.FusedMLPQ6Down(gw, uw, dw, x, got) {
			t.Fatalf("FusedMLPQ6Down declined (H=%d I=%d) — shape gate(%d,%d) up(%d,%d) down(%d,%d)",
				H, I, gw.Out, gw.In, uw.Out, uw.In, dw.Out, dw.In)
		}
		cos, maxRel := cosineAndMaxRel(ref, got)
		if cos < 0.9999 || maxRel > 5e-3 {
			t.Errorf("Q6K-down fused MLP [H=%d I=%d]: cosine=%.6f maxRel=%.4g (want cos>=0.9999, maxRel<=5e-3)\n  ref[:4]=%v\n  got[:4]=%v",
				H, I, cos, maxRel, ref[:4], got[:4])
		} else {
			t.Logf("Q6K-down fused MLP [H=%d I=%d]: cosine=%.6f maxRel=%.4g OK", H, I, cos, maxRel)
		}
		metalgemm.ResetQ4K()
	}
}
