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

// buildFusedExpertBatch uploads n experts' resident Q4_K gate/up + Q6_K down at shape (H,I) and
// returns the metalgemm handles + the shared token activation x[H]. Shared by the batch correctness
// gate and the batch benchmark so both exercise the exact same residency the live decode uses.
func buildFusedExpertBatch(tb testing.TB, H, I, n int) ([]*metalgemm.Q4KWeight, []*metalgemm.Q4KWeight, []*metalgemm.Q6KWeight, []float32) {
	gws := make([]*metalgemm.Q4KWeight, n)
	uws := make([]*metalgemm.Q4KWeight, n)
	dws := make([]*metalgemm.Q6KWeight, n)
	for e := 0; e < n; e++ {
		gq := randomQ4KTensor(I, H, int64(10*e+1))
		uq := randomQ4KTensor(I, H, int64(10*e+2))
		dq := randomQ6KTensor(H, I, int64(10*e+3))
		gws[e] = metalgemm.UploadQ4K(gq.raw, I, H)
		uws[e] = metalgemm.UploadQ4K(uq.raw, I, H)
		dws[e] = metalgemm.UploadQ6K(dq.raw, H, I)
		if gws[e] == nil || uws[e] == nil || dws[e] == nil {
			tb.Fatalf("expert %d upload returned nil (H=%d I=%d)", e, H, I)
		}
	}
	return gws, uws, dws, randomVecF(H, 99)
}

// TestMetalFusedMLPQ6DownBatchMatchesSingle is the batch correctness gate (#1382): each expert row
// of FusedMLPQ6DownBatch (all n experts in ONE command buffer, per-expert scratch offsets) must
// equal the single FusedMLPQ6Down of that same expert. A wrong per-expert offset binding in the
// batched kernel would corrupt exactly one row, which this catches at max|Δ|~0 (same kernel, same
// bytes, only the command-buffer packing differs).
func TestMetalFusedMLPQ6DownBatchMatchesSingle(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	const H, I, n = 1024, 1024, 5
	gws, uws, dws, x := buildFusedExpertBatch(t, H, I, n)

	// Single-expert reference: one FusedMLPQ6Down per expert.
	refs := make([][]float32, n)
	for e := 0; e < n; e++ {
		refs[e] = make([]float32, H)
		if !metalgemm.FusedMLPQ6Down(gws[e], uws[e], dws[e], x, refs[e]) {
			t.Fatalf("single FusedMLPQ6Down declined for expert %d", e)
		}
	}

	// Batched path: all n experts in one command buffer.
	ycat := make([]float32, n*H)
	if !metalgemm.FusedMLPQ6DownBatch(gws, uws, dws, x, ycat) {
		t.Fatalf("FusedMLPQ6DownBatch declined (H=%d I=%d n=%d)", H, I, n)
	}
	for e := 0; e < n; e++ {
		row := ycat[e*H : (e+1)*H]
		var maxAbs float64
		for j := 0; j < H; j++ {
			if d := float64(row[j] - refs[e][j]); d > maxAbs || -d > maxAbs {
				if d < 0 {
					d = -d
				}
				maxAbs = d
			}
		}
		if maxAbs != 0 {
			t.Fatalf("expert %d: batched row != single FusedMLPQ6Down, max|Δ|=%g (want 0 — same kernel, offset binding)", e, maxAbs)
		}
	}
	t.Logf("FusedMLPQ6DownBatch == n×FusedMLPQ6Down, max|Δ|=0 over %d experts [H=%d I=%d]", n, H, I)
}

// BenchmarkMetalFusedMLPQ6DownSingle times n separate FusedMLPQ6Down calls (n command buffers) at
// the real Qwen3.6-27B MLP shape (H=5120, I=17408) with n=8 (top-k). This is the per-expert decode
// path today: k experts × one command buffer each per MoE layer. Compare its ns/op against
// BenchmarkMetalFusedMLPQ6DownBatch to read the #1382 lever's command-buffer-collapse speedup.
func BenchmarkMetalFusedMLPQ6DownSingle(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	const H, I, n = 5120, 17408, 8
	gws, uws, dws, x := buildFusedExpertBatch(b, H, I, n)
	ys := make([][]float32, n)
	for e := range ys {
		ys[e] = make([]float32, H)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for e := 0; e < n; e++ {
			metalgemm.FusedMLPQ6Down(gws[e], uws[e], dws[e], x, ys[e])
		}
	}
	b.StopTimer()
	if secs := b.Elapsed().Seconds(); secs > 0 {
		b.ReportMetric(secs/float64(b.N)*1e3, "ms/layer")
	}
}

// BenchmarkMetalFusedMLPQ6DownBatch times the same n experts through ONE FusedMLPQ6DownBatch command
// buffer at the real shape. If the batched ms/layer collapses well below n× the single ms — the way
// BenchmarkMetalQ4KGemvBatch showed the GEMV batch is 5.2× — the #1382 wiring turns that into a live
// decode speedup on the mlp_decode cost the MAC-QWEN36 diagnosis named as dominant.
func BenchmarkMetalFusedMLPQ6DownBatch(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	const H, I, n = 5120, 17408, 8
	gws, uws, dws, x := buildFusedExpertBatch(b, H, I, n)
	ycat := make([]float32, n*H)
	// Trust check before timing: the batch must agree with a single expert on row 0.
	single := make([]float32, H)
	metalgemm.FusedMLPQ6Down(gws[0], uws[0], dws[0], x, single)
	metalgemm.FusedMLPQ6DownBatch(gws, uws, dws, x, ycat)
	for o := 0; o < H; o++ {
		if d := ycat[o] - single[o]; d > 1e-3 || d < -1e-3 {
			b.Fatalf("batch row0[%d]=%g != single %g (offset binding wrong)", o, ycat[o], single[o])
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metalgemm.FusedMLPQ6DownBatch(gws, uws, dws, x, ycat)
	}
	b.StopTimer()
	if secs := b.Elapsed().Seconds(); secs > 0 {
		b.ReportMetric(secs/float64(b.N)*1e3, "ms/layer")
	}
}
