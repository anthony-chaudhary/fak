//go:build cuda

package compute

import (
	"math"
	"testing"
)

// cuda_flash_test.go — the #486 op-level witness + microbench for the fused flash/online-softmax
// attention kernel (k_flash_attention) that replaced the naive one-block-per-head decode kernel
// on the live Attention path. Compiled only under -tags cuda; skips cleanly if no CUDA device is
// registered. It complements TestCUDAForwardMatchesRef (the multi-layer forward gate, whose
// Attention is now this flash kernel) by isolating attention alone:
//
//   1. Caps.FusedAttn is advertised true (the seam the model loop type-asserts).
//   2. Flash attention == the cpuref f32 Reference within the recorded Approx floor
//      (cudaFlashAttnCosineMin), across MHA / GQA / MQA head groupings (grp = nH/nKV).
//   3. Flash == the RETAINED naive device kernel (both f32, same device) — the online-softmax
//      reorder is numerically faithful to the batched softmax it replaced.
//
// The fused-vs-naive THROUGHPUT delta (the issue's "fused beats naive at representative nPos")
// is the BenchmarkCUDA{Flash,Naive}Attention pair below; tools/run_486_acceptance_on_gpu.sh turns
// their ns/op into the speedup verdict on a CUDA node. As with every device number in this package
// the realized cosine + speedup are measured there, not on the win32 build host.

// flashHeadCase is one attention shape: a head grouping (nH query heads over nKV KV heads) at a
// head dim and KV window length.
type flashHeadCase struct {
	name            string
	nH, nKV, hd, nP int
}

// buildAttnKV populates a single-layer KV cache on `be` with nP random post-RoPE K/V rows and
// returns a random query row plus the derived grp/scale, so cpuref and cuda run the SAME inputs
// through Backend.Attention. RoPE is irrelevant to the attention math under test, so kRaw==kRoPE
// (no eviction here); the rows are just fixed random vectors. Built through the Backend interface
// so the device path H2D-copies exactly the bytes the reference holds host-side.
func buildAttnKV(be Backend, g *lcg, c flashHeadCase) (Tensor, KVStore, int, float32) {
	w := c.nKV * c.hd
	grp := c.nH / c.nKV
	scale := float32(1.0 / math.Sqrt(float64(c.hd)))
	kv := be.NewKV(KVConfig{NumLayers: 1, NumKVHeads: c.nKV, HeadDim: c.hd, RopeTheta: 10000})
	for j := 0; j < c.nP; j++ {
		kRow := mkResident(be, []int{w}, rscale(g, w, 1.0))
		vRow := mkResident(be, []int{w}, rscale(g, w, 1.0))
		kv.AppendKV(0, kRow, kRow, vRow, j) // kRaw=kRoPE; layer 0 records pos
	}
	q := mkResident(be, []int{c.nH * c.hd}, rscale(g, c.nH*c.hd, 1.0))
	return q, kv, grp, scale
}

// TestCUDAFlashAttentionMatchesRef — the #486 op-level Approx gate. For each head grouping it runs
// one decode Attention through the fused flash kernel and compares the output to the cpuref f32
// Reference (cosine ≥ cudaFlashAttnCosineMin), and to the retained naive device kernel (same gate).
func TestCUDAFlashAttentionMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	if !cb.Caps().FusedAttn {
		t.Fatalf("#486: cuda backend must advertise Caps.FusedAttn=true (got FusedAttn=false)")
	}
	ref := Default() // cpu-ref

	cases := []flashHeadCase{
		{"mha", 8, 8, 16, 40},   // multi-head: every query head its own KV head
		{"gqa", 8, 2, 16, 64},   // grouped-query: 4 query heads share each KV head
		{"mqa", 8, 1, 32, 50},   // multi-query: all query heads share one KV head
		{"hd64", 16, 4, 64, 96}, // a wider head dim spanning >1 reduction stride
	}
	for _, c := range cases {
		var seed lcg = 486
		g := &seed // same seed per case so ref and cuda see identical random K/V/q

		qRef, kvRef, grp, scale := buildAttnKV(ref, g, c)
		oRef := ref.Read(ref.Attention(qRef, kvRef, 0, true, grp, scale))

		seed = 486 // rewind so the device cache gets the SAME bytes
		qCu, kvCu, _, _ := buildAttnKV(cb, &seed, c)
		oFlash := cb.Read(cb.Attention(qCu, kvCu, 0, true, grp, scale)) // fused flash path
		oNaive := cb.Read(cb.attentionNaive(qCu, kvCu, 0, grp, scale))  // retained naive baseline

		want := c.nH * c.hd
		if len(oRef) != want || len(oFlash) != want || len(oNaive) != want {
			t.Fatalf("%s: out len ref=%d flash=%d naive=%d want %d", c.name, len(oRef), len(oFlash), len(oNaive), want)
		}
		cFlashRef := cosine(oRef, oFlash)
		if cFlashRef < cudaFlashAttnCosineMin {
			t.Fatalf("%s: flash-vs-cpuref cosine %.6f < recorded floor %.4f", c.name, cFlashRef, cudaFlashAttnCosineMin)
		}
		cFlashNaive := cosine(oNaive, oFlash)
		if cFlashNaive < cudaFlashAttnCosineMin {
			t.Fatalf("%s: flash-vs-naive cosine %.6f < recorded floor %.4f", c.name, cFlashNaive, cudaFlashAttnCosineMin)
		}
		t.Logf("#486 flash attention parity [%s nH=%d nKV=%d hd=%d nPos=%d]: flash/ref cosine=%.8f (maxAbs=%.2e), flash/naive cosine=%.8f, gate=%.4f",
			c.name, c.nH, c.nKV, c.hd, c.nP, cFlashRef, maxAbsDelta(oRef, oFlash), cFlashNaive, cudaFlashAttnCosineMin)
	}
	t.Logf("#486 flash witness: fused k_flash_attention == cpuref within the recorded Approx floor across MHA/GQA/MQA. "+
		"Realized cosine + the fused-vs-naive speedup (BenchmarkCUDA{Flash,Naive}Attention) are measured on a CUDA node via "+
		"tools/run_486_acceptance_on_gpu.sh. device=%s tier=%s class=%s", cb.Name(), cb.Tier(), cb.Class())
}

// flashBenchCase is the representative decode-attention shape the fused-vs-naive microbench times:
// a Llama-ish 32 query heads over 8 KV heads (GQA), head dim 128, at a 1024-token KV window — the
// regime where the naive kernel's full global scores[nH*nPos] row + four passes cost most, and the
// flash kernel's single streaming pass with no scores row should win.
func flashBenchCase() flashHeadCase { return flashHeadCase{"bench", 32, 8, 128, 1024} }

// BenchmarkCUDAFlashAttention — the fused flash/online-softmax path (the live Attention). Its ns/op
// is compared against BenchmarkCUDANaiveAttention by tools/run_486_acceptance_on_gpu.sh.
func BenchmarkCUDAFlashAttention(b *testing.B) {
	cb := cudaTBOrSkip(b)
	var seed lcg = 486
	q, kv, grp, scale := buildAttnKV(cb, &seed, flashBenchCase())
	cb.Read(cb.Attention(q, kv, 0, true, grp, scale)) // warm: pool + scratch
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.Read(cb.Attention(q, kv, 0, true, grp, scale))
		cb.Recycle()
	}
	b.StopTimer()
}

// BenchmarkCUDANaiveAttention — the retained naive baseline (full global scores row, four passes),
// SAME inputs as the flash bench. The flash/naive ns/op ratio is the issue's "fused beats naive".
func BenchmarkCUDANaiveAttention(b *testing.B) {
	cb := cudaTBOrSkip(b)
	var seed lcg = 486
	q, kv, grp, scale := buildAttnKV(cb, &seed, flashBenchCase())
	cb.Read(cb.attentionNaive(q, kv, 0, grp, scale)) // warm
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.Read(cb.attentionNaive(q, kv, 0, grp, scale))
		cb.Recycle()
	}
	b.StopTimer()
}
