package model

import (
	"math/rand"
	"testing"
)

// glm_expert_dispatch_bench_test.go — witnesses the MoE expert-dispatch scaling finding
// behind docs/notes/GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md (lever 2). A GLM-5.2 decode
// step runs each routed expert's gate/up/down GEMV through its OWN parFor (one per
// expert×proj — ~24 tiny dispatches per MoE layer). parFor's per-dispatch cost (the global
// dispatch mutex + worker wake + busy-wait drain + per-chunk atomic cursor contention)
// dominates when each call is small, capping parallel scaling well below the core count.
//
// These benchmarks compare K experts' gate GEMVs run as K separate parFors (today's host
// path) against the SAME total work as ONE parFor over a fused [K*MI, H] tensor (the batched
// dispatch lever 2 would install). The batched form pays the dispatch cost once. On a 32-core
// amd64 box the fused form measured ~1.8x the looped throughput — the lever-2 headroom. Run:
//
//	go test ./internal/model/ -run x -bench 'GLMExpertDispatch' -benchmem
//
// The fused tensor is bit-identical per row to the per-expert GEMVs (parFor only reassigns
// which core computes which row), so a real batched path stays exact — same property as
// TestParallelMatchesSerial.

func buildQ4KExpertBench(rng *rand.Rand, out, in int) *q4kTensor {
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			randQ4KBlock(rng, blk)
			off := (o*nblk + b) * q4kBlockBytes
			copy(raw[off:off+q4kBlockBytes], blk)
		}
	}
	return quantizeQ4KFromRaw(raw, out, in)
}

// benchGLMExpertDispatch times K gate-GEMVs of shape [MI,H] sharing one activation, either as
// K separate q4kMatRowsInto calls (looped) or one call over a fused [K*MI,H] tensor (batched).
func benchGLMExpertDispatch(b *testing.B, K, MI, H int, batched bool) {
	rng := rand.New(rand.NewSource(11))
	x := make([]float32, H)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	weightGiB := float64(K*MI*H/qkK*q4kBlockBytes) / (1 << 30)
	if batched {
		big := buildQ4KExpertBench(rng, K*MI, H)
		y := make([]float32, K*MI)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q4kMatRowsInto(big, x, y)
		}
	} else {
		gates := make([]*q4kTensor, K)
		for e := range gates {
			gates[e] = buildQ4KExpertBench(rng, MI, H)
		}
		y := make([]float32, MI)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, g := range gates {
				q4kMatRowsInto(g, x, y)
			}
		}
	}
	b.ReportMetric(weightGiB, "weightGiB")
}

// BenchmarkGLMExpertDispatchLooped is today's host path: one parFor per expert GEMV.
func BenchmarkGLMExpertDispatchLooped(b *testing.B) { benchGLMExpertDispatch(b, 8, 1536, 5120, false) }

// BenchmarkGLMExpertDispatchBatched is the lever-2 target: one parFor over all experts' rows.
func BenchmarkGLMExpertDispatchBatched(b *testing.B) { benchGLMExpertDispatch(b, 8, 1536, 5120, true) }
