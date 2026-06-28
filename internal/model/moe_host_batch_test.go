package model

import (
	"math"
	"math/rand"
	"testing"
)

// TestBatchedExpertDeltaMatchesLoop pins lever 2's core invariant: batchedExpertDelta
// (one parFor per projection across experts) is BIT-IDENTICAL to the per-expert loop
// (q4kMatRows gate/up -> SwiGLU -> kQuantMatRows down, gate-weighted sum). It runs both
// int8-on and int8-off, asserting max|Δ|==0 — the batched dispatch only reassigns which
// core computes a row, never the per-row reduction, so the result must match exactly.
func TestBatchedExpertDeltaMatchesLoop(t *testing.T) {
	const H, MI, K = 768, 512, 4 // small but realistic shapes; in multiples of 256
	cfg := Config{HiddenSize: H, IntermediateSize: MI, MoEIntermediateSize: MI}
	rng := rand.New(rand.NewSource(99))

	mkQ4K := func(out, in int) *q4kTensor {
		nblk := in / qkK
		raw := make([]byte, out*nblk*q4kBlockBytes)
		blk := make([]byte, q4kBlockBytes)
		for o := 0; o < out; o++ {
			for b := 0; b < nblk; b++ {
				randQ4KBlock(rng, blk)
				copy(raw[(o*nblk+b)*q4kBlockBytes:], blk)
			}
		}
		return quantizeQ4KFromRaw(raw, out, in)
	}
	mkQ5K := func(out, in int) *kQuantTensor {
		nblk := in / qkK
		raw := make([]byte, out*nblk*q5kBlockBytes)
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
		return quantizeKQuantFromRaw(raw, out, in, kindQ5K)
	}

	gate := make([]*q4kTensor, K)
	up := make([]*q4kTensor, K)
	down := make([]*kQuantTensor, K)
	for e := 0; e < K; e++ {
		gate[e] = mkQ4K(MI, H)
		up[e] = mkQ4K(MI, H)
		down[e] = mkQ5K(H, MI)
	}
	xn := make([]float32, H)
	for i := range xn {
		xn[i] = float32(rng.NormFloat64())
	}
	picks := make([]routePick, K)
	for i := range picks {
		picks[i] = routePick{expert: i, weight: float32(rng.NormFloat64())}
	}

	ref := func() []float32 {
		delta := make([]float32, H)
		for i, pk := range picks {
			g := q4kMatRows(gate[i], xn)
			u := q4kMatRows(up[i], xn)
			for j := 0; j < MI; j++ {
				g[j] = act(g[j], cfg) * u[j]
			}
			d := kQuantMatRows(down[i], g)
			for j := 0; j < H; j++ {
				delta[j] += pk.weight * d[j]
			}
		}
		return delta
	}

	for _, int8on := range []bool{false, true} {
		setQ4KSDOTForTest(int8on)
		setKQuantSDOTForTest(int8on)
		want := ref()
		got := make([]float32, H)
		batchedExpertDelta(cfg, picks, gate, up, down, xn, got)
		q4kSDOTForce, kQuantSDOTForce = 0, 0
		var maxAbs float64
		for j := 0; j < H; j++ {
			if d := math.Abs(float64(got[j] - want[j])); d > maxAbs {
				maxAbs = d
			}
		}
		if maxAbs != 0 {
			t.Fatalf("int8=%v: batched vs loop max|Δ|=%g, want 0 (bit-identity)", int8on, maxAbs)
		}
		t.Logf("int8=%v: batched == loop, max|Δ|=0", int8on)
	}
}
