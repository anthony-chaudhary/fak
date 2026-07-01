package model

import (
	"math"
	"math/rand"
	"testing"
)

// TestMoEFFNBatchedMatchesLoop pins the Mixtral/Qwen3-MoE decode lever: moeFFN.apply's
// resident-host batched path (hostBatchedGLMExperts, ONE parFor per projection across the
// top-k experts) is BIT-IDENTICAL to the per-expert loop it replaces. This is the moeFFN
// analogue of TestBatchedExpertDeltaMatchesLoop (the primitive-level test) — it drives the
// FULL moeFFN.apply dispatch with a residentKernel and resident q4kw/kqw experts, so it
// proves the WIRING (mat.(residentKernel) gate + xn []float32 + hostBatchedGLMExperts) routes
// a Qwen3.6-style q4_k_m checkpoint (gate/up Q4_K in q4kw, down Q6_K in kqw) through the
// batched path, and that the result equals the expert-loop delta.
//
// The reference is the same q4kMatRows/kQuantMatRows reduction moeFFN's expertSwiGLU loop
// runs through residentMatRows, accumulated in route order — so max|Δ| must be exactly 0.
// (The Metal sessionQ4KKernel batched path — q4kFusedMLPBatch — is device-only, so its parity
// is the on-device decode token-match gate, not this host unit test.)
func TestMoEFFNBatchedMatchesLoop(t *testing.T) {
	const H, MI, E, K = 256, 256, 6, 2 // dims in multiples of qkK (256); realistic top-k
	cfg := Config{
		HiddenSize: H, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 2,
		IntermediateSize: MI, MoEIntermediateSize: MI, VocabSize: 4,
		RMSNormEps: 1e-5, RopeTheta: 10000,
		NumExperts: E, NumExpertsPerTok: K, NormTopKProb: true, EOSTokenID: -1,
	}
	rng := rand.New(rand.NewSource(4242))

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
	mkQ6K := func(out, in int) *kQuantTensor {
		nblk := in / qkK
		raw := make([]byte, out*nblk*q6kBlockBytes)
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
		return quantizeKQuantFromRaw(raw, out, in, kindQ6K)
	}

	// A bare model with resident experts populated directly (the q4_k_m residency shape:
	// gate/up Q4_K -> q4kw, down Q6_K -> kqw) plus a resident E-row router so route() runs
	// through residentMatRows exactly as the live decode does.
	m := &Model{Cfg: cfg}
	m.q4kw = map[string]*q4kTensor{}
	m.kqw = map[string]*kQuantTensor{}
	m.q4kw[routerName(0)] = mkQ4K(E, H)
	for e := 0; e < E; e++ {
		m.q4kw[expertName(0, e, "gate_proj.weight")] = mkQ4K(MI, H)
		m.q4kw[expertName(0, e, "up_proj.weight")] = mkQ4K(MI, H)
		m.kqw[expertName(0, e, "down_proj.weight")] = mkQ6K(H, MI)
	}

	mat := residentKernel{m}
	xn := make([]float32, H)
	for i := range xn {
		xn[i] = float32(rng.NormFloat64())
	}

	picks := route(m, 0, xn, mat)
	if len(picks) != K {
		t.Fatalf("router returned %d picks, want top-k=%d", len(picks), K)
	}

	// Reference: the exact per-expert loop moeFFN.apply ran before the batched wire-up —
	// expertSwiGLU per pick (residentMatRows gate/up -> SwiGLU -> down), gate-weighted sum
	// in route order. This is what the batched delta must equal bit-for-bit.
	wantLoop := make([]float32, H)
	for _, pk := range picks {
		out := expertSwiGLU(m, 0, pk.expert, xn, mat)
		for i := 0; i < H; i++ {
			wantLoop[i] += pk.weight * out[i]
		}
	}

	// The live path: moeFFN.apply now tries hostBatchedGLMExperts first (residentKernel gate).
	got := moeFFN{}.apply(m, 0, xn, mat)

	// Prove the batched path actually FIRED (not silently declined to the loop): call it
	// directly and confirm it returns true for this resident config.
	probe := make([]float32, H)
	if !m.hostBatchedGLMExperts(0, xn, probe, picks) {
		t.Fatalf("hostBatchedGLMExperts declined for a resident q4_k_m expert set — the lever did not fire")
	}

	var maxAbs float64
	for i := 0; i < H; i++ {
		if d := math.Abs(float64(got[i] - wantLoop[i])); d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs != 0 {
		t.Fatalf("moeFFN batched vs per-expert loop max|Δ|=%g, want 0 (bit-identity)", maxAbs)
	}
	// Guard against a vacuous test: the delta must be non-trivial.
	var norm float64
	for _, v := range got {
		norm += float64(v) * float64(v)
	}
	if norm == 0 {
		t.Fatalf("delta is all-zero; test is vacuous (experts contributed nothing)")
	}
	t.Logf("moeFFN batched == per-expert loop, max|Δ|=0 over %d picks / %d experts", len(picks), E)
}
