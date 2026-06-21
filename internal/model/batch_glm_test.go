package model

import (
	"math"
	"testing"
)

// batch_glm_test.go — GLM MoE native batched decode correctness.
//
// batchDecodeFastPathOK excludes any MoE config (batch.go: cfg.IsMoE()), so a GLM
// model — MoE by construction — always takes BatchSession's per-user serial
// fallback (Seqs[b].Step). That fallback IS the multi-user decode a DGX serves;
// it was untested for GLM. This proves it is bit-identical to B independent
// serial Sessions, so GLM native multi-user decode is correct today. The fused
// MoE+DSA batch GEMM (the aggregate-throughput speedup) is the remaining lever
// (GLM-5.2-NATIVE-ENGINE-GAP gap #6 / #4) and is a SPEED change over this proven
// CORRECTNESS baseline, not a behavior change.

// TestGLMMoeBatchedDecodeMatchesSerial runs B GLM-MoE users through one
// BatchSession (PrefillEach + StepBatch) and asserts bit-identical logits to B
// independent serial Sessions, across prefill and several decode steps.
func TestGLMMoeBatchedDecodeMatchesSerial(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		NumExperts: 4, NumExpertsPerTok: 2, NormTopKProb: true,
		TieWordEmbeddings: true, EOSTokenID: -1,
		ModelType: "glm_moe", Architectures: []string{"GlmMoeForCausalLM"},
	}
	m := NewSyntheticMoE(cfg)
	if !m.Cfg.IsMoE() {
		t.Fatalf("synthetic GLM config is not MoE; the MoE serial-fallback path is not exercised")
	}
	if batchDecodeFastPathOK(m.Cfg, false) {
		t.Fatalf("GLM MoE unexpectedly on the batch fast path; this test targets the serial fallback")
	}
	V := cfg.VocabSize
	B := 5

	// Distinct prompts of distinct lengths => users sit at distinct absolute
	// positions, exercising per-user RoPE and per-user cache-length attention.
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		n := 3 + b*2
		p := make([]int, n)
		for i := range p {
			p[i] = (b*97 + i*31 + 5) % V
		}
		prompts[b] = p
	}

	// Reference: B independent serial sessions.
	ref := make([]*Session, B)
	refLogits := make([][]float32, B)
	for b := 0; b < B; b++ {
		ref[b] = m.NewSession()
		refLogits[b] = ref[b].Prefill(prompts[b])
	}

	// Batch: one BatchSession over the same prompts.
	bs := m.NewBatchSession(B)
	batLogits := bs.PrefillEach(prompts)

	assertBitEqual := func(step int, a, c [][]float32) {
		t.Helper()
		for b := 0; b < B; b++ {
			if len(a[b]) != len(c[b]) {
				t.Fatalf("step %d user %d: logit len %d != %d", step, b, len(a[b]), len(c[b]))
			}
			for i := range a[b] {
				if math.Float32bits(a[b][i]) != math.Float32bits(c[b][i]) {
					t.Fatalf("step %d user %d logit[%d]: serial %v != batched %v (NOT bit-identical)",
						step, b, i, a[b][i], c[b][i])
				}
			}
		}
	}
	assertBitEqual(0, refLogits, batLogits)

	// Several lockstep decode steps; each user feeds a distinct id per step.
	for step := 1; step <= 4; step++ {
		ids := make([]int, B)
		for b := 0; b < B; b++ {
			ids[b] = (b*53 + step*29 + 7) % V
		}
		want := make([][]float32, B)
		for b := 0; b < B; b++ {
			want[b] = ref[b].Step(ids[b])
		}
		got := bs.StepBatch(ids)
		assertBitEqual(step, want, got)
	}
}

// TestGLMDsaBatchedDecodeMatchesSerial is the GLM-5.2 (glm_moe_dsa) analogue of the
// dense-MoE test above. It exercises the ACTUAL GLM-5.2 attention architecture — MLA
// (q_a/q_b/kv_a/kv_b) + Dynamic Sparse Attention with a learned indexer + IndexShare —
// through BatchSession, where the prior test only covered the standard-attention
// GLM-MoE FFN.
//
// This is also a regression test for a real mis-routing bug. The batch fast-path gate
// excluded MoE configs but NOT the GLM-DSA attention arch, so a DENSE glm_moe_dsa (the
// synthetic / cmd/pipelinegen form, NumExperts==0) satisfied batchPreNormFastPathOK and
// was routed onto the standard q/k/v panel GEMM, which panics on the q_proj weight a
// GLM-DSA checkpoint does not have. Real GLM-5.2 is MoE and so escaped only incidentally;
// the fix excludes isGLMMoeDsa() explicitly. This test pins that both the dense and the
// MoE GLM-DSA forms take the per-user serial path and stay bit-identical to independent
// Sessions across prefill + several decode steps, at distinct per-user positions.
func TestGLMDsaBatchedDecodeMatchesSerial(t *testing.T) {
	for _, tc := range []struct {
		name    string
		withMoE bool
	}{
		{"dense", false},
		{"moe", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{
				HiddenSize: 32, NumLayers: 3, NumHeads: 4, NumKVHeads: 4, HeadDim: 8,
				IntermediateSize: 64, VocabSize: 41, RMSNormEps: 1e-5, RopeTheta: 10000,
				ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"},
				QLoraRank: 32, KVLoraRank: 32, QKNopeHeadDim: 4, QKRopeHeadDim: 4, VHeadDim: 8,
				IndexNHeads: 4, IndexHeadDim: 8, IndexTopK: 2,
				IndexerTypes: []string{"full", "shared", "full"},
				EOSTokenID:   -1,
			}
			if tc.withMoE {
				cfg.NumExperts = 4
				cfg.NumExpertsPerTok = 2
				cfg.NormTopKProb = true
			}
			if !cfg.isGLMMoeDsa() {
				t.Fatalf("config is not glm_moe_dsa")
			}
			// The whole point: GLM-DSA must NOT be on the shared-panel fast path (its
			// MLA + sparse attention is not the standard q/k/v the panel GEMM assumes).
			if batchDecodeFastPathOK(cfg, false) || batchRectFastPathOK(cfg, false) {
				t.Fatalf("glm_moe_dsa (%s) unexpectedly on a batch fast path; it must take the serial fallback", tc.name)
			}

			m := NewSyntheticGLMDsa(cfg)
			V := cfg.VocabSize
			B := 5

			// Distinct prompts of distinct lengths => users sit at distinct absolute
			// positions, exercising per-user RoPE + per-user DSA top-k over each user's
			// own index/KV cache length.
			prompts := make([][]int, B)
			for b := 0; b < B; b++ {
				n := 3 + b*2
				p := make([]int, n)
				for i := range p {
					p[i] = (b*97 + i*31 + 5) % V
				}
				prompts[b] = p
			}

			// Reference: B independent serial GLM-DSA sessions.
			ref := make([]*Session, B)
			refLogits := make([][]float32, B)
			for b := 0; b < B; b++ {
				ref[b] = m.NewSession()
				refLogits[b] = ref[b].Prefill(prompts[b])
			}

			// Batch: one BatchSession over the same prompts (takes the serial fallback).
			bs := m.NewBatchSession(B)
			batLogits := bs.PrefillEach(prompts)

			assertBitEqual := func(step int, a, c [][]float32) {
				t.Helper()
				for b := 0; b < B; b++ {
					if len(a[b]) != len(c[b]) {
						t.Fatalf("step %d user %d: logit len %d != %d", step, b, len(a[b]), len(c[b]))
					}
					for i := range a[b] {
						if math.Float32bits(a[b][i]) != math.Float32bits(c[b][i]) {
							t.Fatalf("step %d user %d logit[%d]: serial %v != batched %v (NOT bit-identical)",
								step, b, i, a[b][i], c[b][i])
						}
					}
				}
			}
			assertBitEqual(0, refLogits, batLogits)

			for step := 1; step <= 4; step++ {
				ids := make([]int, B)
				for b := 0; b < B; b++ {
					ids[b] = (b*53 + step*29 + 7) % V
				}
				want := make([][]float32, B)
				for b := 0; b < B; b++ {
					want[b] = ref[b].Step(ids[b])
				}
				got := bs.StepBatch(ids)
				assertBitEqual(step, want, got)
			}
		})
	}
}

// TestGLMDsaGenerateBatchMatchesSerial closes the loop on GLM-5.2 multi-user serving: the
// lockstep greedy GenerateBatch (PrefillEach + StepBatch + per-user argmax feedback) must
// yield, per user, the EXACT token sequence a serial Session.Generate produces. This is the
// end-to-end analogue of TestGenerateBatchMatchesSerial for the glm_moe_dsa (MLA + DSA)
// arch — the real serving path fleetserve runs, output-identical to running each user alone.
func TestGLMDsaGenerateBatchMatchesSerial(t *testing.T) {
	for _, tc := range []struct {
		name    string
		withMoE bool
	}{
		{"dense", false},
		{"moe", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{
				HiddenSize: 32, NumLayers: 3, NumHeads: 4, NumKVHeads: 4, HeadDim: 8,
				IntermediateSize: 64, VocabSize: 41, RMSNormEps: 1e-5, RopeTheta: 10000,
				ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"},
				QLoraRank: 32, KVLoraRank: 32, QKNopeHeadDim: 4, QKRopeHeadDim: 4, VHeadDim: 8,
				IndexNHeads: 4, IndexHeadDim: 8, IndexTopK: 2,
				IndexerTypes: []string{"full", "shared", "full"},
				EOSTokenID:   -1, // never early-stop: decode exactly n
			}
			if tc.withMoE {
				cfg.NumExperts = 4
				cfg.NumExpertsPerTok = 2
				cfg.NormTopKProb = true
			}
			m := NewSyntheticGLMDsa(cfg)
			V := cfg.VocabSize
			B := 4

			prompts := make([][]int, B)
			for b := 0; b < B; b++ {
				p := make([]int, 4+b)
				for i := range p {
					p[i] = (b*53 + i*29 + 3) % V
				}
				prompts[b] = p
			}

			const n = 10
			want := make([][]int, B)
			for b := 0; b < B; b++ {
				want[b] = m.NewSession().Generate(prompts[b], n)
			}
			got := m.NewBatchSession(B).GenerateBatch(prompts, n)

			for b := 0; b < B; b++ {
				if len(got[b]) != len(want[b]) {
					t.Fatalf("user %d generated %d tokens, serial generated %d: got=%v want=%v",
						b, len(got[b]), len(want[b]), got[b], want[b])
				}
				for i := range want[b] {
					if got[b][i] != want[b][i] {
						t.Fatalf("user %d token %d: batch %d != serial %d", b, i, got[b][i], want[b][i])
					}
				}
			}
		})
	}
}
