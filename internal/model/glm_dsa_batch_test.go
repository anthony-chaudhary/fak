package model

import (
	"math"
	"testing"
)

// glm_dsa_batch_test.go — the bit-exact witness for the batched GLM-5.2 (glm_moe_dsa)
// prefill projections. glmDsaAttentionOutputFromTopKNormed and glmDsaTopKIndicesNormed now
// hoist their MLA (q_a/q_b/kv_a/kv_b/o_proj) and indexer (wq_b/wk/weights_proj) projections
// out of the per-token loop into full-sequence residentMatMulBatch GEMMs — each weight row
// read once and reused across all seq prompt tokens, the same batched-prefill lever
// prefill_batch.go gives dense/GQA, now for GLM-5.2's MLA + DSA. These tests pin that the
// batched path is byte-for-byte the per-token residentMatRows path it replaced (matMulBatch
// shares fdot with matRows), so the oracle rungs stay exact and no forward bit changed.

func syntheticGLMDsaForBatch() *Model {
	cfg := Config{
		HiddenSize: 32, NumLayers: 3, NumHeads: 4, NumKVHeads: 4, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 41, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"},
		QLoraRank: 32, KVLoraRank: 32, QKNopeHeadDim: 4, QKRopeHeadDim: 4, VHeadDim: 8,
		IndexNHeads: 4, IndexHeadDim: 8, IndexTopK: 2,
		IndexerTypes: []string{"full", "shared", "full"},
		NumExperts:   4, NumExpertsPerTok: 2, NormTopKProb: true,
		EOSTokenID: -1,
	}
	return NewSyntheticGLMDsa(cfg)
}

// deterministicRows fills a [seq*width] panel with a small reproducible LCG pattern — the
// bit-exact comparison does not depend on the values being a "real" normalized hidden state.
func deterministicRows(seq, width int, seed uint32) []float32 {
	x := make([]float32, seq*width)
	s := seed | 1
	for i := range x {
		s = s*1664525 + 1013904223
		x[i] = (float32(s>>8)/float32(1<<24))*2 - 1 // in [-1,1)
	}
	return x
}

// TestResidentMatMulBatchMatchesPerToken proves the primitive every batched projection rides
// on: for each real GLM-DSA projection weight, residentMatMulBatch's row t is bit-for-bit the
// per-token residentMatRows(name, X[t]) it replaces, across several panel sizes.
func TestResidentMatMulBatchMatchesPerToken(t *testing.T) {
	m := syntheticGLMDsaForBatch()
	cfg := m.Cfg
	H, nH := cfg.HiddenSize, cfg.NumHeads
	qLora, kvLora := cfg.QLoraRank, cfg.KVLoraRank
	qkHead := cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	ap := layerPrefix(0) + "self_attn."

	cases := []struct {
		name    string
		out, in int
	}{
		{ap + "q_a_proj.weight", qLora, H},
		{ap + "q_b_proj.weight", nH * qkHead, qLora},
		{ap + "kv_a_proj_with_mqa.weight", kvLora + cfg.QKRopeHeadDim, H},
		{ap + "kv_b_proj.weight", nH * (cfg.QKNopeHeadDim + cfg.VHeadDim), kvLora},
		{ap + "o_proj.weight", H, nH * cfg.VHeadDim},
		{ap + "indexer.wq_b.weight", cfg.IndexNHeads * cfg.IndexHeadDim, qLora},
		{ap + "indexer.wk.weight", cfg.IndexHeadDim, H},
		{ap + "indexer.weights_proj.weight", cfg.IndexNHeads, H},
	}
	for _, P := range []int{1, 2, 5, 8} {
		for _, c := range cases {
			X := deterministicRows(P, c.in, uint32(P*131+c.out*7+c.in))
			got := m.residentMatMulBatch(c.name, X, c.out, c.in, P)
			for tk := 0; tk < P; tk++ {
				want := m.residentMatRows(c.name, X[tk*c.in:(tk+1)*c.in], c.out, c.in)
				for i := range want {
					if math.Float32bits(got[tk*c.out+i]) != math.Float32bits(want[i]) {
						t.Fatalf("P=%d %s row %d[%d]: batched %v != per-token %v (not bit-identical)",
							P, c.name, tk, i, got[tk*c.out+i], want[i])
					}
				}
			}
		}
	}
}

// --- per-token reference implementations (the pre-batch code) — the golden the batched
// production functions must reproduce bit-for-bit. Kept in the test, mirroring the
// batched-vs-serial reference pattern of batch_glm_test.go / TestPrefillBatchedMatchesSerial.

func glmDsaAttnOutputPerTokenRef(m *Model, layer int, xnFlat []float32, seq int, topK [][]int) ([]float32, bool) {
	cfg := m.Cfg
	H, nH := cfg.HiddenSize, cfg.NumHeads
	qLora, kvLora := cfg.QLoraRank, cfg.KVLoraRank
	qkNope, qkRope, vHead := cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.VHeadDim
	qkHead := qkNope + qkRope
	ap := layerPrefix(layer) + "self_attn."
	qANorm := m.tensor(ap + "q_a_layernorm.weight")
	kvANorm := m.tensor(ap + "kv_a_layernorm.weight")

	qStates := make([][]float32, seq)
	kStates := make([][]float32, seq)
	vStates := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xn := xnFlat[t*H : (t+1)*H]
		qResid := m.residentMatRows(ap+"q_a_proj.weight", xn, qLora, H)
		addOptionalBias(qResid, m.tensorOptional(ap+"q_a_proj.bias"))
		qResid = rmsnorm(qResid, qANorm, glmDsaInnerNormEps)
		qFull := m.residentMatRows(ap+"q_b_proj.weight", qResid, nH*qkHead, qLora)
		compressedKV := m.residentMatRows(ap+"kv_a_proj_with_mqa.weight", xn, kvLora+qkRope, H)
		addOptionalBias(compressedKV, m.tensorOptional(ap+"kv_a_proj_with_mqa.bias"))
		kvLatent := rmsnorm(compressedKV[:kvLora], kvANorm, glmDsaInnerNormEps)
		kvFull := m.residentMatRows(ap+"kv_b_proj.weight", kvLatent, nH*(qkNope+vHead), kvLora)
		kRotRaw := compressedKV[kvLora:]
		cos, sin := ropeRowForLayer(cfg, layer, t)
		qStates[t] = make([]float32, nH*qkHead)
		kStates[t] = make([]float32, nH*qkHead)
		vStates[t] = make([]float32, nH*vHead)
		for h := 0; h < nH; h++ {
			qSrc := qFull[h*qkHead : (h+1)*qkHead]
			qDst := qStates[t][h*qkHead : (h+1)*qkHead]
			copy(qDst[:qkNope], qSrc[:qkNope])
			copy(qDst[qkNope:], glmDsaApplyInterleavedRoPE(qSrc[qkNope:], cos, sin))
			kvSrc := kvFull[h*(qkNope+vHead) : (h+1)*(qkNope+vHead)]
			kDst := kStates[t][h*qkHead : (h+1)*qkHead]
			copy(kDst[:qkNope], kvSrc[:qkNope])
			copy(kDst[qkNope:], glmDsaApplyInterleavedRoPE(kRotRaw, cos, sin))
			copy(vStates[t][h*vHead:(h+1)*vHead], kvSrc[qkNope:])
		}
	}
	scale := float32(cfg.ropeAttentionFactor() / math.Sqrt(float64(qkHead)))
	out := make([]float32, seq*H)
	for t := 0; t < seq; t++ {
		selected, ok := glmDsaSelectedCausalKeys(topK[t], t, seq)
		if !ok || len(selected) == 0 {
			return nil, false
		}
		attnConcat := make([]float32, nH*vHead)
		for h := 0; h < nH; h++ {
			qh := qStates[t][h*qkHead : (h+1)*qkHead]
			scores := make([]float32, len(selected))
			for i, keyPos := range selected {
				kh := kStates[keyPos][h*qkHead : (h+1)*qkHead]
				scores[i] = dot(qh, kh) * scale
			}
			softmaxInPlace(scores)
			oh := attnConcat[h*vHead : (h+1)*vHead]
			for i, keyPos := range selected {
				vh := vStates[keyPos][h*vHead : (h+1)*vHead]
				w := scores[i]
				for d := 0; d < vHead; d++ {
					oh[d] += w * vh[d]
				}
			}
		}
		ot := m.residentMatRows(ap+"o_proj.weight", attnConcat, H, nH*vHead)
		addOptionalBias(ot, m.tensorOptional(ap+"o_proj.bias"))
		copy(out[t*H:(t+1)*H], ot)
	}
	return out, true
}

func glmDsaTopKPerTokenRef(m *Model, layer int, xnFlat []float32, seq int) ([][]int, bool) {
	cfg := m.Cfg
	H, qLora := cfg.HiddenSize, cfg.QLoraRank
	indexHeads, indexDim := cfg.IndexNHeads, cfg.IndexHeadDim
	qkRope := cfg.QKRopeHeadDim
	ap := layerPrefix(layer) + "self_attn."
	qANorm := m.tensor(ap + "q_a_layernorm.weight")
	kNormW := m.tensor(ap + "indexer.k_norm.weight")
	kNormB := m.tensor(ap + "indexer.k_norm.bias")

	indexQ := make([][][]float64, seq)
	indexK := make([][]float64, seq)
	indexWeights := make([][]float64, seq)
	for t := 0; t < seq; t++ {
		xn := xnFlat[t*H : (t+1)*H]
		qResid := m.residentMatRows(ap+"q_a_proj.weight", xn, qLora, H)
		addOptionalBias(qResid, m.tensorOptional(ap+"q_a_proj.bias"))
		qResid = rmsnorm(qResid, qANorm, glmDsaInnerNormEps)
		qFull := m.residentMatRows(ap+"indexer.wq_b.weight", qResid, indexHeads*indexDim, qLora)
		k := layernorm(m.residentMatRows(ap+"indexer.wk.weight", xn, indexDim, H), kNormW, kNormB, glmDsaInnerNormEps)
		weights := m.residentMatRows(ap+"indexer.weights_proj.weight", xn, indexHeads, H)
		weightScale := float32(1.0 / math.Sqrt(float64(indexHeads)))
		for i := range weights {
			weights[i] *= weightScale
		}
		cos, sin := ropeRowForLayer(cfg, layer, t)
		indexQ[t] = make([][]float64, indexHeads)
		for h := 0; h < indexHeads; h++ {
			head := append([]float32(nil), qFull[h*indexDim:(h+1)*indexDim]...)
			glmDsaApplyIndexerRoPE(head[:qkRope], cos, sin)
			indexQ[t][h] = float32To64(head)
		}
		k = append([]float32(nil), k...)
		glmDsaApplyIndexerRoPE(k[:qkRope], cos, sin)
		indexK[t] = float32To64(k)
		indexWeights[t] = float32To64(weights)
	}
	scores, ok := dsaIndexScores(indexQ, indexK, indexWeights, 1.0/math.Sqrt(float64(indexDim)))
	if !ok {
		return nil, false
	}
	positions := make([]int, seq)
	for i := range positions {
		positions[i] = i
	}
	return dsaTopKIndices(scores, positions, positions, cfg.IndexTopK)
}

// TestGLMDsaTopKBatchedMatchesPerTokenRef pins the batched indexer top-k selection identical
// to the per-token reference across several sequence lengths.
func TestGLMDsaTopKBatchedMatchesPerTokenRef(t *testing.T) {
	m := syntheticGLMDsaForBatch()
	H := m.Cfg.HiddenSize
	for _, seq := range []int{1, 2, 5, 9} {
		xnFlat := deterministicRows(seq, H, uint32(seq*17+3))
		got, ok := glmDsaTopKIndicesNormed(m, 0, xnFlat, seq)
		if !ok {
			t.Fatalf("seq=%d: batched top-k rejected", seq)
		}
		want, ok := glmDsaTopKPerTokenRef(m, 0, xnFlat, seq)
		if !ok {
			t.Fatalf("seq=%d: reference top-k rejected", seq)
		}
		if len(got) != len(want) {
			t.Fatalf("seq=%d: batched %d rows != reference %d", seq, len(got), len(want))
		}
		for q := range want {
			if len(got[q]) != len(want[q]) {
				t.Fatalf("seq=%d row %d: batched %v != reference %v", seq, q, got[q], want[q])
			}
			for i := range want[q] {
				if got[q][i] != want[q][i] {
					t.Fatalf("seq=%d row %d: batched %v != reference %v", seq, q, got[q], want[q])
				}
			}
		}
	}
}

// TestGLMDsaAttentionBatchedMatchesPerTokenRef pins the batched MLA attention output
// bit-for-bit identical to the per-token reference, feeding both the same top-k so the
// comparison isolates the projection batching from the index selection.
func TestGLMDsaAttentionBatchedMatchesPerTokenRef(t *testing.T) {
	m := syntheticGLMDsaForBatch()
	H := m.Cfg.HiddenSize
	for _, seq := range []int{1, 2, 5, 9} {
		xnFlat := deterministicRows(seq, H, uint32(seq*29+11))
		topK, ok := glmDsaTopKPerTokenRef(m, 0, xnFlat, seq)
		if !ok {
			t.Fatalf("seq=%d: reference top-k rejected", seq)
		}
		got, ok := glmDsaAttentionOutputFromTopKNormed(m, 0, xnFlat, seq, topK)
		if !ok {
			t.Fatalf("seq=%d: batched attention rejected", seq)
		}
		want, ok := glmDsaAttnOutputPerTokenRef(m, 0, xnFlat, seq, topK)
		if !ok {
			t.Fatalf("seq=%d: reference attention rejected", seq)
		}
		if len(got) != len(want) {
			t.Fatalf("seq=%d: batched len %d != reference %d", seq, len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("seq=%d out[%d]: batched %v != reference %v (not bit-identical)",
					seq, i, got[i], want[i])
			}
		}
	}
}
