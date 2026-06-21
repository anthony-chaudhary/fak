package model

import (
	"math"
	"strings"
)

const glmDsaInnerNormEps = 1e-6

// glmDsaAttentionOutputFromTopK reproduces the GLM-MoE-DSA attention sublayer
// from already-selected DSA top-k indices. It wires the real GLM tensor geometry:
// q_a/q_b MLA query projection, kv_a/kv_b key/value projection, GLM's interleaved
// RoPE for main attention, sparse causal masking by top-k, and o_proj.
//
// It deliberately accepts topK as input. Computing the learned DSA indexer trace
// is a separate rung; this helper is the native attention-output rung that the
// optional HF trace can now verify.
func glmDsaAttentionOutputFromTopK(m *Model, layer int, hidden []float32, seq int, topK [][]int) ([]float32, bool) {
	xn, ok := glmDsaNormalizeLayerInput(m, layer, hidden, seq)
	if !ok {
		return nil, false
	}
	return glmDsaAttentionOutputFromTopKNormed(m, layer, xn, seq, topK)
}

func glmDsaAttentionOutputFromTopKNormed(m *Model, layer int, xnFlat []float32, seq int, topK [][]int) ([]float32, bool) {
	cfg := m.Cfg
	if !cfg.isGLMMoeDsa() || seq <= 0 || len(xnFlat) != seq*cfg.HiddenSize || len(topK) != seq {
		return nil, false
	}
	H, nH := cfg.HiddenSize, cfg.NumHeads
	qLora, kvLora := cfg.QLoraRank, cfg.KVLoraRank
	qkNope, qkRope, vHead := cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.VHeadDim
	qkHead := qkNope + qkRope
	if H == 0 || nH == 0 || qLora == 0 || kvLora == 0 || qkNope == 0 || qkRope == 0 || vHead == 0 || qkRope%2 != 0 {
		return nil, false
	}

	lp := layerPrefix(layer)
	ap := lp + "self_attn."
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

	scale := float32(1.0 / math.Sqrt(float64(qkHead)))
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

func glmDsaTopKIndices(m *Model, layer int, hidden []float32, seq int) ([][]int, bool) {
	xn, ok := glmDsaNormalizeLayerInput(m, layer, hidden, seq)
	if !ok {
		return nil, false
	}
	return glmDsaTopKIndicesNormed(m, layer, xn, seq)
}

func glmDsaTopKIndicesNormed(m *Model, layer int, xnFlat []float32, seq int) ([][]int, bool) {
	cfg := m.Cfg
	if !cfg.isGLMMoeDsa() || seq <= 0 || len(xnFlat) != seq*cfg.HiddenSize {
		return nil, false
	}
	H, qLora := cfg.HiddenSize, cfg.QLoraRank
	indexHeads, indexDim := cfg.IndexNHeads, cfg.IndexHeadDim
	qkRope := cfg.QKRopeHeadDim
	if H == 0 || qLora == 0 || indexHeads == 0 || indexDim == 0 || qkRope <= 0 || qkRope > indexDim || qkRope%2 != 0 {
		return nil, false
	}

	lp := layerPrefix(layer)
	ap := lp + "self_attn."
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

func glmDsaSelectedCausalKeys(row []int, queryPos, seq int) ([]int, bool) {
	if len(row) == 0 {
		return nil, false
	}
	selected := make([]int, 0, len(row))
	seen := make(map[int]struct{}, len(row))
	for _, keyPos := range row {
		if keyPos < 0 || keyPos >= seq {
			return nil, false
		}
		if keyPos > queryPos {
			continue
		}
		if _, dup := seen[keyPos]; dup {
			return nil, false
		}
		seen[keyPos] = struct{}{}
		selected = append(selected, keyPos)
	}
	return selected, true
}

func glmDsaIndexerIsShared(cfg Config, layer int) bool {
	return glmDsaIndexerKind(cfg, layer) == "shared"
}

func glmDsaIndexerIsFull(cfg Config, layer int) bool {
	return glmDsaIndexerKind(cfg, layer) == "full"
}

func glmDsaIndexerKind(cfg Config, layer int) string {
	if layer < 0 || layer >= len(cfg.IndexerTypes) {
		return "full"
	}
	switch strings.ToLower(strings.TrimSpace(cfg.IndexerTypes[layer])) {
	case "", "full":
		return "full"
	case "shared", "share":
		return "shared"
	default:
		return "unknown"
	}
}

func cloneIndexRow(in []int) []int {
	return append([]int(nil), in...)
}

func glmDsaNormalizeLayerInput(m *Model, layer int, hidden []float32, seq int) ([]float32, bool) {
	cfg := m.Cfg
	if !cfg.isGLMMoeDsa() || seq <= 0 || len(hidden) != seq*cfg.HiddenSize {
		return nil, false
	}
	H := cfg.HiddenSize
	w := m.tensor(layerPrefix(layer) + "input_layernorm.weight")
	out := make([]float32, len(hidden))
	for t := 0; t < seq; t++ {
		copy(out[t*H:(t+1)*H], rmsnorm(hidden[t*H:(t+1)*H], w, float32(cfg.RMSNormEps)))
	}
	return out, true
}

func splitFlatRows(flat []float32, rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	for r := 0; r < rows; r++ {
		out[r] = flat[r*cols : (r+1)*cols]
	}
	return out
}

func glmDsaApplyInterleavedRoPE(x, cos, sin []float32) []float32 {
	half := len(x) / 2
	out := make([]float32, len(x))
	for j := 0; j < half; j++ {
		a, b := x[2*j], x[2*j+1]
		out[j] = float32(a*cos[j]) - float32(b*sin[j])
		out[j+half] = float32(b*cos[j]) + float32(a*sin[j])
	}
	return out
}

func glmDsaApplyIndexerRoPE(x, cos, sin []float32) {
	half := len(x) / 2
	applyRopeRow(x, cos[:half], sin[:half])
}

func addOptionalBias(y, b []float32) {
	if b != nil {
		addBias(y, b)
	}
}

func float32To64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

func float64To32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}
