package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"sort"
)

// dsaIndexScores computes GLM-MoE-DSA's indexer score formula from already
// projected indexer tensors. HF computes, per query/key:
//
//	sum_h weights[q,h] * relu(scale * dot(index_q[q,h], index_k[k]))
//
// The learned projections (wq_b, wk, weights_proj, norms, and RoPE) happen before
// this helper. Keeping that boundary explicit lets tests witness the native DSA
// scoring and sparse-attention semantics without claiming the full GLM layer is
// executable.
func dsaIndexScores(indexQ [][][]float64, indexK [][]float64, indexWeights [][]float64, scale float64) ([][]float64, bool) {
	if len(indexQ) == 0 || len(indexK) == 0 || len(indexWeights) != len(indexQ) || scale <= 0 {
		return nil, false
	}
	nHeads := len(indexQ[0])
	dim := len(indexK[0])
	if nHeads == 0 || dim == 0 {
		return nil, false
	}
	out := make([][]float64, len(indexQ))
	for qi := range indexQ {
		if len(indexQ[qi]) != nHeads || len(indexWeights[qi]) != nHeads {
			return nil, false
		}
		out[qi] = make([]float64, len(indexK))
		for ki := range indexK {
			if len(indexK[ki]) != dim {
				return nil, false
			}
			var score float64
			for h := 0; h < nHeads; h++ {
				if len(indexQ[qi][h]) != dim {
					return nil, false
				}
				headScore := dot64(indexQ[qi][h], indexK[ki]) * scale
				if math.IsNaN(headScore) {
					return nil, false
				}
				if headScore < 0 {
					headScore = 0
				}
				score += indexWeights[qi][h] * headScore
			}
			out[qi][ki] = score
		}
	}
	return out, true
}

// dsaTopKIndices selects the top-k key positions for each query after applying
// the DSA causal mask. It models the cache-relevant part of GLM-MoE-DSA's
// indexer: score every candidate key, mask keys whose absolute position is
// greater than the query position, then keep top-k indices. The real GLM indexer
// produces these scores from learned q/k/weight projections; this helper starts
// from the already-computed scores so tests can witness causality and reuse
// without claiming a native DSA forward kernel.
func dsaTopKIndices(scores [][]float64, queryPositions, keyPositions []int, topK int) ([][]int, bool) {
	if topK <= 0 || len(scores) != len(queryPositions) || len(keyPositions) == 0 {
		return nil, false
	}
	out := make([][]int, len(scores))
	for qi, row := range scores {
		if len(row) != len(keyPositions) {
			return nil, false
		}
		type candidate struct {
			keyPos int
			score  float64
		}
		candidates := make([]candidate, 0, len(keyPositions))
		for ki, keyPos := range keyPositions {
			if keyPos > queryPositions[qi] {
				continue
			}
			score := row[ki]
			if math.IsNaN(score) {
				continue
			}
			candidates = append(candidates, candidate{keyPos: keyPos, score: score})
		}
		if len(candidates) == 0 {
			return nil, false
		}
		sort.SliceStable(candidates, func(i, j int) bool {
			if candidates[i].score == candidates[j].score {
				return candidates[i].keyPos < candidates[j].keyPos
			}
			return candidates[i].score > candidates[j].score
		})
		n := topK
		if n > len(candidates) {
			n = len(candidates)
		}
		out[qi] = make([]int, n)
		for i := 0; i < n; i++ {
			out[qi][i] = candidates[i].keyPos
		}
	}
	return out, true
}

// dsaSparseAttention applies the DSA-selected sparse key set to the actual
// attention output. topK holds key positions, not dense row offsets, mirroring
// the indexer/cache boundary: only selected, causal positions are materialized
// into the softmax. A non-causal or unknown key selection fails closed.
func dsaSparseAttention(query, key, value [][][]float64, queryPositions, keyPositions []int, topK [][]int, scale float64) ([][][]float64, bool) {
	if len(query) == 0 || len(key) == 0 || len(key) != len(value) ||
		len(queryPositions) != len(query) || len(keyPositions) != len(key) ||
		len(topK) != len(query) || scale <= 0 {
		return nil, false
	}
	nHeads := len(query[0])
	if nHeads == 0 || len(value[0]) == 0 {
		return nil, false
	}
	qDim := len(query[0][0])
	vDim := len(value[0][0])
	if qDim == 0 || vDim == 0 {
		return nil, false
	}
	posToKey := make(map[int]int, len(keyPositions))
	for ki, pos := range keyPositions {
		if _, exists := posToKey[pos]; exists {
			return nil, false
		}
		posToKey[pos] = ki
		if len(key[ki]) != nHeads || len(value[ki]) != nHeads {
			return nil, false
		}
		for h := 0; h < nHeads; h++ {
			if len(key[ki][h]) != qDim || len(value[ki][h]) != vDim {
				return nil, false
			}
		}
	}

	out := make([][][]float64, len(query))
	for qi := range query {
		if len(query[qi]) != nHeads || len(topK[qi]) == 0 {
			return nil, false
		}
		selected := make([]int, 0, len(topK[qi]))
		seen := make(map[int]struct{}, len(topK[qi]))
		for _, keyPos := range topK[qi] {
			if keyPos > queryPositions[qi] {
				return nil, false
			}
			if _, dup := seen[keyPos]; dup {
				return nil, false
			}
			seen[keyPos] = struct{}{}
			ki, ok := posToKey[keyPos]
			if !ok {
				return nil, false
			}
			selected = append(selected, ki)
		}
		out[qi] = make([][]float64, nHeads)
		for h := 0; h < nHeads; h++ {
			if len(query[qi][h]) != qDim {
				return nil, false
			}
			scores := make([]float64, len(selected))
			maxScore := math.Inf(-1)
			for i, ki := range selected {
				score := dot64(query[qi][h], key[ki][h]) * scale
				if math.IsNaN(score) {
					return nil, false
				}
				scores[i] = score
				if score > maxScore {
					maxScore = score
				}
			}
			var denom float64
			for i := range scores {
				scores[i] = math.Exp(scores[i] - maxScore)
				denom += scores[i]
			}
			if denom == 0 || math.IsNaN(denom) {
				return nil, false
			}
			out[qi][h] = make([]float64, vDim)
			for i, ki := range selected {
				w := scores[i] / denom
				for d := 0; d < vDim; d++ {
					out[qi][h][d] += w * value[ki][h][d]
				}
			}
		}
	}
	return out, true
}

// dsaIndexShare expands full-indexer decisions over a layer plan where "full"
// layers compute indices and "shared" layers reuse the immediately preceding
// full layer's top-k. This is the IndexShare contract GLM-5.2 relies on for
// every-four-layer sharing; it is metadata/control-flow only, not attention math.
func dsaIndexShare(layerTypes []string, fullByLayer map[int][][]int) (map[int][][]int, bool) {
	out := make(map[int][][]int, len(layerTypes))
	var current [][]int
	for layer, typ := range layerTypes {
		switch typ {
		case "full":
			decision, ok := fullByLayer[layer]
			if !ok || len(decision) == 0 {
				return nil, false
			}
			current = cloneIndexDecision(decision)
			out[layer] = cloneIndexDecision(current)
		case "shared":
			if current == nil {
				return nil, false
			}
			out[layer] = cloneIndexDecision(current)
		default:
			return nil, false
		}
	}
	return out, true
}

func dsaIndexDigest(indices [][]int) string {
	h := sha256.New()
	var buf [8]byte
	for _, row := range indices {
		binary.BigEndian.PutUint64(buf[:], uint64(len(row)))
		_, _ = h.Write(buf[:])
		for _, idx := range row {
			binary.BigEndian.PutUint64(buf[:], uint64(int64(idx)))
			_, _ = h.Write(buf[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func cloneIndexDecision(in [][]int) [][]int {
	out := make([][]int, len(in))
	for i := range in {
		out[i] = append([]int(nil), in[i]...)
	}
	return out
}

func dot64(a, b []float64) float64 {
	var out float64
	for i := range a {
		out += a[i] * b[i]
	}
	return out
}
