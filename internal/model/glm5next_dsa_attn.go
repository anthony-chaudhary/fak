package model

import (
	"math"
)

// ExpandDSABlocksToTokens maps selected block indices to individual token indices.
// For block b, token indices are [b*stride, min((b+1)*stride, totalTokens)).
func ExpandDSABlocksToTokens(selectedBlocks []int, stride, totalTokens int) []int {
	if stride <= 0 {
		stride = 4
	}
	tokens := make([]int, 0, len(selectedBlocks)*stride)
	seen := make(map[int]bool, len(selectedBlocks)*stride)

	for _, b := range selectedBlocks {
		start := b * stride
		end := start + stride
		if end > totalTokens {
			end = totalTokens
		}
		for t := start; t < end; t++ {
			if !seen[t] {
				seen[t] = true
				tokens = append(tokens, t)
			}
		}
	}
	return tokens
}

// ComputeGLM5NextDSASparseMixer computes multi-head sparse mixer over the set of selected tokens:
// q: [numHeads * qkNopeHeadDim] for the current query token.
// allK: [totalTokens * numHeads * qkNopeHeadDim] (No-PE keys)
// allV: [totalTokens * numHeads * vHeadDim] (values)
// selectedTokens: slice of token indices to attend to.
// currentToken: index of current query (causal bound: tokens > currentToken are masked out).
// Returns out: [numHeads * vHeadDim].
func ComputeGLM5NextDSASparseMixer(
	q []float32,
	allK, allV []float32,
	selectedTokens []int,
	currentToken int,
	numHeads, qkNopeHeadDim, vHeadDim int,
) []float32 {
	out := make([]float32, numHeads*vHeadDim)
	if len(selectedTokens) == 0 {
		return out
	}

	scale := float32(1.0 / math.Sqrt(float64(qkNopeHeadDim)))
	kStride := numHeads * qkNopeHeadDim
	vStride := numHeads * vHeadDim

	// Filter causally valid tokens
	validTokens := make([]int, 0, len(selectedTokens))
	for _, t := range selectedTokens {
		if t <= currentToken && t >= 0 {
			validTokens = append(validTokens, t)
		}
	}
	if len(validTokens) == 0 {
		return out
	}

	S := len(validTokens)
	scores := make([]float32, S)

	for h := 0; h < numHeads; h++ {
		qHead := q[h*qkNopeHeadDim : (h+1)*qkNopeHeadDim]

		// 1. Dot product scores
		maxScore := float32(-math.MaxFloat32)
		for i, t := range validTokens {
			kOff := t*kStride + h*qkNopeHeadDim
			kHead := allK[kOff : kOff+qkNopeHeadDim]

			var dot float32
			for d := 0; d < qkNopeHeadDim; d++ {
				dot += qHead[d] * kHead[d]
			}
			s := dot * scale
			scores[i] = s
			if s > maxScore {
				maxScore = s
			}
		}

		// 2. Softmax
		var sumExp float64
		for i := 0; i < S; i++ {
			expVal := math.Exp(float64(scores[i] - maxScore))
			scores[i] = float32(expVal)
			sumExp += expVal
		}
		invSum := float32(1.0 / sumExp)
		for i := 0; i < S; i++ {
			scores[i] *= invSum
		}

		// 3. Weighted sum of values
		outHead := out[h*vHeadDim : (h+1)*vHeadDim]
		for i, t := range validTokens {
			weight := scores[i]
			if weight == 0 {
				continue
			}
			vOff := t*vStride + h*vHeadDim
			vHead := allV[vOff : vOff+vHeadDim]
			for d := 0; d < vHeadDim; d++ {
				outHead[d] += weight * vHead[d]
			}
		}
	}

	return out
}
