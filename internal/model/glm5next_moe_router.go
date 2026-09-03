package model

import (
	"sort"
)

// GLM5NextMoERouteResult holds the top-8 routed expert indices and normalized weights.
type GLM5NextMoERouteResult struct {
	ExpertIndices []int
	Weights       []float32
}

// RouteGLM5NextMoE computes router logits = WRouter * x for 288 experts,
// applies sigmoid gating to each logit, selects top-8 experts, and normalizes
// their weights to sum to 1.0.
func RouteGLM5NextMoE(
	x []float32,
	WRouter []float32,
	numExperts, topK int,
) GLM5NextMoERouteResult {
	if numExperts <= 0 {
		numExperts = 288
	}
	if topK <= 0 {
		topK = 8
	}
	hiddenSize := len(x)
	if hiddenSize == 0 || len(WRouter) < numExperts*hiddenSize {
		return GLM5NextMoERouteResult{}
	}

	type expertScore struct {
		idx   int
		logit float32
	}

	scores := make([]expertScore, numExperts)
	for e := 0; e < numExperts; e++ {
		var dot float32
		rowOff := e * hiddenSize
		for j := 0; j < hiddenSize; j++ {
			dot += WRouter[rowOff+j] * x[j]
		}
		scores[e] = expertScore{
			idx:   e,
			logit: dot,
		}
	}

	sort.Slice(scores, func(i, j int) bool {
		if scores[i].logit == scores[j].logit {
			return scores[i].idx > scores[j].idx
		}
		return scores[i].logit > scores[j].logit
	})

	k := min(topK, numExperts)
	res := GLM5NextMoERouteResult{
		ExpertIndices: make([]int, k),
		Weights:       make([]float32, k),
	}

	var sumProb float32
	for i := 0; i < k; i++ {
		res.ExpertIndices[i] = scores[i].idx
		prob := kdaSigmoid(scores[i].logit)
		res.Weights[i] = prob
		sumProb += prob
	}

	if sumProb > 0 {
		invSum := 1.0 / sumProb
		for i := 0; i < k; i++ {
			res.Weights[i] *= invSum
		}
	}

	return res
}
