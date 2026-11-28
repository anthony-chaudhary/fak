package compute

// Group-constrained MoE top-k routing (noaux_tc style): pure-Go host reference
// of moe_grouped_topk.cu.
//
// ADAPT of vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41
// (csrc/moe/grouped_topk_kernels.cu:480,639,922,1095, Apache-2.0):
// grouped_topk_fused_kernel family re-expressed cleanly for fak.
//
// Pre-build reconnaissance (served model config check):
//   - internal/model/config.go declares NGroup and TopKGroup, and
//     internal/model/glm5next_forward.go implements noaux_tc grouped routing
//     (top-2 sum per group -> topk_group selection -> top-k experts + optional
//     renorm + routed_scaling_factor), confirming grouped routing is an active
//     served model configuration.
//   - moe_grouped_topk.cu remains standalone/unwired from cuda_kernels.cu until
//     device qualification. Do not touch V4.1 hash/gate router seams or
//     bitonic_topk.go.
//   - [SW-VERIFIED] host contract only: no GPU execution or throughput claim.

import (
	"errors"
	"fmt"
	"math"
)

// GroupedTopKScoring selects the router score transform for grouped top-k.
type GroupedTopKScoring int

const (
	// GroupedTopKSoftmax computes stable row-wise softmax scores over experts.
	GroupedTopKSoftmax GroupedTopKScoring = iota
	// GroupedTopKSigmoid computes elementwise logistic sigmoid scores over experts.
	GroupedTopKSigmoid
)

var (
	// ErrGroupedTopKGeometry is the closed refusal for malformed grouped routing
	// parameters, group divisibility, capacity, bias length, or scale.
	ErrGroupedTopKGeometry = errors.New("compute: moe grouped topk invalid geometry")
	// ErrGroupedTopKNonFinite is the closed refusal for NaN/Inf logits or biases.
	ErrGroupedTopKNonFinite = errors.New("compute: moe grouped topk non-finite input")
)

// GroupedTopKConfig configures group-constrained MoE routing.
type GroupedTopKConfig struct {
	NumExperts          int
	TopK                int
	NumExpertGroups     int
	TopKGroup           int
	Scoring             GroupedTopKScoring
	Renormalize         bool
	RoutedScalingFactor float32   // 0 defaults to 1.0; negative or non-finite is refused
	CorrectionBias      []float32 // optional [NumExperts]; affects group/expert selection only
}

// GroupedTopKResult contains row-major [numTokens, TopK] routed indices and weights.
type GroupedTopKResult struct {
	Weights []float32
	Indices []int32
}

// GroupedTopK executes group-constrained top-k routing per token:
//  1. Compute expert scores (softmax or sigmoid) and selection keys
//     (score + optional CorrectionBias).
//  2. Score each of NumExpertGroups groups by summing its top-2 selection keys
//     (or lone key when ExpertsPerGroup == 1) and keep the top TopKGroup groups
//     (deterministic tie-break: higher group score, then lower group index).
//  3. Select TopK experts across all kept groups (cross-group fill; higher
//     selection key, then lower expert index).
//  4. Optionally renormalize weights and multiply by RoutedScalingFactor.
func GroupedTopK(logits []float32, numTokens int, cfg GroupedTopKConfig) (*GroupedTopKResult, error) {
	if numTokens <= 0 || cfg.NumExperts <= 0 || cfg.NumExpertGroups <= 0 {
		return nil, fmt.Errorf("%w: tokens=%d experts=%d groups=%d",
			ErrGroupedTopKGeometry, numTokens, cfg.NumExperts, cfg.NumExpertGroups)
	}
	if cfg.NumExperts%cfg.NumExpertGroups != 0 {
		return nil, fmt.Errorf("%w: experts=%d not divisible by groups=%d",
			ErrGroupedTopKGeometry, cfg.NumExperts, cfg.NumExpertGroups)
	}
	if cfg.TopKGroup <= 0 || cfg.TopKGroup > cfg.NumExpertGroups {
		return nil, fmt.Errorf("%w: topkGroup=%d groups=%d",
			ErrGroupedTopKGeometry, cfg.TopKGroup, cfg.NumExpertGroups)
	}
	expertsPerGroup := cfg.NumExperts / cfg.NumExpertGroups
	if cfg.TopK <= 0 || cfg.TopK > cfg.TopKGroup*expertsPerGroup {
		return nil, fmt.Errorf("%w: topK=%d exceeds kept group capacity %d",
			ErrGroupedTopKGeometry, cfg.TopK, cfg.TopKGroup*expertsPerGroup)
	}
	if cfg.Scoring != GroupedTopKSoftmax && cfg.Scoring != GroupedTopKSigmoid {
		return nil, fmt.Errorf("%w: scoring=%d", ErrGroupedTopKGeometry, cfg.Scoring)
	}
	if len(logits) != numTokens*cfg.NumExperts {
		return nil, fmt.Errorf("%w: logits len=%d want %d", ErrGroupedTopKGeometry, len(logits), numTokens*cfg.NumExperts)
	}
	if len(cfg.CorrectionBias) != 0 && len(cfg.CorrectionBias) != cfg.NumExperts {
		return nil, fmt.Errorf("%w: correctionBias len=%d want %d",
			ErrGroupedTopKGeometry, len(cfg.CorrectionBias), cfg.NumExperts)
	}
	for i, b := range cfg.CorrectionBias {
		if math.IsNaN(float64(b)) || math.IsInf(float64(b), 0) {
			return nil, fmt.Errorf("%w: correctionBias[%d]=%v", ErrGroupedTopKNonFinite, i, b)
		}
	}
	scale := cfg.RoutedScalingFactor
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || scale < 0 {
		return nil, fmt.Errorf("%w: routedScalingFactor=%v", ErrGroupedTopKGeometry, scale)
	}
	if scale == 0 {
		scale = 1.0
	}

	weights := make([]float32, numTokens*cfg.TopK)
	indices := make([]int32, numTokens*cfg.TopK)
	scores := make([]float64, cfg.NumExperts)
	choices := make([]float64, cfg.NumExperts)
	groupScore := make([]float64, cfg.NumExpertGroups)
	keepGroup := make([]bool, cfg.NumExpertGroups)
	taken := make([]bool, cfg.NumExperts)
	hasBias := len(cfg.CorrectionBias) == cfg.NumExperts

	for t := 0; t < numTokens; t++ {
		row := logits[t*cfg.NumExperts : (t+1)*cfg.NumExperts]
		rowMax := math.Inf(-1)
		for e, v := range row {
			f := float64(v)
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("%w: token %d expert %d logit=%v", ErrGroupedTopKNonFinite, t, e, v)
			}
			if f > rowMax {
				rowMax = f
			}
		}

		switch cfg.Scoring {
		case GroupedTopKSoftmax:
			var sumExp float64
			for e, v := range row {
				ex := math.Exp(float64(v) - rowMax)
				scores[e] = ex
				sumExp += ex
			}
			invSum := 1.0 / sumExp
			for e := range scores {
				scores[e] *= invSum
			}
		case GroupedTopKSigmoid:
			for e, v := range row {
				scores[e] = 1.0 / (1.0 + math.Exp(-float64(v)))
			}
		}

		for e := range scores {
			if hasBias {
				choices[e] = scores[e] + float64(cfg.CorrectionBias[e])
			} else {
				choices[e] = scores[e]
			}
			taken[e] = false
		}

		if cfg.NumExpertGroups == 1 {
			keepGroup[0] = true
		} else {
			for g := 0; g < cfg.NumExpertGroups; g++ {
				keepGroup[g] = false
				base := g * expertsPerGroup
				if expertsPerGroup == 1 {
					groupScore[g] = choices[base]
					continue
				}
				best, second := math.Inf(-1), math.Inf(-1)
				for i := 0; i < expertsPerGroup; i++ {
					c := choices[base+i]
					if c > best {
						second = best
						best = c
					} else if c > second {
						second = c
					}
				}
				groupScore[g] = best + second
			}
			for kg := 0; kg < cfg.TopKGroup; kg++ {
				bestG := -1
				for g := 0; g < cfg.NumExpertGroups; g++ {
					if keepGroup[g] {
						continue
					}
					if bestG < 0 || groupScore[g] > groupScore[bestG] {
						bestG = g
					}
				}
				keepGroup[bestG] = true
			}
		}

		var wsum float64
		for k := 0; k < cfg.TopK; k++ {
			bestE := -1
			for e := 0; e < cfg.NumExperts; e++ {
				if taken[e] || !keepGroup[e/expertsPerGroup] {
					continue
				}
				if bestE < 0 || choices[e] > choices[bestE] {
					bestE = e
				}
			}
			taken[bestE] = true
			indices[t*cfg.TopK+k] = int32(bestE)
			wsum += scores[bestE]
		}

		scale64 := float64(scale)
		norm := 1.0
		if cfg.Renormalize && wsum > 0 {
			norm = 1.0 / wsum
		}
		for k := 0; k < cfg.TopK; k++ {
			id := indices[t*cfg.TopK+k]
			weights[t*cfg.TopK+k] = float32(scores[id] * norm * scale64)
		}
	}

	return &GroupedTopKResult{
		Weights: weights,
		Indices: indices,
	}, nil
}
