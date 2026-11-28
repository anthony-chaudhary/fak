package compute

// General device top-k gating with softmax/sigmoid scoring and optional weight
// renormalization: the pure-Go host reference of moe_topk_gating.cu.
//
// ADAPT of vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41
// (csrc/moe/topk_softmax_kernels.cu:80,142,165,280, Apache-2.0): topkGating,
// moeTopK, moeSoftmax, and moeSigmoid re-expressed for general non-V4.1 MoE
// routers.
//
// Reconnaissance record (gated on second MoE model landing):
//   - Existing served MoE routers in internal/model use either V4.1
//     sqrt(softplus) argmax-topk (internal/compute/cuda_v41_router.cu,
//     k_v41_router_argmax_topk) or GLM grouped sigmoid routing
//     (internal/model/glm5next_forward.go, noaux_tc).
//   - This pure-Go reference compiles and is verified in every build while
//     moe_topk_gating.cu stays unwired from cuda_kernels.cu until qualified on
//     a served non-V4.1 MoE model. Do not touch k_v41_router_argmax_topk or
//     v41_router*.go.
//   - [SW-VERIFIED] host contract only: no GPU execution or throughput claim.

import (
	"errors"
	"fmt"
	"math"
)

// MoeGatingScoring selects the expert activation applied to router logits.
type MoeGatingScoring int

const (
	// MoeGatingSoftmax applies numerically stable row-wise softmax over experts.
	MoeGatingSoftmax MoeGatingScoring = iota
	// MoeGatingSigmoid applies elementwise logistic sigmoid over experts.
	MoeGatingSigmoid
)

// MoeTopKGatingMaxExperts bounds expert counts aligned with single-CTA block
// MoE align/gating conventions.
const MoeTopKGatingMaxExperts = 4096

var (
	// ErrMoeTopKGatingGeometry is the closed refusal for malformed token,
	// expert, top-k, scoring, bias, or scaling arguments.
	ErrMoeTopKGatingGeometry = errors.New("compute: moe topk gating invalid geometry")
	// ErrMoeTopKGatingNonFinite is the closed refusal for NaN or Inf logits or biases.
	ErrMoeTopKGatingNonFinite = errors.New("compute: moe topk gating non-finite input")
)

// MoeTopKGatingConfig configures general (non-grouped) MoE top-k gating.
type MoeTopKGatingConfig struct {
	NumExperts          int
	TopK                int
	Scoring             MoeGatingScoring
	Renormalize         bool
	CorrectionBias      []float32 // optional [NumExperts]; affects selection only (noaux style)
	RoutedScalingFactor float32   // 0 defaults to 1.0; negative or non-finite is refused
}

// MoeTopKGatingResult contains row-major [numTokens, TopK] router outputs.
type MoeTopKGatingResult struct {
	Weights []float32
	Indices []int32
}

// MoeTopKGating scores logits [numTokens, cfg.NumExperts], picks the top-k
// experts per token (deterministic tie-break: higher selection key wins, ties
// go to the lower expert index), and optionally renormalizes and scales weights.
func MoeTopKGating(logits []float32, numTokens int, cfg MoeTopKGatingConfig) (*MoeTopKGatingResult, error) {
	if numTokens <= 0 || cfg.NumExperts <= 0 || cfg.NumExperts > MoeTopKGatingMaxExperts {
		return nil, fmt.Errorf("%w: tokens=%d experts=%d", ErrMoeTopKGatingGeometry, numTokens, cfg.NumExperts)
	}
	if cfg.TopK <= 0 || cfg.TopK > cfg.NumExperts {
		return nil, fmt.Errorf("%w: topK=%d experts=%d", ErrMoeTopKGatingGeometry, cfg.TopK, cfg.NumExperts)
	}
	if cfg.Scoring != MoeGatingSoftmax && cfg.Scoring != MoeGatingSigmoid {
		return nil, fmt.Errorf("%w: scoring=%d", ErrMoeTopKGatingGeometry, cfg.Scoring)
	}
	if len(logits) != numTokens*cfg.NumExperts {
		return nil, fmt.Errorf("%w: logits len=%d want %d", ErrMoeTopKGatingGeometry, len(logits), numTokens*cfg.NumExperts)
	}
	if len(cfg.CorrectionBias) != 0 && len(cfg.CorrectionBias) != cfg.NumExperts {
		return nil, fmt.Errorf("%w: correctionBias len=%d want %d", ErrMoeTopKGatingGeometry, len(cfg.CorrectionBias), cfg.NumExperts)
	}
	for i, b := range cfg.CorrectionBias {
		if math.IsNaN(float64(b)) || math.IsInf(float64(b), 0) {
			return nil, fmt.Errorf("%w: correctionBias[%d]=%v", ErrMoeTopKGatingNonFinite, i, b)
		}
	}
	scale := cfg.RoutedScalingFactor
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || scale < 0 {
		return nil, fmt.Errorf("%w: routedScalingFactor=%v", ErrMoeTopKGatingGeometry, scale)
	}
	if scale == 0 {
		scale = 1.0
	}

	weights := make([]float32, numTokens*cfg.TopK)
	indices := make([]int32, numTokens*cfg.TopK)
	scores := make([]float64, cfg.NumExperts)
	choices := make([]float64, cfg.NumExperts)
	taken := make([]bool, cfg.NumExperts)
	hasBias := len(cfg.CorrectionBias) == cfg.NumExperts

	for t := 0; t < numTokens; t++ {
		row := logits[t*cfg.NumExperts : (t+1)*cfg.NumExperts]
		rowMax := math.Inf(-1)
		for e, v := range row {
			f := float64(v)
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("%w: token %d expert %d logit=%v", ErrMoeTopKGatingNonFinite, t, e, v)
			}
			if f > rowMax {
				rowMax = f
			}
		}

		switch cfg.Scoring {
		case MoeGatingSoftmax:
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
		case MoeGatingSigmoid:
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

		var wsum float64
		for k := 0; k < cfg.TopK; k++ {
			best := -1
			for e := 0; e < cfg.NumExperts; e++ {
				if taken[e] {
					continue
				}
				if best < 0 || choices[e] > choices[best] {
					best = e
				}
			}
			taken[best] = true
			indices[t*cfg.TopK+k] = int32(best)
			wsum += scores[best]
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

	return &MoeTopKGatingResult{
		Weights: weights,
		Indices: indices,
	}, nil
}
