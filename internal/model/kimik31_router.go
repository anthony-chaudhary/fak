package model

// kimik31_router.go — the Kimi-K3.1 native MoE router host oracle (ticket
// 1611-kimi-k31, port leaf sglang-kimi-k31-4 host side; goal Wave 1.4/1.5). K3.1
// routes 896 experts, top-16, with a sigmoid scoring function and the noaux_tc
// contract: expert SELECTION ranks by `sigmoid(logit) + e_score_correction_bias`,
// while the returned WEIGHTS use the UNBIASED `sigmoid(logit)`, renormalized (the
// config's moe_renormalize) and scaled by routed_scaling_factor (1.0 for K3.1, a
// wired no-op). The existing GLM5Next router (RouteGLM5NextMoE) has no correction
// bias — bias ranking is load-bearing for expert selection — so K3.1 earns its own
// host oracle here.
//
// SOURCE: the noaux_tc bias-ranked top-k is the DeepSeek-V3/Kimi contract already
// transcribed in fak for V4.1 (internal/model/v41_router.go v41RouteWithNormalization:
// "Selection uses the noaux_tc score-plus-bias choice, weights use the unbiased
// score", reference DeepSeek-V4.1-Flash inference/model.py Gate.forward, MIT); K3.1
// uses the same contract with sigmoid scores, 896 experts, top-16, num_expert_group 1
// (grouped top-k degenerates to plain top-k at one group), and routed_scaling 1.0.
//
// WHAT THIS OWNS: the host routing contract (selection, weights, scaling) for one
// token. WHAT THIS DOES NOT OWN: the expert GEMM execution (own leaves) and the
// CUDA router kernel (the reference dispatches by capability).

import (
	"fmt"
	"sort"
)

// KimiK31RouterConfig holds the K3.1 router geometry. K3.1 ships Experts 896,
// TopK 16, RouteScale 1.0, Renormalize true, GroupCount 1.
type KimiK31RouterConfig struct {
	Experts     int
	TopK        int
	RouteScale  float32
	Renormalize bool
	// GroupCount is the grouped-top-k group count (num_expert_group). K3.1 ships
	// 1, which degenerates grouped top-k to a plain top-k over all experts.
	GroupCount int
	// TopKGroup is the per-group selection count when GroupCount > 1. Unused at
	// GroupCount == 1.
	TopKGroup int
}

// KimiK31RoutePick is one selected expert: its index and its (unbiased,
// renormalized, scaled) router weight.
type KimiK31RoutePick struct {
	Expert int
	Weight float32
}

// ErrKimiK31RouterShape is the closed sentinel for fail-closed router shape and
// domain validation failures.
var ErrKimiK31RouterShape = fmt.Errorf("model: kimi-k31 router shape mismatch")

// RouteKimiK31MoE routes one token. logits is the router's raw pre-activation
// output ([Experts]); correctionBias is the checkpoint's e_score_correction_bias
// ([Experts], optional — pass nil for a config that carries none). It returns the
// top-K picks ordered by descending selection score.
//
// Contract (noaux_tc): selection ranks by sigmoid(logit)+bias; weights are the
// unbiased sigmoid(logit); with Renormalize the weights are divided by their sum;
// every weight is multiplied by RouteScale. A non-finite logit or bias, a width
// mismatch, or a non-positive weight sum is refused with ErrKimiK31RouterShape.
func RouteKimiK31MoE(logits, correctionBias []float32, cfg KimiK31RouterConfig) ([]KimiK31RoutePick, error) {
	if cfg.Experts <= 0 || cfg.TopK <= 0 || cfg.TopK > cfg.Experts {
		return nil, fmt.Errorf("%w: geometry experts=%d topk=%d", ErrKimiK31RouterShape, cfg.Experts, cfg.TopK)
	}
	if cfg.GroupCount < 1 {
		return nil, fmt.Errorf("%w: group count %d must be >= 1", ErrKimiK31RouterShape, cfg.GroupCount)
	}
	if len(logits) != cfg.Experts {
		return nil, fmt.Errorf("%w: logits width %d != experts %d", ErrKimiK31RouterShape, len(logits), cfg.Experts)
	}
	if len(correctionBias) != 0 && len(correctionBias) != cfg.Experts {
		return nil, fmt.Errorf("%w: correction bias width %d != experts %d", ErrKimiK31RouterShape, len(correctionBias), cfg.Experts)
	}

	type candidate struct {
		expert    int
		score     float32 // unbiased sigmoid(logit) — the weight
		selection float32 // score + bias — the ranking key
	}
	cands := make([]candidate, cfg.Experts)
	for i, z := range logits {
		if !finite32(z) {
			return nil, fmt.Errorf("%w: non-finite logit at expert %d", ErrKimiK31RouterShape, i)
		}
		score := kdaSigmoid(z)
		sel := score
		if len(correctionBias) != 0 {
			b := correctionBias[i]
			if !finite32(b) {
				return nil, fmt.Errorf("%w: non-finite correction bias at expert %d", ErrKimiK31RouterShape, i)
			}
			sel += b
		}
		if !finite32(score) || !finite32(sel) {
			return nil, fmt.Errorf("%w: non-finite score at expert %d", ErrKimiK31RouterShape, i)
		}
		cands[i] = candidate{expert: i, score: score, selection: sel}
	}

	// Selection: descending selection score, ties by lower expert id (the stable
	// DeepSeek rank order). At GroupCount == 1 the grouped top-k is a plain top-k.
	// Grouped selection (GroupCount > 1) is currently refused: no K3.1 checkpoint
	// ships a multi-group router (num_expert_group 1), so inventing the group walk
	// here would be unverifiable; a config that asks for it fails closed.
	if cfg.GroupCount > 1 {
		return nil, fmt.Errorf("%w: grouped routing (group_count %d) is not implemented", ErrKimiK31RouterShape, cfg.GroupCount)
	}
	sort.SliceStable(cands, func(a, b int) bool {
		if cands[a].selection != cands[b].selection { // NaN compares unequal: it sorts last.
			return cands[a].selection > cands[b].selection
		}
		return cands[a].expert < cands[b].expert
	})

	picks := make([]KimiK31RoutePick, cfg.TopK)
	var sum float32
	for i := 0; i < cfg.TopK; i++ {
		picks[i] = KimiK31RoutePick{Expert: cands[i].expert, Weight: cands[i].score}
		sum += cands[i].score
	}
	if !finite32(sum) || sum <= 0 {
		return nil, fmt.Errorf("%w: selected weight sum must be finite and positive, got %g", ErrKimiK31RouterShape, sum)
	}
	for i := range picks {
		w := picks[i].Weight
		if cfg.Renormalize {
			w /= sum
		}
		w *= cfg.RouteScale
		if !finite32(w) {
			return nil, fmt.Errorf("%w: non-finite weight for expert %d", ErrKimiK31RouterShape, picks[i].Expert)
		}
		picks[i].Weight = w
	}
	return picks, nil
}
