package model

import (
	"fmt"
	"math"
	"sort"
)

// v4RouteError is a fail-closed admission error for the scored-layer V4
// router. Hash-routed layers are deliberately a separate seam: callers must
// supply the artifact's token-to-expert table rather than guessing from logits.
type v4RouteError struct {
	Field  string
	Reason string
}

func (e *v4RouteError) Error() string {
	return fmt.Sprintf("model: DeepSeek V4 route %s: %s", e.Field, e.Reason)
}

// v4ScoredRoute implements the score-based Gate.forward contract from the
// pinned DeepSeek-V4-Pro inference artifact. Selection uses
// sqrt(softplus(logit))+bias while returned weights use the unbiased score,
// normalized across the selected experts and multiplied by routeScale.
func v4ScoredRoute(logits, correctionBias []float32, topK int, routeScale float32) ([]routePick, error) {
	if len(logits) == 0 {
		return nil, &v4RouteError{Field: "logits", Reason: "empty"}
	}
	if len(correctionBias) != 0 && len(correctionBias) != len(logits) {
		return nil, &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("width %d does not match logits width %d", len(correctionBias), len(logits))}
	}
	if topK <= 0 || topK > len(logits) {
		return nil, &v4RouteError{Field: "top_k", Reason: fmt.Sprintf("%d outside [1,%d]", topK, len(logits))}
	}
	if !finite32(routeScale) || routeScale <= 0 {
		return nil, &v4RouteError{Field: "route_scale", Reason: fmt.Sprintf("must be finite and positive, got %g", routeScale)}
	}

	raw := make([]float32, len(logits))
	choice := make([]float32, len(logits))
	for i, z := range logits {
		if !finite32(z) {
			return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
		}
		// max(z,0)+log1p(exp(-abs(z))) is softplus without overflow.
		zf := float64(z)
		softplus := math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf)))
		score := float32(math.Sqrt(softplus))
		if !finite32(score) {
			return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite score at expert %d", i)}
		}
		raw[i] = score
		choice[i] = score
		if len(correctionBias) != 0 {
			bias := correctionBias[i]
			if !finite32(bias) {
				return nil, &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
			}
			choice[i] += bias
			if !finite32(choice[i]) {
				return nil, &v4RouteError{Field: "selection_score", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
			}
		}
	}

	indices := make([]int, len(logits))
	for i := range indices {
		indices[i] = i
	}
	// Deterministic equivalent of topk: descending selection score, then the
	// lower expert index for ties.
	sort.SliceStable(indices, func(i, j int) bool {
		return choice[indices[i]] > choice[indices[j]]
	})

	picks := make([]routePick, topK)
	var sum float32
	for i := 0; i < topK; i++ {
		expert := indices[i]
		picks[i] = routePick{expert: expert, weight: raw[expert]}
		sum += raw[expert]
	}
	if !finite32(sum) || sum <= 0 {
		return nil, &v4RouteError{Field: "normalization", Reason: fmt.Sprintf("selected score sum must be finite and positive, got %g", sum)}
	}
	for i := range picks {
		picks[i].weight = picks[i].weight / sum * routeScale
		if !finite32(picks[i].weight) {
			return nil, &v4RouteError{Field: "weight", Reason: fmt.Sprintf("non-finite result for expert %d", picks[i].expert)}
		}
	}
	return picks, nil
}

func finite32(v float32) bool {
	return !float32IsNaN(v) && !float32IsInf(v)
}

func float32IsNaN(v float32) bool { return v != v }
func float32IsInf(v float32) bool {
	return v > math.MaxFloat32 || v < -math.MaxFloat32
}

// v4HashRouteUnsupported gives the first three V4 layers an explicit refusal
// until the pinned tid2eid table is connected to the live token path.
func v4HashRouteUnsupported(layer int) error {
	return &v4RouteError{Field: "hash_layer", Reason: fmt.Sprintf("layer %d requires token-to-expert tid2eid routing", layer)}
}
