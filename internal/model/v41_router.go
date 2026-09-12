package model

import (
	"fmt"
	"math"
	"sort"
)

const (
	V41RouterExperts     = 384
	V41RouterTopK        = 6
	V41RouterSharedCount = 1
	V41RouterMoEWidth    = 2304
	V41RouterRouteScale  = 1.5
)

// v41RouterConfig is the admitted V4.1 routed+shared geometry.
type v41RouterConfig struct {
	Experts     int
	TopK        int
	SharedCount int
	RouteScale  float32
}

// v41DefaultRouterConfig returns the published V4.1 geometry.
func v41DefaultRouterConfig() v41RouterConfig {
	return v41RouterConfig{
		Experts:     V41RouterExperts,
		TopK:        V41RouterTopK,
		SharedCount: V41RouterSharedCount,
		RouteScale:  V41RouterRouteScale,
	}
}

// v41RouterConfigFromConfig reads the routed+shared geometry from a Config and
// refuses anything but the admitted 384/top-6/1-shared 1.5-scale V4.1 shapes.
func v41RouterConfigFromConfig(c Config) (v41RouterConfig, error) {
	if c.NumExperts != V41RouterExperts {
		return v41RouterConfig{}, &v4RouteError{Field: "v41_experts", Reason: fmt.Sprintf("want %d, got %d", V41RouterExperts, c.NumExperts)}
	}
	if c.NumExpertsPerTok != V41RouterTopK {
		return v41RouterConfig{}, &v4RouteError{Field: "v41_topk", Reason: fmt.Sprintf("want %d, got %d", V41RouterTopK, c.NumExpertsPerTok)}
	}
	if c.NSharedExperts != V41RouterSharedCount {
		return v41RouterConfig{}, &v4RouteError{Field: "v41_shared", Reason: fmt.Sprintf("want %d, got %d", V41RouterSharedCount, c.NSharedExperts)}
	}
	if c.RoutedScalingFactor != float64(V41RouterRouteScale) {
		return v41RouterConfig{}, &v4RouteError{Field: "v41_route_scale", Reason: fmt.Sprintf("want %g, got %g", float64(V41RouterRouteScale), c.RoutedScalingFactor)}
	}
	return v41DefaultRouterConfig(), nil
}

// v41SqrtSoftplus is the overflow-safe sqrt(softplus(z)) scoring function.
func v41SqrtSoftplus(z float32) float32 {
	zf := float64(z)
	return float32(math.Sqrt(math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf)))))
}

func (cfg v41RouterConfig) validate() error {
	if cfg.Experts != V41RouterExperts {
		return &v4RouteError{Field: "v41_experts", Reason: fmt.Sprintf("want %d, got %d", V41RouterExperts, cfg.Experts)}
	}
	if cfg.TopK != V41RouterTopK {
		return &v4RouteError{Field: "v41_topk", Reason: fmt.Sprintf("want %d, got %d", V41RouterTopK, cfg.TopK)}
	}
	if cfg.SharedCount < 1 {
		return &v4RouteError{Field: "v41_shared", Reason: fmt.Sprintf("want >=1, got %d", cfg.SharedCount)}
	}
	if !finite32(cfg.RouteScale) || cfg.RouteScale <= 0 {
		return &v4RouteError{Field: "v41_route_scale", Reason: fmt.Sprintf("must be finite and positive, got %g", cfg.RouteScale)}
	}
	return nil
}

// v41Route performs the V4.1 route for one token. Selection uses the noaux_tc
// score-plus-bias choice, weights use the unbiased score.
func v41Route(logits, correctionBias []float32, cfg v41RouterConfig) ([]routePick, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if len(logits) != cfg.Experts {
		return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("width %d, want %d", len(logits), cfg.Experts)}
	}
	if len(correctionBias) != 0 && len(correctionBias) != cfg.Experts {
		return nil, &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("width %d does not match experts width %d", len(correctionBias), cfg.Experts)}
	}
	if cfg.TopK > cfg.Experts {
		return nil, &v4RouteError{Field: "v41_topk", Reason: fmt.Sprintf("%d exceeds experts %d", cfg.TopK, cfg.Experts)}
	}

	raw := make([]float32, cfg.Experts)
	choice := make([]float32, cfg.Experts)
	for i, z := range logits {
		if !finite32(z) {
			return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
		}
		score := v41SqrtSoftplus(z)
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

	indices := make([]int, cfg.Experts)
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(i, j int) bool {
		return choice[indices[i]] > choice[indices[j]]
	})

	picks := make([]routePick, cfg.TopK)
	var sum float32
	for i := 0; i < cfg.TopK; i++ {
		expert := indices[i]
		picks[i] = routePick{expert: expert, weight: raw[expert]}
		sum += raw[expert]
	}
	if !finite32(sum) || sum <= 0 {
		return nil, &v4RouteError{Field: "normalization", Reason: fmt.Sprintf("selected score sum must be finite and positive, got %g", sum)}
	}
	for i := range picks {
		picks[i].weight = picks[i].weight / sum * cfg.RouteScale
		if !finite32(picks[i].weight) {
			return nil, &v4RouteError{Field: "weight", Reason: fmt.Sprintf("non-finite result for expert %d", picks[i].expert)}
		}
	}
	return picks, nil
}

// v41SharedExpertAdd adds the always-on shared-expert output to the routed
// accumulator on every token and returns the combined vector.
func v41SharedExpertAdd(routed, shared []float32, cfg v41RouterConfig) ([]float32, error) {
	if cfg.SharedCount < 1 {
		return nil, &v4RouteError{Field: "v41_shared", Reason: fmt.Sprintf("want >=1, got %d", cfg.SharedCount)}
	}
	if len(routed) == 0 || len(shared) == 0 {
		return nil, &v4RouteError{Field: "shared_add", Reason: "empty operand"}
	}
	if len(routed) != len(shared) {
		return nil, &v4RouteError{Field: "shared_add", Reason: fmt.Sprintf("width %d does not match %d", len(routed), len(shared))}
	}
	for i := range routed {
		if !finite32(routed[i]) {
			return nil, &v4RouteError{Field: "routed", Reason: fmt.Sprintf("non-finite value at %d", i)}
		}
		if !finite32(shared[i]) {
			return nil, &v4RouteError{Field: "shared", Reason: fmt.Sprintf("non-finite value at %d", i)}
		}
	}
	out := make([]float32, len(routed))
	for i := range routed {
		out[i] = routed[i] + shared[i]
		if !finite32(out[i]) {
			return nil, &v4RouteError{Field: "shared_add", Reason: fmt.Sprintf("non-finite result at %d", i)}
		}
	}
	return out, nil
}
