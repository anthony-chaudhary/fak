package model

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

const (
	V41RouterExperts     = 384
	V41RouterTopK        = 6
	V41RouterSharedCount = 1
	V41RouterMoEWidth    = 2304
	V41RouterRouteScale  = 1.5
)

// V41RoutedTopKEnv opts a native V4.1 process into a reduced routed-expert
// count ("faster mode"): the router keeps its top N of the published top-6 and
// renormalises the route weights over those N. Unset keeps the published 6.
// Valid values are 1..V41RouterTopK or "auto". Routed expert bytes per token scale with
// N, so N=5 cuts them by 1/6. The rule is adapted (no code copied) from Strata
// src/dsv41/ds4_experts_rule.hpp:30-44 @46c152a1 (MIT), which reports +4% to
// +13% decode at a KL cost of ~0.0015 for 5 vs 6 (upstream-reported, not a
// fak witness). The default stays 6 until a fak quality gate passes.
const V41RoutedTopKEnv = "FAK_V41_ROUTED_TOPK"

// ErrV41RoutedTopK classifies a refused routed top-k override.
var ErrV41RoutedTopK = errors.New("model: V4.1 routed top-k override refused")

// V41RoutedTopKAuto is the opt-in auto value of V41RoutedTopKEnv. It resolves
// through V41AutoRoutedTopK from the registered device probe; with no probe it
// keeps the published 6.
const V41RoutedTopKAuto = "auto"

// v41RoutedTopKAutoVRAMCeiling is Strata's small-VRAM threshold (16 GiB).
const v41RoutedTopKAutoVRAMCeiling int64 = 16 << 30

// V41AutoRoutedTopK is the auto rule adapted from Strata ds4_experts_rule.hpp
// @46c152a1 (MIT, rule only): 5 when the GPU shares host RAM or its total VRAM
// is known and at most 16 GiB, otherwise the published 6.
func V41AutoRoutedTopK(totalVRAMBytes int64, known, sharesHostRAM bool) int {
	if sharesHostRAM || (known && totalVRAMBytes > 0 && totalVRAMBytes <= v41RoutedTopKAutoVRAMCeiling) {
		return V41RouterTopK - 1
	}
	return V41RouterTopK
}

// v41RoutedTopKDeviceProbe reports the serving device for the auto rule. It is
// nil until a runtime registers one, so auto resolves to 6 by default.
var v41RoutedTopKDeviceProbe func() (totalVRAMBytes int64, known, sharesHostRAM bool)

// v41RoutedTopK parses a routed top-k override; empty means the published 6.
func v41RoutedTopK(raw string) (int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return V41RouterTopK, nil
	}
	if strings.EqualFold(raw, V41RoutedTopKAuto) {
		if v41RoutedTopKDeviceProbe == nil {
			return V41RouterTopK, nil
		}
		return V41AutoRoutedTopK(v41RoutedTopKDeviceProbe()), nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("%w: %s=%q is not an integer", ErrV41RoutedTopK, V41RoutedTopKEnv, raw)
	}
	if n < 1 || n > V41RouterTopK {
		return 0, fmt.Errorf("%w: %s=%d outside 1..%d", ErrV41RoutedTopK, V41RoutedTopKEnv, n, V41RouterTopK)
	}
	return n, nil
}

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
// An explicit V41RoutedTopKEnv override then narrows the routed count.
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
	topK, err := v41RoutedTopK(os.Getenv(V41RoutedTopKEnv))
	if err != nil {
		return v41RouterConfig{}, err
	}
	cfg := v41DefaultRouterConfig()
	cfg.TopK = topK
	return cfg, nil
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
	if cfg.TopK < 1 || cfg.TopK > V41RouterTopK {
		return &v4RouteError{Field: "v41_topk", Reason: fmt.Sprintf("want 1..%d, got %d", V41RouterTopK, cfg.TopK)}
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
	return v41RouteWithNormalization(logits, correctionBias, cfg, false)
}

// v41RouteForGeometry preserves the reduced router contract. The official full
// Gate divides by sum + 1e-20, including a valid all-zero selected-score row.
// This changes only that F32 normalization sequence, not score or top-k math.
// Reference: DeepSeek-V4.1-Flash inference/model.py Gate.forward, revision
// dba1be0a40aa45a94ad051997016db3960a90277, lines 824-826 (MIT).
func v41RouteForGeometry(logits, correctionBias []float32, cfg v41RouterConfig, full bool) ([]routePick, error) {
	return v41RouteWithNormalization(logits, correctionBias, cfg, full)
}

func v41RouteWithNormalization(logits, correctionBias []float32, cfg v41RouterConfig, full bool) ([]routePick, error) {
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

	// Preserve the explicitly selected generic kernel. The ordinary host path
	// needs only six candidates, not expert-wide score arrays and a full sort.
	partial := v4BitonicTopKEnabled()
	var raw, choice []float32
	if partial {
		raw = make([]float32, cfg.Experts)
		choice = make([]float32, cfg.Experts)
	}
	var selected [V41RouterTopK]routePick
	var selectedChoice [V41RouterTopK]float32
	count := 0
	for i, z := range logits {
		if !finite32(z) {
			return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
		}
		var score float32
		if full {
			// Gate.forward publishes F.softplus(scores) as F32 before sqrt.
			// Keep the stable host formula, but do not retain tiny F64 values
			// across that tensor boundary. This is not GPU exp/log parity.
			zf := float64(z)
			softplus := float32(math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf))))
			score = float32(math.Sqrt(float64(softplus)))
		} else {
			score = v41SqrtSoftplus(z)
		}
		if !finite32(score) {
			return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite score at expert %d", i)}
		}
		selection := score
		if len(correctionBias) != 0 {
			bias := correctionBias[i]
			if !finite32(bias) {
				return nil, &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
			}
			selection += bias
			if !finite32(selection) {
				return nil, &v4RouteError{Field: "selection_score", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
			}
		}
		if partial {
			raw[i], choice[i] = score, selection
			continue
		}
		// Experts arrive in ascending ID order. Strict comparison puts an
		// equal score after earlier IDs, exactly like the stable full sort.
		at := count
		for at > 0 && selection > selectedChoice[at-1] {
			at--
		}
		if at < cfg.TopK {
			end := min(count, cfg.TopK-1)
			copy(selected[at+1:end+1], selected[at:end])
			copy(selectedChoice[at+1:end+1], selectedChoice[at:end])
			selected[at] = routePick{expert: i, weight: score}
			selectedChoice[at] = selection
			if count < cfg.TopK {
				count++
			}
		}
	}

	picks := make([]routePick, cfg.TopK)
	if partial {
		indices := v4TopKIndices(choice, cfg.TopK)
		for i := range picks {
			expert := indices[i]
			picks[i] = routePick{expert: expert, weight: raw[expert]}
		}
	} else {
		copy(picks, selected[:])
	}
	var sum float32
	for _, pick := range picks {
		sum += pick.weight
	}
	if !finite32(sum) || sum < 0 || (!full && sum == 0) {
		return nil, &v4RouteError{Field: "normalization", Reason: fmt.Sprintf("selected score sum must be finite and positive, got %g", sum)}
	}
	denominator := sum
	if full {
		denominator = float32(sum + float32(1e-20))
	}
	for i := range picks {
		if full {
			picks[i].weight = float32(picks[i].weight / denominator)
			picks[i].weight = float32(picks[i].weight * cfg.RouteScale)
		} else {
			picks[i].weight = picks[i].weight / sum * cfg.RouteScale
		}
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
	copy(out, routed)
	if err := ffn.AddScaled(out, shared, 1); err != nil {
		return nil, &v4RouteError{Field: "shared_add", Reason: err.Error()}
	}
	for i := range out {
		if !finite32(out[i]) {
			return nil, &v4RouteError{Field: "shared_add", Reason: fmt.Sprintf("non-finite result at %d", i)}
		}
	}
	return out, nil
}
