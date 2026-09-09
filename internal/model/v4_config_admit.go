package model

import (
	"errors"
	"fmt"
	"math"
)

var ErrV4ConfigAdmission = errors.New("model: DeepSeek V4 config is not admitted")

// IsDeepSeekV4 reports only the architecture identity. Call AdmitDeepSeekV4Config
// before constructing a runtime; identity alone never authorizes a fallback path.
func (c Config) IsDeepSeekV4() bool {
	return c.ModelType == "deepseek_v4"
}

func AdmitDeepSeekV4Config(c Config) error {
	if !c.IsDeepSeekV4() {
		return fmt.Errorf("%w: model_type=%q", ErrV4ConfigAdmission, c.ModelType)
	}
	supportedProfile := (c.NumLayers == 61 &&
		c.HiddenSize == 7168 &&
		c.NumExperts == 384 &&
		c.MoEIntermediateSize == 3072 &&
		c.RoutedScalingFactor == 2.5) ||
		(c.NumLayers == 43 &&
			c.HiddenSize == 4096 &&
			c.NumExperts == 256 &&
			c.MoEIntermediateSize == 2048 &&
			c.RoutedScalingFactor == 1.5)
	checks := []struct {
		name string
		ok   bool
		got  any
	}{
		{"profile", supportedProfile, fmt.Sprintf("layers=%d hidden=%d experts=%d moe_intermediate=%d route_scale=%g", c.NumLayers, c.HiddenSize, c.NumExperts, c.MoEIntermediateSize, c.RoutedScalingFactor)},
		{"num_experts_per_tok", c.NumExpertsPerTok == 6, c.NumExpertsPerTok},
		{"n_shared_experts", c.NSharedExperts == 1, c.NSharedExperts},
		{"expert_dtype", c.ExpertDtype == "fp4", c.ExpertDtype},
		{"norm_topk_prob", c.NormTopKProb, c.NormTopKProb},
		{"scoring_func", c.ScoringFunc == "sqrtsoftplus", c.ScoringFunc},
		{"topk_method", c.TopKMethod == "noaux_tc", c.TopKMethod},
		{"swiglu_limit", c.SwigluLimit >= 0 && !math.IsNaN(c.SwigluLimit) && !math.IsInf(c.SwigluLimit, 0), c.SwigluLimit},
	}
	for _, check := range checks {
		if !check.ok {
			return fmt.Errorf("%w: %s=%v", ErrV4ConfigAdmission, check.name, check.got)
		}
	}
	return nil
}
