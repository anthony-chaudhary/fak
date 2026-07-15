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
	checks := []struct {
		name string
		ok   bool
		got  any
	}{
		{"num_hidden_layers", c.NumLayers == 61, c.NumLayers},
		{"hidden_size", c.HiddenSize == 7168, c.HiddenSize},
		{"n_routed_experts", c.NumExperts == 384, c.NumExperts},
		{"num_experts_per_tok", c.NumExpertsPerTok == 6, c.NumExpertsPerTok},
		{"moe_intermediate_size", c.MoEIntermediateSize == 3072, c.MoEIntermediateSize},
		{"n_shared_experts", c.NSharedExperts == 1, c.NSharedExperts},
		{"expert_dtype", c.ExpertDtype == "fp4", c.ExpertDtype},
		{"norm_topk_prob", c.NormTopKProb, c.NormTopKProb},
		{"routed_scaling_factor", c.RoutedScalingFactor == 2.5, c.RoutedScalingFactor},
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
