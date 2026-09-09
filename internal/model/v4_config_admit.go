package model

import (
	"errors"
	"fmt"
	"math"
)

var ErrV4ConfigAdmission = errors.New("model: DeepSeek V4 config is not admitted")

var deepSeekV4FlashCompressRatios = [...]int{
	0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
	4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
	4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
}

type v4ConfigCheck struct {
	name string
	ok   bool
	got  any
}

// IsDeepSeekV4 reports only the architecture identity. Call AdmitDeepSeekV4Config
// before constructing a runtime; identity alone never authorizes a fallback path.
func (c Config) IsDeepSeekV4() bool {
	return c.ModelType == "deepseek_v4"
}

func AdmitDeepSeekV4Config(c Config) error {
	if !c.IsDeepSeekV4() {
		return fmt.Errorf("%w: model_type=%q", ErrV4ConfigAdmission, c.ModelType)
	}
	proProfile := c.NumLayers == 61 &&
		c.HiddenSize == 7168 &&
		c.NumExperts == 384 &&
		c.MoEIntermediateSize == 3072 &&
		c.RoutedScalingFactor == 2.5
	flashProfile := isDeepSeekV4FlashProfile(c)
	supportedProfile := proProfile || flashProfile
	checks := []v4ConfigCheck{
		{"profile", supportedProfile, fmt.Sprintf("layers=%d hidden=%d experts=%d moe_intermediate=%d route_scale=%g", c.NumLayers, c.HiddenSize, c.NumExperts, c.MoEIntermediateSize, c.RoutedScalingFactor)},
		{"num_experts_per_tok", c.NumExpertsPerTok == 6, c.NumExpertsPerTok},
		{"n_shared_experts", c.NSharedExperts == 1, c.NSharedExperts},
		{"expert_dtype", c.ExpertDtype == "fp4", c.ExpertDtype},
		{"norm_topk_prob", c.NormTopKProb, c.NormTopKProb},
		{"scoring_func", c.ScoringFunc == "sqrtsoftplus", c.ScoringFunc},
		{"topk_method", c.TopKMethod == "noaux_tc", c.TopKMethod},
		{"swiglu_limit", c.SwigluLimit >= 0 && !math.IsNaN(c.SwigluLimit) && !math.IsInf(c.SwigluLimit, 0), c.SwigluLimit},
	}
	if flashProfile {
		checks = append(checks,
			v4ConfigCheck{"compress_ratios", isDeepSeekV4FlashCompressSchedule(c.CompressRatios), c.CompressRatios},
			v4ConfigCheck{"hc_mult", c.HCMult == 4, c.HCMult},
			v4ConfigCheck{"hc_eps", c.HCEps == 1e-6, c.HCEps},
			v4ConfigCheck{"hc_sinkhorn_iters", c.HCSinkhornIters == 20, c.HCSinkhornIters},
			v4ConfigCheck{"o_groups", c.OGroups == 8, c.OGroups},
			v4ConfigCheck{"o_lora_rank", c.OLoraRank == 1024, c.OLoraRank},
		)
	}
	for _, check := range checks {
		if !check.ok {
			return fmt.Errorf("%w: %s=%v", ErrV4ConfigAdmission, check.name, check.got)
		}
	}
	return nil
}

func isDeepSeekV4FlashProfile(c Config) bool {
	return c.IsDeepSeekV4() &&
		c.NumLayers == 43 &&
		c.HiddenSize == 4096 &&
		c.NumExperts == 256 &&
		c.MoEIntermediateSize == 2048 &&
		c.RoutedScalingFactor == 1.5
}

func isDeepSeekV4FlashCompressSchedule(got []int) bool {
	if len(got) != len(deepSeekV4FlashCompressRatios) {
		return false
	}
	for i, want := range deepSeekV4FlashCompressRatios {
		if got[i] != want {
			return false
		}
	}
	return true
}
