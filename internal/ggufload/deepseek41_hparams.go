package ggufload

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// ErrDeepSeek41HParam is the typed refusal for a deepseek41 hyperparameter key
// that is present but unusable. An absent key leaves the config axis at its
// zero value; on the published decoder schedule model.NewDeepSeekV41GGUFConfig
// then refuses it with model.ErrV41ConfigAdmission.
var ErrDeepSeek41HParam = errors.New("gguf: deepseek41 hyperparameter malformed")

const (
	ds41KeyRopeScalingType     = "rope.scaling.type"
	ds41KeyRopeScalingFactor   = "rope.scaling.factor"
	ds41KeyRopeScalingOrigCtx  = "rope.scaling.original_context_length"
	ds41KeyRopeYarnBetaFast    = "rope.scaling.yarn_beta_fast"
	ds41KeyRopeYarnBetaSlow    = "rope.scaling.yarn_beta_slow"
	ds41KeySwigluClampExpert   = "swiglu_clamp_exp"
	ds41KeySwigluClampShexpert = "swiglu_clamp_shexp"
	ds41KeyExpertGatingFunc    = "expert_gating_func"
	ds41KeyHashLayerCount      = "hash_layer_count"

	// ds41GatingSqrtSoftplus is llama.cpp's gating enum for sqrt(softplus), the only
	// router score the native V4.1 forward implements (v41SqrtSoftplus).
	ds41GatingSqrtSoftplus = 4
)

func ds41HParamErr(key, format string, args ...any) error {
	return fmt.Errorf("%w: %s %s", ErrDeepSeek41HParam, key, fmt.Sprintf(format, args...))
}

func finitePositive(v float64) bool { return v > 0 && !math.IsInf(v, 0) && !math.IsNaN(v) }

// applyDeepSeek41HParams maps the V4.1 GGUF keys the HF config path already
// carries (compress_rope_theta, sliding_window, rope_scaling yarn,
// swiglu_limit) so a GGUF load matches the HF-derived Config on these axes.
func applyDeepSeek41HParams(f *File, p string, cfg *model.Config) error {
	if f.hasKey(p + ds41KeyCompressRopeBase) {
		v, ok := f.Float64(p + ds41KeyCompressRopeBase)
		if !ok || !finitePositive(v) {
			return ds41HParamErr(p+ds41KeyCompressRopeBase, "must be a finite positive float, got %v", f.Metadata[p+ds41KeyCompressRopeBase].Value)
		}
		cfg.CompressRopeTheta = v
	}

	if f.hasKey(p + ds41KeySlidingWindow) {
		w, ok := f.Uint64(p + ds41KeySlidingWindow)
		if !ok || w == 0 || w > math.MaxInt32 {
			return ds41HParamErr(p+ds41KeySlidingWindow, "must be a positive integer, got %v", f.Metadata[p+ds41KeySlidingWindow].Value)
		}
		if len(cfg.Window) == 0 {
			cfg.Window = make([]int, cfg.NumLayers)
			for l := range cfg.Window {
				cfg.Window[l] = int(w)
			}
		}
	}

	if kind, ok := f.String(p + ds41KeyRopeScalingType); ok && kind != "" && kind != "none" {
		if kind != "yarn" {
			return ds41HParamErr(p+ds41KeyRopeScalingType, "is %q; only yarn is mapped for deepseek41", kind)
		}
		factor, okF := f.Float64(p + ds41KeyRopeScalingFactor)
		orig, okO := f.Uint64(p + ds41KeyRopeScalingOrigCtx)
		if !okF || !finitePositive(factor) {
			return ds41HParamErr(p+ds41KeyRopeScalingFactor, "must be a finite positive float for yarn")
		}
		if !okO || orig == 0 || orig > math.MaxInt32 {
			return ds41HParamErr(p+ds41KeyRopeScalingOrigCtx, "must be a positive integer for yarn")
		}
		rs := &model.RopeScaling{Type: "yarn", RopeType: "yarn", Factor: factor, OriginalMaxPositionEmbeddings: int(orig)}
		for _, b := range []struct {
			key string
			dst *float64
		}{{ds41KeyRopeYarnBetaFast, &rs.BetaFast}, {ds41KeyRopeYarnBetaSlow, &rs.BetaSlow}} {
			if !f.hasKey(p + b.key) {
				continue
			}
			v, ok := f.Float64(p + b.key)
			if !ok || !finitePositive(v) {
				return ds41HParamErr(p+b.key, "must be a finite positive float")
			}
			*b.dst = v
		}
		if cfg.RopeScaling == "" {
			cfg.RopeScaling = "yarn"
			cfg.RopeFactor = factor
			cfg.RopeOrigContext = int(orig)
			cfg.LongRope = rs
		}
	}

	limit, have := 0.0, false
	for _, key := range []string{ds41KeySwigluClampExpert, ds41KeySwigluClampShexpert} {
		vals, present, err := ds41FloatArray(f, p+key)
		if err != nil {
			return err
		}
		if !present {
			continue
		}
		for _, v := range vals {
			if v < 0 || math.IsInf(v, 0) || math.IsNaN(v) {
				return ds41HParamErr(p+key, "carries a non-finite or negative clamp %g", v)
			}
			if !have {
				limit, have = v, true
			} else if v != limit {
				return ds41HParamErr(p+key, "carries per-layer clamp %g != %g; Config has one SwigluLimit", v, limit)
			}
		}
	}
	if have && cfg.SwigluLimit == 0 {
		cfg.SwigluLimit = limit
	}

	if f.hasKey(p + ds41KeyExpertWeightsNorm) {
		if v, ok := f.Bool(p + ds41KeyExpertWeightsNorm); !ok || !v {
			return ds41HParamErr(p+ds41KeyExpertWeightsNorm, "is %v; the native V4.1 router always renormalizes its top-k weights", f.Metadata[p+ds41KeyExpertWeightsNorm].Value)
		}
	}
	if f.hasKey(p + ds41KeyExpertGatingFunc) {
		g, ok := f.Uint64(p + ds41KeyExpertGatingFunc)
		if !ok || g != ds41GatingSqrtSoftplus {
			return ds41HParamErr(p+ds41KeyExpertGatingFunc, "is %v; the native V4.1 router implements only sqrtsoftplus (%d)", f.Metadata[p+ds41KeyExpertGatingFunc].Value, ds41GatingSqrtSoftplus)
		}
		cfg.ScoringFunc = "sqrtsoftplus"
	}
	if f.hasKey(p + ds41KeyHashLayerCount) {
		n, ok := f.Uint64(p + ds41KeyHashLayerCount)
		if !ok || n != 0 {
			return ds41HParamErr(p+ds41KeyHashLayerCount, "is %v; the native V4.1 forward has no token-hash routed layers", f.Metadata[p+ds41KeyHashLayerCount].Value)
		}
	}
	return nil
}

func ds41FloatArray(f *File, key string) ([]float64, bool, error) {
	if !f.hasKey(key) {
		return nil, false, nil
	}
	vals, ok := metadataArray(f, key, func(item Value) (float64, bool) {
		switch item.Type {
		case TypeFloat32:
			return float64(item.Value.(float32)), true
		case TypeFloat64:
			return item.Value.(float64), true
		default:
			return 0, false
		}
	})
	if !ok || len(vals) == 0 {
		return nil, true, ds41HParamErr(key, "must be a non-empty float array")
	}
	return vals, true, nil
}
