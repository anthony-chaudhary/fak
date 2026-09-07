package model

import (
	"fmt"
	"strings"
)

// ValidateGLM5NextManifest verifies that a materialized tensor manifest conforms to
// the pinned 45-layer GLM-5.3-Flash contract:
// - Embeddings and final norm (with canonical 'model.' or upstream 'model.language_model.' prefix)
// - Exactly 45 layers
// - For layers 0..44:
//   - Input norm
//   - Cadence: layer%4 != 3 -> KDA linear mixer weights (linear_attn.in_proj OR q_proj + k_conv1d)
//   - Cadence: layer%4 == 3 -> DSA sparse mixer weights (q_proj)
//   - Cadence: layer < 3   -> Dense MLP weights (gate_proj)
//   - Cadence: layer >= 3  -> MoE weights (router OR experts.0.down_proj)
func ValidateGLM5NextManifest(manifest map[string]tensorMeta) error {
	if manifest == nil {
		return fmt.Errorf("model: nil GLM5Next manifest")
	}

	prefix := "model."
	for k := range manifest {
		if strings.HasPrefix(k, "model.language_model.") {
			prefix = "model.language_model."
			break
		}
	}

	has := func(key string) bool {
		if _, ok := manifest[key]; ok {
			return true
		}
		if strings.HasPrefix(key, "model.language_model.") {
			alt := "model." + strings.TrimPrefix(key, "model.language_model.")
			if _, ok := manifest[alt]; ok {
				return true
			}
		} else if strings.HasPrefix(key, "model.") {
			alt := "model.language_model." + strings.TrimPrefix(key, "model.")
			if _, ok := manifest[alt]; ok {
				return true
			}
		}
		return false
	}

	requiredGlobals := []string{
		"embed_tokens.weight",
		"norm.weight",
	}
	for _, req := range requiredGlobals {
		key := prefix + req
		if !has(key) {
			return fmt.Errorf("model: GLM5Next manifest missing required tensor %s", key)
		}
	}

	for l := 0; l < 45; l++ {
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, l)
		normKey := layerPrefix + ".input_layernorm.weight"
		if !has(normKey) {
			return fmt.Errorf("model: layer %d missing %s", l, normKey)
		}

		isKDA := l%4 != 3
		if isKDA {
			linearKey := layerPrefix + ".self_attn.linear_attn.in_proj.weight"
			qKey := layerPrefix + ".self_attn.q_proj.weight"
			kConvKey := layerPrefix + ".self_attn.k_conv1d.weight"

			hasLinear := has(linearKey)
			hasUpstreamKDA := has(qKey) && has(kConvKey)
			if !hasLinear && !hasUpstreamKDA {
				return fmt.Errorf("model: KDA layer %d missing %s", l, linearKey)
			}
		} else {
			dsaKey := layerPrefix + ".self_attn.q_proj.weight"
			if !has(dsaKey) {
				return fmt.Errorf("model: DSA layer %d missing %s", l, dsaKey)
			}
		}

		isDense := l < 3
		if isDense {
			mlpKey := layerPrefix + ".mlp.gate_proj.weight"
			if !has(mlpKey) {
				return fmt.Errorf("model: dense MLP layer %d missing %s", l, mlpKey)
			}
		} else {
			routerKey := layerPrefix + ".mlp.router.weight"
			expertKey := layerPrefix + ".mlp.experts.0.down_proj.weight"

			hasRouter := has(routerKey)
			hasExpert := has(expertKey)
			if !hasRouter && !hasExpert {
				return fmt.Errorf("model: sparse MoE layer %d missing %s", l, routerKey)
			}
		}
	}

	return nil
}
