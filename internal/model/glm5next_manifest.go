package model

import (
	"fmt"
)

// ValidateGLM5NextManifest verifies that a materialized tensor manifest conforms to
// the pinned 45-layer GLM-5.3-Flash contract:
// - Embeddings and final norm
// - Exactly 45 layers
// - For layers 0..44:
//   - Input norm
//   - Cadence: layer%4 != 3 -> KDA linear mixer weights
//   - Cadence: layer%4 == 3 -> DSA sparse mixer weights
//   - Cadence: layer < 3   -> Dense MLP weights
//   - Cadence: layer >= 3  -> MoE weights (shared expert + router)
func ValidateGLM5NextManifest(manifest map[string]tensorMeta) error {
	if manifest == nil {
		return fmt.Errorf("model: nil GLM5Next manifest")
	}

	requiredGlobals := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
	}
	for _, req := range requiredGlobals {
		if _, ok := manifest[req]; !ok {
			return fmt.Errorf("model: GLM5Next manifest missing required tensor %s", req)
		}
	}

	for l := 0; l < 45; l++ {
		prefix := fmt.Sprintf("model.layers.%d", l)
		normKey := prefix + ".input_layernorm.weight"
		if _, ok := manifest[normKey]; !ok {
			return fmt.Errorf("model: layer %d missing %s", l, normKey)
		}

		isKDA := l%4 != 3
		if isKDA {
			kdaKey := prefix + ".self_attn.linear_attn.in_proj.weight"
			if _, ok := manifest[kdaKey]; !ok {
				return fmt.Errorf("model: KDA layer %d missing %s", l, kdaKey)
			}
		} else {
			dsaKey := prefix + ".self_attn.q_proj.weight"
			if _, ok := manifest[dsaKey]; !ok {
				return fmt.Errorf("model: DSA layer %d missing %s", l, dsaKey)
			}
		}

		isDense := l < 3
		if isDense {
			mlpKey := prefix + ".mlp.gate_proj.weight"
			if _, ok := manifest[mlpKey]; !ok {
				return fmt.Errorf("model: dense MLP layer %d missing %s", l, mlpKey)
			}
		} else {
			moeKey := prefix + ".mlp.router.weight"
			if _, ok := manifest[moeKey]; !ok {
				return fmt.Errorf("model: sparse MoE layer %d missing %s", l, moeKey)
			}
		}
	}

	return nil
}
