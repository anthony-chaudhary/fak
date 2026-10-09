package model

// v41InverseAttentionOutputInPlace removes the query's rotation from every
// attention output head before the grouped output projection. DeepSeek-V4.1's
// reference Attention.forward conjugates the same layer/absolute-position table
// used for Q; only the trailing rotary dimensions participate.
//
// Both attention implementations return owned host rows. Keep this operation
// outside their options: the selected shared-attention device contract excludes
// inverse rotation. Callers have already validated head and rotary geometry.
func v41InverseAttentionOutputInPlace(cfg Config, layer, pos int, out []float32, heads, headDim int) {
	cos, sin := v41RopeTableForLayer(cfg, layer, pos)
	for j := range sin {
		sin[j] = -sin[j]
	}
	for h := 0; h < heads; h++ {
		applyRopeTailInterleaved(out[h*headDim:(h+1)*headDim], cos, sin, cfg.QKRopeHeadDim)
	}
}
