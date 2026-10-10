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

// v41AttentionOutputForProjection stages sparse_attn's BF16 output before the
// F32 inverse complex multiply and BF16 copyback (model.py780-781). Returning
// a fresh full row avoids writing into a borrowed attention callback result.
// The existing inverse helper remains the reduced/F32 arithmetic seam.
func v41AttentionOutputForProjection(cfg Config, layer, pos int, out []float32, heads, headDim int) ([]float32, error) {
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil, err
	}
	if !full {
		v41InverseAttentionOutputInPlace(cfg, layer, pos, out, heads, headDim)
		return out, nil
	}
	width, ok := checkedMulInt(heads, headDim)
	if !ok || heads <= 0 || headDim <= 0 || len(out) != width || cfg.QKRopeHeadDim <= 0 || cfg.QKRopeHeadDim%2 != 0 || cfg.QKRopeHeadDim > headDim {
		return nil, v41RoPEPublicationErr(layer, errV41TailRoPEResult)
	}
	staged, err := v41RoPEBF16Copy(layer, out)
	if err != nil {
		return nil, err
	}
	v41InverseAttentionOutputInPlace(cfg, layer, pos, staged, heads, headDim)
	return v41RoPEBF16Copy(layer, staged)
}
