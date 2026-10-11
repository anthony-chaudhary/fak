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
	cos, sin := v41InverseOutputTable(cfg, layer, pos)
	for h := 0; h < heads; h++ {
		applyRopeTailInterleaved(out[h*headDim:(h+1)*headDim], cos, sin, cfg.QKRopeHeadDim)
	}
}

// v41InverseOutputTable negates the sine of the layer/position query table.
// Negation is exact, so host and device rotate with identical operands.
func v41InverseOutputTable(cfg Config, layer, pos int) (cos, conjSin []float32) {
	cos, conjSin = v41RopeTableForLayer(cfg, layer, pos)
	for j := range conjSin {
		conjSin[j] = -conjSin[j]
	}
	return cos, conjSin
}

// v41InverseAttentionOutputDevice runs the selected inverse rotation and owns
// its typed failure. The input row is never written.
func v41InverseAttentionOutputDevice(cfg Config, layer, pos int, out []float32, heads, headDim int, rotate v41InverseOutputRoPEFunc) ([]float32, error) {
	cos, conjSin := v41InverseOutputTable(cfg, layer, pos)
	rotated, err := rotate(layer, out, cos, conjSin, heads, headDim, cfg.QKRopeHeadDim)
	if err == nil && len(rotated) != len(out) {
		err = errV41TailRoPEResult
	}
	if err != nil {
		return nil, v41RoPEPublicationErr(layer, err)
	}
	return rotated, nil
}

// v41AttentionOutputForProjection stages sparse_attn's BF16 output before the
// F32 inverse complex multiply and BF16 copyback (model.py780-781). Returning
// a fresh full row avoids writing into a borrowed attention callback result.
// The existing inverse helper remains the reduced/F32 arithmetic seam.
func v41AttentionOutputForProjection(cfg Config, layer, pos int, out []float32, heads, headDim int) ([]float32, error) {
	return v41RotatedOutputForProjection(cfg, layer, pos, out, heads, headDim, nil)
}

// v41RotatedOutputForProjection dispatches the inverse rotation through the
// session's device seam when ops binds one. The BF16 staging and copyback stay
// identical; a selected failure propagates without a host retry.
func v41RotatedOutputForProjection(cfg Config, layer, pos int, out []float32, heads, headDim int, ops *v41DeviceGatherOps) ([]float32, error) {
	rotate := ops.inverseOutputFunc()
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil, err
	}
	if !full {
		if rotate == nil {
			v41InverseAttentionOutputInPlace(cfg, layer, pos, out, heads, headDim)
			return out, nil
		}
		rotated, err := v41InverseAttentionOutputDevice(cfg, layer, pos, out, heads, headDim, rotate)
		if err != nil {
			return nil, err
		}
		copy(out, rotated)
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
	if rotate == nil {
		v41InverseAttentionOutputInPlace(cfg, layer, pos, staged, heads, headDim)
	} else if staged, err = v41InverseAttentionOutputDevice(cfg, layer, pos, staged, heads, headDim, rotate); err != nil {
		return nil, err
	}
	return v41RoPEBF16Copy(layer, staged)
}
