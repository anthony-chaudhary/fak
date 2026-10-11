package model

// The graph owner stages BF16 collapse/output boundaries around the device's
// F32 learned RMSNorm. A selected callback cannot decline or retry on the host.
type v41AttentionInputNormFunc func(layer int, input []float32) ([]float32, error)

func (s *Session) v41AttentionInputNormFunc() v41AttentionInputNormFunc {
	normalize := s.v41RMSNormFunc()
	if normalize == nil {
		return nil
	}
	full, err := v41ForwardGeometry(s.M.Cfg)
	if err != nil || !full {
		return nil
	}
	return func(layer int, input []float32) ([]float32, error) {
		return normalize(layerName(layer, "attn_norm.weight"), input, s.M.Cfg.HiddenSize, "v41-attention-input-norm", layer)
	}
}

func (m *Model) v41AttentionInputNormWithDevice(layer int, collapsed []float32, eps float32, normalize v41AttentionInputNormFunc) ([]float32, error) {
	return m.v41AttentionInputNormWithPublications(layer, collapsed, eps, normalize, nil)
}

// Qualified publications move both BF16 boundaries onto the selected backend;
// the RMSNorm in between is unchanged. Nil keeps the legacy helper's exact path.
func (m *Model) v41AttentionInputNormWithPublications(layer int, collapsed []float32, eps float32, normalize v41AttentionInputNormFunc, publish v41BF16BoundaryFunc) ([]float32, error) {
	if normalize == nil {
		return m.v41AttentionInputNorm(layer, collapsed, eps)
	}
	const leaf = "attn_norm.weight"
	if len(collapsed) != m.Cfg.HiddenSize || len(collapsed) == 0 || !finite32(eps) || eps <= 0 {
		return nil, v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
	}
	if publish != nil && eps != float32(m.Cfg.RMSNormEps) {
		return nil, v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
	}
	var input []float32
	var err error
	if publish == nil {
		input, err = v41LatentNormBF16Copy(layer, leaf, "collapse", collapsed)
	} else {
		input, err = v41BF16BoundaryCopy(layer, leaf+" input", collapsed, publish)
	}
	if err != nil {
		return nil, err
	}
	values, err := normalize(layer, input)
	if err != nil {
		return nil, v41ProjectionOperationErr(layer, leaf, err)
	}
	if len(values) != len(collapsed) {
		return nil, v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
	}
	if publish == nil {
		return v41LatentNormBF16Copy(layer, leaf, "normalization", values)
	}
	return v41BF16BoundaryCopy(layer, leaf+" output", values, publish)
}
