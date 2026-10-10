package model

// The full FFN input collapses the attention-post streams with attn_pre.
// The owner stages BF16 input/output around this unchanged F32 learned RMSNorm
// callback. The reduced graph retains its legacy hidden input. A callback has no decline
// result: failures cannot fall through to host normalization or replay.
type v41FFNNormFunc func(layer int, input []float32) ([]float32, error)

func (s *Session) v41FFNNormFunc() v41FFNNormFunc {
	normalize := s.v41RMSNormFunc()
	if normalize == nil {
		return nil
	}
	full, err := v41ForwardGeometry(s.M.Cfg)
	if err != nil || !full {
		return nil
	}
	return func(layer int, input []float32) ([]float32, error) {
		return normalize(layerName(layer, "ffn_norm.weight"), input, s.M.Cfg.HiddenSize, "v41-ffn-norm", layer)
	}
}

func (m *Model) v41FFNNorm(layer int, input []float32, eps float32, normalize v41FFNNormFunc) ([]float32, error) {
	const leaf = "ffn_norm.weight"
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil {
		return nil, err
	}
	staged := input
	if full {
		if len(input) != m.Cfg.HiddenSize {
			return nil, v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
		}
		staged, err = v41LatentNormBF16Copy(layer, leaf, "collapse", input)
		if err != nil {
			return nil, err
		}
	}
	var values []float32
	if normalize == nil {
		if full {
			// Plain learned RMSNorm, without unrelated generic norm flags.
			values, err = m.v41AttentionInputNormWithLeaf(layer, leaf, staged, eps)
			return values, err
		}
		return rmsnormCfg(input, m.tensor(layerName(layer, leaf)), eps, m.Cfg), nil
	}
	values, err = normalize(layer, staged)
	if err == nil && len(values) != len(input) {
		err = errV41ProjectionResult
	}
	if err == nil {
		for _, v := range values {
			if !finite32(v) {
				err = errV41ProjectionResult
				break
			}
		}
	}
	if err != nil {
		return nil, v41ProjectionOperationErr(layer, leaf, err)
	}
	if full {
		return v41LatentNormBF16Copy(layer, leaf, "normalization", values)
	}
	return values, nil
}
