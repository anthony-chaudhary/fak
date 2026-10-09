package model

// The FFN input is the layer's hidden row before the combined attention/MoE
// post-mix, with the learned ffn_norm gain. A selected callback has no decline
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
	if normalize == nil {
		return rmsnormCfg(input, m.tensor(layerName(layer, leaf)), eps, m.Cfg), nil
	}
	values, err := normalize(layer, input)
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
	return values, nil
}
