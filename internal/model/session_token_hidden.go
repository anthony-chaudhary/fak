package model

// tokenHidden runs one position (absolute index pos, embedding-looked-up hidden x)
// through all layers against the cache, appending this position's K/V, and returns
// the post-final-norm hidden vector (NOT yet projected to logits). Immediately before
// finalNorm it captures the exact target residual required by Qwen3.8 MTP; the head
// is applied by the caller.
func (s *Session) tokenHidden(id, pos int) (out []float32) {
	if s.Quant {
		return s.tokenHiddenQ(id, pos)
	}
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	tap := s.activeTap()
	if tap != nil && !tap.wants(pos) {
		tap = nil
	}
	prevTap := s.tapActive
	s.tapActive = tap
	defer func() { s.tapActive = prevTap }()

	var x []float32
	if m.Q2KEmbedding != nil {
		x = make([]float32, H)
		if err := m.Q2KEmbedding.GatherRow(id, x, cfg.embedScale()); err != nil {
			panic(err)
		}
	} else {
		embed := m.embedRows()
		x = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x, cfg) // Gemma sqrt(hidden); no-op for Llama
	}

	for l := 0; l < cfg.NumLayers; l++ {
		cos, sin := ropeRowForLayer(cfg, l, pos)
		x = s.blockStep(l, pos, x, cos, sin, f32Kernel{m})
		if capture := s.qwen35TargetLayerCapture; capture != nil {
			capture.record(id, pos, l, x)
		}
	}
	s.Cache.appendPosition(pos, id)
	if tap != nil {
		tap.writeMeta(cfg, H, pos)
	}
	s.rememberTargetHidden(pos, id, x)
	out = m.finalNorm(x)
	return out
}
