package model

// qwen35PrefillNormWorkspace owns exactly one P*H normalization destination
// panel for one ordinary host-prefill invocation. It is not a Session field or
// a pool. Like the existing batch scratch contract, every element is overwritten
// before a consumer reads it; unlike grow, capacity never exceeds the exact
// invocation extent. Only the private builtin producer may reuse this storage.
type qwen35PrefillNormWorkspace struct {
	rows, width int
	normalized  []float32
}

func (s *Session) newQwen35PrefillNormWorkspace(rows int) *qwen35PrefillNormWorkspace {
	if !qwen35PrefillNormWorkspaceQualification || s == nil || s.M == nil ||
		s.Backend != nil || s.Metal || s.MetalQ4K || s.M.prism != nil {
		return nil
	}
	cfg := s.M.Cfg
	if !cfg.IsQwen35Hybrid() || cfg.NumLayers <= 0 || rows <= 0 ||
		cfg.HiddenSize <= 0 || rows > int(^uint(0)>>1)/cfg.HiddenSize {
		return nil
	}
	return &qwen35PrefillNormWorkspace{rows: rows, width: cfg.HiddenSize}
}

// panel borrows the next serialized normalization destination. The ordinary
// callers fully overwrite all rows under a joined parFor, synchronously consume
// the panel through the builtin attention/projection path, and finish the
// residual join before borrowing it again. Q8 panel quantization may remember an
// F32 input on Accelerate, but its CBLAS consumers finish synchronously before
// the next quantization/borrow. Metal/HAL/Prism and arbitrary callback producers
// never receive this owner. No returned logits, KV row, recurrent state, or
// projection output aliases it; panic/cancel follows the invocation's existing
// boundary and cannot publish or retain the owner.
func (w *qwen35PrefillNormWorkspace) panel(rows, width int) []float32 {
	if w == nil || rows != w.rows || width != w.width {
		return make([]float32, rows*width)
	}
	if w.normalized == nil {
		w.normalized = make([]float32, rows*width)
	}
	return w.normalized
}
