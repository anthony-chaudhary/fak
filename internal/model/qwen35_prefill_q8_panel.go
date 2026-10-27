package model

// Names are the complete synchronous consumer set for this activation. No
// descriptor or activation is retained by the target-aware quantizer.
type hybridQ4KQuantize func(x []float32, rows, width int, names ...string) *q8Panel

// qwen35PrefillQ8PanelUnused admits only panels that projectRotated cannot read:
// resident Q4/K/Q2 consumers use f32, and a declared Prism Q8 consumer builds its
// own rotated Q8 panel. Any undeclared Q8/on-demand target keeps the old eager
// panel. Classification follows the exact resident-format dispatch precedence;
// a Q8 shadow must not override a resident compact-format target.
func (s *Session) qwen35PrefillQ8PanelUnused(x []float32, rows, width int, names []string) bool {
	if s == nil || s.M == nil || s.Backend != nil || s.MetalQ4K || s.PhaseProfiler != nil ||
		s.tapActive != nil || s.M.attnObs != nil || s.M.lora != nil || q8PanelBuildHook != nil || qprofOn ||
		qgemmMode == qgemmModeLegacy || qgemmMode == qgemmModeAccel || rows <= 1 || width <= 0 ||
		width%qBlk != 0 || len(x)%width != 0 || len(x)/width != rows || len(names) == 0 {
		return false
	}
	m := s.M
	for _, name := range names {
		if qt := m.q4kw[name]; qt != nil {
			if qt.in != width {
				return false
			}
			continue
		}
		if qt := m.kqw[name]; qt != nil {
			if qt.in != width {
				return false
			}
			continue
		}
		if qt := m.q2w[name]; qt != nil {
			if qt.in != width {
				return false
			}
			continue
		}
		if m.prism != nil && m.prism.weightWidth[name] != 0 {
			if m.prism.weightWidth[name] != width {
				return false
			}
			continue
		}
		return false
	}
	return true
}
