package model

// ResidualHook may mutate the completed residual stream for one transformer block in
// place. The slice is model-owned and valid only for the duration of the call. Hooks must
// not retain it. This write seam is gated by Config.EnableResidualHook.
type ResidualHook func(layer int, hidden []float32)

// SetResidualHook installs or clears the model's activation-space residual hook. The hook
// is invoked only while Config.EnableResidualHook is true.
func (m *Model) SetResidualHook(hook ResidualHook) { m.Cfg.residualHook = hook }

// ResidualHookSet reports whether a hook is installed, independently of whether its config
// gate is enabled.
func (m *Model) ResidualHookSet() bool { return m.Cfg.residualHook != nil }

func invokeResidualHook(cfg Config, layer int, hidden []float32) {
	if cfg.EnableResidualHook && cfg.residualHook != nil {
		cfg.residualHook(layer, hidden)
	}
}
