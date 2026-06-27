//go:build !(darwin && cgo && fakmetal)

package model

// metal_decode_off.go — the default (non-fakmetal) stubs for the GPU-resident Q8 decode forward.
// The real path lives in metal_decode.go behind -tags fakmetal; here it always declines so
// tokenHiddenQ runs the proven CPU Q8 decode.

func (m *Model) metalDecodeConfig() bool { return false }

func (s *Session) metalDecodeLogitsQ8(id, pos int) []float32 { return nil }
