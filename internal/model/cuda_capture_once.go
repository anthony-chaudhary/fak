package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// Separate optional qualification capability. Existing graphBackend remains
// unchanged, so all non-qualification builds retain their current behavior.
type captureOnceBackend interface {
	GraphOnceReplay(compute.Tensor, compute.KVStore, int) (compute.Tensor, bool)
	GraphOnceBegin(compute.Tensor, compute.KVStore, int) bool
	GraphOnceEnd()
	GraphOnceSave(compute.Tensor)
	GraphOnceAbort()
	GraphOnceInvalidate()
}

func (s *Session) captureOnceF32Weights() bool {
	if len(s.halW) == 0 {
		return false
	}
	for _, w := range s.halW {
		if w.Dtype != compute.F32 {
			return false
		}
	}
	return true
}
