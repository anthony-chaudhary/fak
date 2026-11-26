//go:build darwin && arm64 && cgo

package ggufload

import (
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// inkling2DFlash2Store views BF16 payloads of the draft checkpoint mapping as
// Metal weights without a copy.
type inkling2DFlash2Store struct {
	store  *metalgemm.PackedGGUFStore
	mapped []byte
}

func openInkling2DFlash2Store(mapped []byte) (*inkling2DFlash2Store, error) {
	st, err := metalgemm.NewPackedGGUFStore(mapped)
	if err != nil {
		return nil, err
	}
	return &inkling2DFlash2Store{store: st, mapped: mapped}, nil
}

// view is the in-place weight of raw, a payload inside the mapping; ok is
// false when the device cannot alias it.
func (s *inkling2DFlash2Store) view(raw []byte, rows, cols int) (compute.Tensor, bool) {
	if s == nil || len(raw) == 0 {
		return compute.Tensor{}, false
	}
	offset := int(uintptr(unsafe.Pointer(unsafe.SliceData(raw))) - uintptr(unsafe.Pointer(unsafe.SliceData(s.mapped))))
	if offset < 0 || offset+len(raw) > len(s.mapped) {
		return compute.Tensor{}, false
	}
	const bf16 = 30
	pm, err := s.store.Matrix(offset, bf16, rows, cols, 1)
	if err != nil {
		return compute.Tensor{}, false
	}
	t, err := compute.MetalInkling2Weight(pm, 0)
	return t, err == nil
}

func (s *inkling2DFlash2Store) Close() { s.store.Close() }
