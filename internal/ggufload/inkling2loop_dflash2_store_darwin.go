//go:build darwin && arm64 && cgo

package ggufload

import (
	"sync"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// inkling2DFlash2Store views BF16 payloads of the draft checkpoint mapping as
// Metal weights without a copy. Views are kept, so a store shared by
// requests wires each payload into a Metal buffer once.
type inkling2DFlash2Store struct {
	store  *metalgemm.PackedGGUFStore
	mapped []byte
	mu     sync.Mutex
	views  map[[3]int]compute.Tensor
}

func openInkling2DFlash2Store(mapped []byte) (*inkling2DFlash2Store, error) {
	st, err := metalgemm.NewPackedGGUFStore(mapped)
	if err != nil {
		return nil, err
	}
	return &inkling2DFlash2Store{store: st, mapped: mapped, views: map[[3]int]compute.Tensor{}}, nil
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
	key := [3]int{offset, rows, cols}
	s.mu.Lock()
	defer s.mu.Unlock()
	if t, ok := s.views[key]; ok {
		return t, true
	}
	const bf16 = 30
	pm, err := s.store.Matrix(offset, bf16, rows, cols, 1)
	if err != nil {
		return compute.Tensor{}, false
	}
	t, err := compute.MetalInkling2Weight(pm, 0)
	if err != nil {
		return compute.Tensor{}, false
	}
	s.views[key] = t
	return t, true
}

func (s *inkling2DFlash2Store) Close() { s.store.Close() }

// drainClose closes the store once the shared device has finished every
// command that may still read its views.
func (s *inkling2DFlash2Store) drainClose() {
	if be, ok := compute.MetalInkling2Backend(); ok {
		if dev, ok := be.(compute.Inkling2Device); ok {
			if t, err := dev.Ink2Alloc(1); err == nil {
				_, _ = dev.Ink2Read(compute.Ink2At(t, 0), 1)
				be.Free(t)
			}
		}
	}
	s.Close()
}

// ink2Buffers is every Metal view the draft's matrices alias.
func (s *inkling2DFlash2Store) ink2Buffers() []metalgemm.Ink2Buf {
	if s == nil {
		return nil
	}
	return s.store.Ink2Buffers()
}
