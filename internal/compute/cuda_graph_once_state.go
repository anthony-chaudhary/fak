//go:build cuda && cgo

package compute

import "unsafe"

// CUDA graph-owned arena. These strong references survive Recycle and Free.
// Only a successful native owner release permits clearing pins and freeing bytes.
// The native graph and upload cache share one stream across backend receivers.
// Access is serialized by cudaMu; retain this owner after any uncertain release.
var cudaGraphOnceOwner *cudaBackend

type cudaGraphOnceState struct {
	invalidated       bool
	borrowed          map[*cudaBuf]struct{}
	owner             unsafe.Pointer
	kv                *cudaKV
	input, output     Tensor
	buffers           []*cudaBuf
	keys, raw, values []dslice
	next              int
	capturing         bool
}

func (c *cudaBackend) graphOncePinLocked(b *cudaBuf) {
	s := c.graphOnce
	if s == nil || !s.capturing || b == nil || b.graphOncePinned {
		return
	}
	b.graphOncePinned = true
	s.buffers = append(s.buffers, b)
}

// Borrowed dependencies (weights) are not arena-owned: keep references and
// invalidate before explicit Free, never free them as part of arena teardown.
func (c *cudaBackend) graphOnceObserveLocked(b *cudaBuf) {
	s := c.graphOnce
	if s == nil || !s.capturing || b == nil || b.graphOncePinned {
		return
	}
	if s.borrowed == nil {
		s.borrowed = make(map[*cudaBuf]struct{})
	}
	s.borrowed[b] = struct{}{}
}
