//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// CUDA graph-owned arena. These strong references survive Recycle and Free.
// Only a successful native owner release permits clearing pins and freeing bytes.
// The native graph and upload cache share one stream across backend receivers.
// Access is serialized by cudaMu; retain this owner after any uncertain release.
var cudaGraphOnceOwner *cudaBackend

type cudaGraphOnceState struct {
	invalidated                              bool
	dependencyInvalidated                    bool
	dependencies                             map[*cudaBuf]cudaGraphOnceDependency
	reservationPointers                      []unsafe.Pointer
	reservationBytes                         []C.size_t
	reservationIDs                           []C.uint64_t
	borrowed                                 map[*cudaBuf]struct{}
	owner                                    unsafe.Pointer
	kv                                       *cudaKV
	kvBindingAttempted, kvBound              bool
	kvBackend                                *cudaBackend
	kvLayers, kvHeads, kvHeadDim, kvCapacity int
	kvTickets                                []uint64
	input, output                            Tensor
	buffers                                  []*cudaBuf
	keys, raw, values                        []dslice
	next                                     int
	capturing                                bool
}

// A receipt freezes allocation metadata and native ledger reservation tickets,
// not weight contents, COW versions or CUDA context identity. The host owner is
// separately retained without comparing interfaces, whose concrete HostBuffer
// implementation need not be comparable.
// Access, registration, validation and retirement all require existing cudaMu.
type cudaGraphOnceDependency struct {
	buf                            *cudaBuf
	ptr, scales                    unsafe.Pointer
	n, scalesN, device             int
	arenaIndex                     int
	reservationIndex               int
	reservation, scaleReservation  uint64
	class                          MemoryClass
	be                             *cudaBackend
	bornGen                        uint64
	managed, managedWeight, pinned bool
	budgetedWeightBytes            int64
	host                           uintptr
	hostDt                         Dtype
	hostLo                         Layout
	hostKeep                       HostBuffer
}

// graphOnceBindKVLocked freezes complete F32 allocations, not the changing row
// length. Caller holds cudaMu and retains the actual store. This is a one-shot
// observation before graph prewarm/create: zero/partial tickets cannot refresh.
func (s *cudaGraphOnceState) graphOnceBindKVLocked(c *cudaBackend, pos int) bool {
	if s == nil || s.kvBindingAttempted {
		return false
	}
	s.kvBindingAttempted = true
	k := s.kv
	if c == nil || k == nil || k.be != c || k.releaseRetained || k.snapshotCopy != nil ||
		k.cfg.Precision != KVPrecisionF32 || k.cfg.NumLayers <= 0 || k.cfg.NumLayers > 2147483647/3 ||
		len(k.K) != k.cfg.NumLayers || len(k.Kraw) != len(k.K) || len(k.V) != len(k.K) ||
		k.maxPos <= 0 || k.maxPos >= 2147483647 || pos < 0 || pos >= k.maxPos || len(k.pos) != pos ||
		k.cfg.NumKVHeads <= 0 || k.cfg.HeadDim <= 0 || k.cfg.NumKVHeads > 2147483647/k.cfg.HeadDim ||
		len(s.reservationPointers) != 0 || len(s.reservationBytes) != 0 || len(s.reservationIDs) != 0 {
		return false
	}
	w := k.cfg.NumKVHeads * k.cfg.HeadDim
	maxFloats := int(^uint(0)>>1) / 4
	// Capture kernels use wide row offsets, but ordinary F32 warmup writes still
	// take an int element offset. Keep this qualification's full capacity within
	// that signed range as well; this does not repair general eager growable KV.
	if maxFloats > 2147483647 {
		maxFloats = 2147483647
	}
	if k.maxPos > maxFloats/w {
		return false
	}
	required := k.maxPos * w
	s.kvBackend, s.kvLayers, s.kvHeads, s.kvHeadDim, s.kvCapacity = c, k.cfg.NumLayers, k.cfg.NumKVHeads, k.cfg.HeadDim, k.maxPos
	s.keys, s.raw, s.values = append([]dslice(nil), k.K...), append([]dslice(nil), k.Kraw...), append([]dslice(nil), k.V...)
	seen := make(map[unsafe.Pointer]struct{}, s.kvLayers*3)
	for l := 0; l < s.kvLayers; l++ {
		for _, d := range [3]dslice{s.keys[l], s.raw[l], s.values[l]} {
			if d.ptr == nil || d.dtype != F32 || d.cap < required || d.cap > maxFloats || d.len != pos*w {
				return false
			}
			if _, exists := seen[d.ptr]; exists {
				return false
			}
			seen[d.ptr] = struct{}{}
			bytes := d.cap * 4
			id := uint64(C.fcuda_alloc_reservation_id(d.ptr, C.size_t(bytes)))
			if id == 0 {
				return false
			}
			s.kvTickets = append(s.kvTickets, id)
			s.reservationPointers = append(s.reservationPointers, d.ptr)
			s.reservationBytes = append(s.reservationBytes, C.size_t(bytes))
			s.reservationIDs = append(s.reservationIDs, C.uint64_t(id))
		}
	}
	s.kvBound = true
	return true
}

// KV tickets occupy an immutable prefix in the existing single native batch.
// Only logical row lengths may change, within the retained capacity. The
// position/complete-row protocol is checked separately by graphOnceSameKV.
func (s *cudaGraphOnceState) graphOnceKVSlotsValid() (int, bool) {
	if s == nil || !s.kvBindingAttempted || !s.kvBound || s.kvLayers <= 0 || s.kvLayers > 2147483647/3 {
		return 0, false
	}
	k := s.kv
	count := s.kvLayers * 3
	if k == nil || s.kvBackend == nil || k.be != s.kvBackend || k.releaseRetained || k.snapshotCopy != nil ||
		k.cfg.Precision != KVPrecisionF32 || k.cfg.NumLayers != s.kvLayers || k.cfg.NumKVHeads != s.kvHeads || k.cfg.HeadDim != s.kvHeadDim || k.maxPos != s.kvCapacity ||
		len(k.K) != s.kvLayers || len(k.Kraw) != s.kvLayers || len(k.V) != s.kvLayers ||
		len(s.keys) != s.kvLayers || len(s.raw) != s.kvLayers || len(s.values) != s.kvLayers || len(s.kvTickets) != count ||
		len(s.reservationPointers) < count || len(s.reservationBytes) < count || len(s.reservationIDs) < count {
		return 0, false
	}
	at := 0
	for l := 0; l < s.kvLayers; l++ {
		for _, pair := range [3][2]dslice{{k.K[l], s.keys[l]}, {k.Kraw[l], s.raw[l]}, {k.V[l], s.values[l]}} {
			live, saved := pair[0], pair[1]
			if saved.ptr == nil || saved.dtype != F32 || saved.cap <= 0 || saved.cap > int(^uint(0)>>1)/4 || saved.cap > 2147483647 ||
				live.ptr != saved.ptr || live.cap != saved.cap || live.dtype != saved.dtype || live.len < 0 || live.len > live.cap ||
				s.kvTickets[at] == 0 || s.reservationPointers[at] != saved.ptr || s.reservationBytes[at] != C.size_t(saved.cap*4) ||
				uint64(s.reservationIDs[at]) != s.kvTickets[at] {
				return 0, false
			}
			at++
		}
	}
	return count, true
}

func graphOnceDependencyOf(b *cudaBuf) cudaGraphOnceDependency {
	d := cudaGraphOnceDependency{buf: b, ptr: b.ptr, scales: b.scales, n: b.n, scalesN: b.scalesN, device: b.device,
		class: b.class, be: b.be, bornGen: b.bornGen, managed: b.managed, managedWeight: b.managedWeight, pinned: b.graphOncePinned,
		budgetedWeightBytes: b.budgetedWeightBytes, host: b.host, hostDt: b.hostDt, hostLo: b.hostLo, hostKeep: b.hostKeep}
	// Acquisition is capture-registration work only. Unknown/default-profile or
	// partial/interior views never receive a usable ticket and are not retried.
	if d.ptr != nil && d.n > 0 {
		d.reservation = uint64(C.fcuda_alloc_reservation_id(d.ptr, C.size_t(d.n)))
	}
	if d.scales != nil && d.scalesN > 0 {
		d.scaleReservation = uint64(C.fcuda_alloc_reservation_id(d.scales, C.size_t(d.scalesN)))
	}
	return d
}

func (d cudaGraphOnceDependency) matches(b *cudaBuf) bool {
	return b != nil && b == d.buf && b.ptr != nil && b.n > 0 && b.device >= 0 && !b.invalidForSubmit() &&
		b.scalesN >= 0 && (b.scales == nil) == (b.scalesN == 0) &&
		b.ptr == d.ptr && b.scales == d.scales && b.n == d.n && b.scalesN == d.scalesN && b.device == d.device &&
		b.class == d.class && b.be == d.be && b.bornGen == d.bornGen && b.managed == d.managed &&
		b.managedWeight == d.managedWeight && b.graphOncePinned == d.pinned && b.budgetedWeightBytes == d.budgetedWeightBytes &&
		b.host == d.host && b.hostDt == d.hostDt && b.hostLo == d.hostLo
}

func (s *cudaGraphOnceState) graphOnceRecordDependency(b *cudaBuf, arenaIndex int) {
	if s.dependencies == nil {
		s.dependencies = make(map[*cudaBuf]cudaGraphOnceDependency)
	}
	d, exists := s.dependencies[b]
	if !exists {
		d = graphOnceDependencyOf(b)
		d.arenaIndex = arenaIndex
		d.reservationIndex = len(s.reservationPointers)
		// Root the immutable receipt before growing any ticket array. An interrupted
		// registration leaves an inconsistent inventory that validation refuses.
		s.dependencies[b] = d
		s.reservationPointers = append(s.reservationPointers, d.ptr)
		s.reservationBytes = append(s.reservationBytes, C.size_t(d.n))
		s.reservationIDs = append(s.reservationIDs, C.uint64_t(d.reservation))
		if d.scales != nil {
			s.reservationPointers = append(s.reservationPointers, d.scales)
			s.reservationBytes = append(s.reservationBytes, C.size_t(d.scalesN))
			s.reservationIDs = append(s.reservationIDs, C.uint64_t(d.scaleReservation))
		}
	}
	// Re-observation can detect a change, but can never overwrite the first
	// receipt or erase evidence when the caller restores the original metadata.
	if !d.matches(b) || d.reservation == 0 || (d.scales != nil && d.scaleReservation == 0) || (d.pinned && d.arenaIndex < 0) {
		s.dependencyInvalidated = true
		s.invalidated = true
	}
}

// Match immutable receipt slots before crossing cgo. Slot/index checks precede
// every array access. The arrays are built during capture and never refreshed on
// replay; an updated native ticket cannot silently replace a stale saved one.
func (d cudaGraphOnceDependency) reservationSlotsMatch(s *cudaGraphOnceState) bool {
	count := 1
	if d.scales != nil {
		count = 2
	}
	if d.reservation == 0 || d.reservationIndex < 0 || len(s.reservationPointers) < count ||
		d.reservationIndex > len(s.reservationPointers)-count {
		return false
	}
	i := d.reservationIndex
	if s.reservationPointers[i] != d.ptr || s.reservationBytes[i] != C.size_t(d.n) ||
		uint64(s.reservationIDs[i]) != d.reservation {
		return false
	}
	return count == 1 || (d.scaleReservation != 0 && s.reservationPointers[i+1] == d.scales &&
		s.reservationBytes[i+1] == C.size_t(d.scalesN) && uint64(s.reservationIDs[i+1]) == d.scaleReservation)
}

func (s *cudaGraphOnceState) graphOnceDependenciesValid() bool {
	if s == nil || s.dependencyInvalidated || len(s.dependencies) != len(s.buffers)+len(s.borrowed) ||
		len(s.reservationPointers) == 0 || len(s.reservationPointers) > 2147483647 ||
		len(s.reservationBytes) != len(s.reservationPointers) || len(s.reservationIDs) != len(s.reservationPointers) {
		return false
	}
	input, ok := s.input.buf.(*cudaBuf)
	if !ok || input == nil {
		return false
	}
	d, ok := s.dependencies[input]
	if !ok || !d.pinned || d.arenaIndex != 0 || !d.matches(input) {
		return false
	}
	expectedTickets, kvOK := s.graphOnceKVSlotsValid()
	if !kvOK {
		return false
	}
	for i, b := range s.buffers {
		d, ok := s.dependencies[b]
		if !ok || !d.pinned || d.arenaIndex != i || !d.matches(b) || !d.reservationSlotsMatch(s) {
			return false
		}
		count := 1
		if d.scales != nil {
			count = 2
		}
		if expectedTickets > len(s.reservationPointers)-count {
			return false
		}
		expectedTickets += count
	}
	for b := range s.borrowed {
		d, ok := s.dependencies[b]
		if !ok || d.pinned || d.arenaIndex != -1 || !d.matches(b) || !d.reservationSlotsMatch(s) {
			return false
		}
		count := 1
		if d.scales != nil {
			count = 2
		}
		if expectedTickets > len(s.reservationPointers)-count {
			return false
		}
		expectedTickets += count
	}
	if expectedTickets != len(s.reservationPointers) {
		return false
	}
	// Exactly one native metadata call per boundary, not one cgo transition per
	// dependency. The arrays and retained subject owners survive that whole call.
	valid := C.fcuda_alloc_reservations_match((*unsafe.Pointer)(unsafe.Pointer(&s.reservationPointers[0])),
		&s.reservationBytes[0], &s.reservationIDs[0], C.int(len(s.reservationPointers))) != 0
	runtime.KeepAlive(s)
	return valid
}

func (c *cudaBackend) graphOncePinLocked(b *cudaBuf) {
	s := c.graphOnce
	if s == nil || !s.capturing || b == nil {
		return
	}
	arenaIndex := -1
	if !b.graphOncePinned {
		arenaIndex = len(s.buffers)
		b.graphOncePinned = true
		s.buffers = append(s.buffers, b)
	}
	s.graphOnceRecordDependency(b, arenaIndex)
}

// Borrowed dependencies remain borrowed and are never freed with the arena.
// Even owned operands are re-observed to detect metadata changes during capture.
func (c *cudaBackend) graphOnceObserveLocked(b *cudaBuf) {
	s := c.graphOnce
	if s == nil || !s.capturing || b == nil {
		return
	}
	if !b.graphOncePinned {
		if s.borrowed == nil {
			s.borrowed = make(map[*cudaBuf]struct{})
		}
		s.borrowed[b] = struct{}{}
	}
	s.graphOnceRecordDependency(b, -1)
}
