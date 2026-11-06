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
	invalidated           bool
	dependencyInvalidated bool
	dependencies          map[*cudaBuf]cudaGraphOnceDependency
	reservationPointers   []unsafe.Pointer
	reservationBytes      []C.size_t
	reservationIDs        []C.uint64_t
	borrowed              map[*cudaBuf]struct{}
	owner                 unsafe.Pointer
	kv                    *cudaKV
	input, output         Tensor
	buffers               []*cudaBuf
	keys, raw, values     []dslice
	next                  int
	capturing             bool
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
	expectedTickets := 0
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
