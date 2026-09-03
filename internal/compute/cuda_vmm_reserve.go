package compute

import (
	"fmt"
	"sync"
)

// VMMReservation decouples logical virtual address space reservation from physical memory backing (#10720, epic #2236).
// At session initialization, virtual capacity matching the full context length (e.g. 128k tokens)
// is reserved; physical device pages are mapped dynamically as tokens arrive, eliminating reallocation copies.
type VMMReservation struct {
	mu            sync.Mutex
	MaxTokens     int
	Stride        int
	VirtualBytes  int64
	PhysicalBytes int64
	MappedPages   map[int]int64 // page index -> physical bytes mapped
	pageSize      int64
}

// NewVMMReservation creates a virtual memory reservation for up to maxTokens.
func NewVMMReservation(maxTokens, stride int, pageSize int64) *VMMReservation {
	if pageSize <= 0 {
		pageSize = 2 * 1024 * 1024 // 2MB standard huge page
	}
	virtualBytes := int64(maxTokens * stride * 4)
	return &VMMReservation{
		MaxTokens:    maxTokens,
		Stride:       stride,
		VirtualBytes: virtualBytes,
		pageSize:     pageSize,
		MappedPages:  make(map[int]int64),
	}
}

// MapPhysicalPage maps a physical page to cover the given token offset.
func (r *VMMReservation) MapPhysicalPage(tokenOffset int) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if tokenOffset < 0 || tokenOffset >= r.MaxTokens {
		return 0, fmt.Errorf("token offset %d outside reserved virtual bounds [0, %d)", tokenOffset, r.MaxTokens)
	}

	byteOffset := int64(tokenOffset * r.Stride * 4)
	pageIdx := int(byteOffset / r.pageSize)

	if _, exists := r.MappedPages[pageIdx]; !exists {
		r.MappedPages[pageIdx] = r.pageSize
		r.PhysicalBytes += r.pageSize
	}
	return pageIdx, nil
}

// UnmapPhysicalPage releases physical backing for a page while preserving the virtual address slot.
func (r *VMMReservation) UnmapPhysicalPage(pageIdx int) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	bytes, ok := r.MappedPages[pageIdx]
	if !ok {
		return false
	}
	delete(r.MappedPages, pageIdx)
	r.PhysicalBytes -= bytes
	return true
}

// PageCount returns the number of physical pages currently mapped.
func (r *VMMReservation) PageCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.MappedPages)
}
