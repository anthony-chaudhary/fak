//go:build ignore

// SOURCE-ONLY: integration sketch for package compute, not a standalone package.
// Original platform constraint: vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
#include "vulkan_v41_packed_storage.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// Prototype-only explicit owner. Deliberately NOT embedded in vulkanBackend:
// no ordinary backend gains admission by linking this file. Its constructor
// has no model caller. Session adoption/Close wiring needs separate review.
type vulkanV41PackedRows struct {
	backend    *vulkanBackend
	generation uint64
	live       map[*vulkanV41PackedRow]bool
	closed     bool
}

type vulkanV41PackedRow struct {
	owner        *vulkanV41PackedRows
	extent       V41PackedRowExtent // immutable, including after release
	buf          *vulkanBuf
	allocationID uint64
	reservation  uint64
	poisoned     bool
	// Separate numerical eligibility: zero-fill is initialized but has invalid zero scales.
	attentionValid map[int]bool
}

func (h *vulkanV41PackedRow) V41PackedRowExtent() V41PackedRowExtent {
	if h == nil {
		return V41PackedRowExtent{}
	}
	return h.extent
}

var _ V41PackedRowBackend = (*vulkanV41PackedRows)(nil)

func packedStorageError(message string) error {
	return fmt.Errorf("%w: %s", ErrV41PackedRowGeometry, message)
}

// All helpers below except the extent accessor require vulkanMu. No method
// recursively takes that lock or uses generic Tensor ownership/free machinery.
func (p *vulkanV41PackedRows) state() error {
	if p == nil || p.backend == nil || p.closed || p.live == nil {
		return packedStorageError("closed or missing packed owner")
	}
	var generation C.uint64_t
	if err := vulkanV41RowLogStatus("V41PackedRowState", int(C.fvk_v41_packed_state(&generation))); err != nil {
		return err
	}
	if uint64(generation) != p.generation {
		return packedStorageError("stale device generation")
	}
	return nil
}

func newVulkanV41PackedRows(v *vulkanBackend) (*vulkanV41PackedRows, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if v == nil {
		return nil, packedStorageError("missing Vulkan backend")
	}
	var generation C.uint64_t
	if err := vulkanV41RowLogStatus("V41PackedRowOwner", int(C.fvk_v41_packed_state(&generation))); err != nil {
		return nil, err
	}
	return &vulkanV41PackedRows{backend: v, generation: uint64(generation), live: make(map[*vulkanV41PackedRow]bool)}, nil
}

func (p *vulkanV41PackedRows) V41PackedRowAdmission(format V41PackedRowFormat) (bool, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if _, err := NewV41PackedRowExtent(format, 0, 0); err != nil {
		return false, err
	}
	if err := p.state(); err != nil {
		return false, err
	}
	return true, nil
}

func (p *vulkanV41PackedRows) handle(raw V41PackedRowStorage, allowPoison bool) (*vulkanV41PackedRow, error) {
	h, ok := raw.(*vulkanV41PackedRow)
	if !ok || h == nil || h.owner != p || !p.live[h] || (!allowPoison && h.poisoned) {
		return nil, packedStorageError("foreign, released, incomplete or poisoned packed handle")
	}
	n, err := h.extent.CapacityBytes()
	if err != nil {
		return nil, err
	}
	if n == 0 {
		if h.buf != nil || h.reservation != 0 || h.allocationID != 0 {
			return nil, packedStorageError("invalid empty backing")
		}
		return h, nil
	}
	if h.buf == nil || h.buf.ptr == nil || h.buf.n != n || h.buf.class != MemoryKVCache ||
		h.buf.scalePtr != nil || len(h.buf.q8Chunks) != 0 {
		return nil, packedStorageError("invalid packed backing")
	}
	var info C.fvk_buffer_reservation_info
	if C.fvk_buffer_reservation(h.buf.ptr, &info) == 0 || uint64(info.buffer_bytes) != uint64(n) ||
		uint64(info.reservation_bytes) != h.reservation || uint64(info.allocation_id) != h.allocationID || info.binding_offset != 0 {
		return nil, packedStorageError("packed backing reservation changed")
	}
	return h, nil
}

// Alloc is a proposed lifecycle addition, NOT part of V41PackedRowBackend.
// Every capacity byte is zeroed before success, including spare records. Only
// extent.Rows() becomes readable. This establishes initialization, not numerical
// quantizer provenance. No append or mutable-extent operation is supplied.
func (p *vulkanV41PackedRows) Alloc(extent V41PackedRowExtent) (out V41PackedRowStorage, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var h *vulkanV41PackedRow
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41PackedRowAlloc")
			out = nil
			if h != nil {
				h.poisoned = true
			} // retained until explicit Close
		}
	}()
	n, err := extent.CapacityBytes()
	if err != nil {
		return nil, err
	}
	if err := p.state(); err != nil {
		return nil, err
	}
	p.backend.checkResourceCap(n, "V4.1 packed KV")
	h = &vulkanV41PackedRow{owner: p, extent: extent, poisoned: true}
	// Register before allocating. Any post-allocation failure stays owned and
	// accounted until Close can safely release it, never anonymous orphan work.
	p.live[h] = true
	if n != 0 {
		h.buf = p.backend.dallocForClass(n, MemoryKVCache, "V4.1 packed KV")
		var info C.fvk_buffer_reservation_info
		if C.fvk_buffer_reservation(h.buf.ptr, &info) == 0 || uint64(info.buffer_bytes) != uint64(n) ||
			info.binding_offset != 0 || info.allocation_id == 0 || uint64(info.reservation_bytes) < uint64(n) {
			return nil, packedStorageError("unavailable packed allocation accounting")
		}
		h.allocationID, h.reservation = uint64(info.allocation_id), uint64(info.reservation_bytes)
		if err := vulkanV41RowLogStatus("V41PackedRowZero", int(C.fvk_v41_packed_zero(h.buf.ptr, C.uint64_t(p.generation), C.size_t(n)))); err != nil {
			return nil, err
		}
	}
	h.poisoned = false
	return h, nil
}

// Upload replaces complete initialized records without casting or retaining
// host bytes. Validation happens before allocation/commands. A selected failure
// poisons the destination; it cannot be read/copied/replayed after the error.
func (p *vulkanV41PackedRows) Upload(dst V41PackedRowStorage, first int, image []byte) error {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if err := p.state(); err != nil {
		return err
	}
	h, err := p.handle(dst, false)
	if err != nil {
		return err
	}
	if len(image)%288 != 0 {
		return packedStorageError("partial packed record upload")
	}
	offset, count, err := h.extent.ByteRange(first, len(image)/288)
	if err != nil {
		return err
	}
	if count == 0 {
		return nil
	}
	h.poisoned = true
	status := C.fvk_v41_packed_upload(h.buf.ptr, C.uint64_t(p.generation), C.size_t(h.buf.n), C.size_t(offset), unsafe.Pointer(&image[0]), C.size_t(count))
	if err := vulkanV41RowLogStatus("V41PackedRowUpload", int(status)); err != nil {
		return err
	}
	if h.attentionValid == nil {
		h.attentionValid = make(map[int]bool)
	}
	for row := 0; row < len(image)/288; row++ {
		h.attentionValid[first+row] = v41PackedScaleRecordValid(image[row*288 : row*288+288])
	}
	h.poisoned = false
	return nil
}

func (p *vulkanV41PackedRows) V41PackedRowCopy(dst V41PackedRowStorage, dstRow int, src V41PackedRowStorage, srcRow, rows int) error {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if err := p.state(); err != nil {
		return err
	}
	d, err := p.handle(dst, false)
	if err != nil {
		return err
	}
	s, err := p.handle(src, false)
	if err != nil {
		return err
	}
	if d.extent.Format() != s.extent.Format() {
		return packedStorageError("packed format mismatch")
	}
	doff, n, err := d.extent.ByteRange(dstRow, rows)
	if err != nil {
		return err
	}
	soff, _, err := s.extent.ByteRange(srcRow, rows)
	if err != nil {
		return err
	}
	if rows == 0 {
		return nil
	}
	// Conservative same-allocation refusal, including disjoint spans. No aliases
	// are exported by this prototype, but identity is physical, not wrapper-only.
	if d == s || d.allocationID == s.allocationID {
		return packedStorageError("packed allocation alias")
	}
	d.poisoned = true
	status := C.fvk_v41_packed_copy(d.buf.ptr, C.size_t(d.buf.n), C.size_t(doff), s.buf.ptr, C.size_t(s.buf.n), C.size_t(soff), C.uint64_t(p.generation), C.size_t(n))
	if err := vulkanV41RowLogStatus("V41PackedRowCopy", int(status)); err != nil {
		return err
	}
	if d.attentionValid == nil {
		d.attentionValid = make(map[int]bool)
	}
	for row := 0; row < rows; row++ {
		d.attentionValid[dstRow+row] = s.attentionValid[srcRow+row]
	}
	d.poisoned = false
	return nil
}

func (p *vulkanV41PackedRows) release(h *vulkanV41PackedRow) {
	if h.buf != nil && h.buf.ptr != nil {
		// state() established no open/in-flight batch and no uncertain queue.
		// fvk_free may retain physical memory in its pool. That is NOT a driver
		// release and does not reduce native pooled-reservation accounting.
		C.fvk_free(h.buf.ptr)
		h.buf.ptr, h.buf.n = nil, 0
	}
	delete(p.live, h)
}

func (p *vulkanV41PackedRows) Free(raw V41PackedRowStorage) error {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if err := p.state(); err != nil {
		return err
	}
	h, err := p.handle(raw, true)
	if err != nil {
		return err
	}
	p.release(h)
	return nil
}

// Close releases all retained allocations, including failed unpublished ones.
// Poison/busy/stale device returns error with ownership and charges INTACT.
// Native device teardown must not precede owner Close on the healthy path.
func (p *vulkanV41PackedRows) Close() error {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if err := p.state(); err != nil {
		return err
	}
	for h := range p.live {
		p.release(h)
	}
	p.closed = true
	return nil
}

// Accounting reports owner-live bytes, including spare capacity and quarantined
// failed allocations. It is NOT process total or a claim of physical release.
// Generic Session.halW scans cannot see these handles and must NOT be used as
// this owner's ledger. Each standalone native allocation is charged once.
func (p *vulkanV41PackedRows) Accounting() (class MemoryClass, capacity, reserved uint64, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if p == nil || p.live == nil {
		return MemoryKVCache, 0, 0, packedStorageError("missing packed owner")
	}
	for h := range p.live {
		n := uint64(0)
		if h.buf != nil {
			n = uint64(h.buf.n)
		}
		if n > math.MaxUint64-capacity || h.reservation > math.MaxUint64-reserved {
			return MemoryKVCache, 0, 0, packedStorageError("accounting overflow")
		}
		if n != 0 && h.reservation == 0 {
			return MemoryKVCache, 0, 0, packedStorageError("allocation accounting unavailable")
		}
		capacity += n
		reserved += h.reservation
	}
	return MemoryKVCache, capacity, reserved, nil
}

// Numerical eligibility is separate from storage initialization. Every payload
// nibble is finite; only the 32 scale bytes restrict the attention boundary.
func v41PackedScaleRecordValid(record []byte) bool {
	if len(record) != 288 {
		return false
	}
	for _, scale := range record[256:] {
		if scale == 0 || scale >= 127 {
			return false
		}
	}
	return true
}
