package compute

import (
	"errors"
	"fmt"
	"math"
)

// V41PackedRowFormat identifies a complete record encoding, not a Tensor dtype
// or a GGUF weight format. Zero and unknown values are invalid.
type V41PackedRowFormat uint8

const (
	// V41MainKVPer16E4M3NoGlobal stores 512 E2M1 values, low nibble first:
	// 224 NoPE payload bytes, 32 RoPE payload bytes, 28 NoPE scale bytes,
	// then 4 RoPE scale bytes. Each scale is E4M3 for 16 consecutive values;
	// there is no global multiplier. This names storage, not a quantizer's
	// rounding policy or permission to quantize an existing publication again.
	V41MainKVPer16E4M3NoGlobal V41PackedRowFormat = 1
)

// ErrV41PackedRowGeometry marks invalid or unrepresentable packed storage.
var ErrV41PackedRowGeometry = errors.New("compute: invalid V4.1 packed-row geometry")

// V41PackedRowExtent is immutable metadata for a contiguous row-record buffer.
// Rows is the initialized logical prefix; Capacity includes spare records, which
// readers must not address. It owns no bytes and proves neither allocation,
// initialization, backend provenance nor completion. A backend must establish
// those facts before publishing a handle with this extent.
//
// This format has no inter-record padding. Backend allocation alignment is not
// part of CapacityBytes and must be accounted for separately by its owner.
type V41PackedRowExtent struct {
	format   V41PackedRowFormat
	rows     int
	capacity int
}

// NewV41PackedRowExtent checks all byte products without allocating storage.
// Empty prefixes and zero-capacity buffers are valid for a known format.
func NewV41PackedRowExtent(format V41PackedRowFormat, rows, capacity int) (V41PackedRowExtent, error) {
	e := V41PackedRowExtent{format: format, rows: rows, capacity: capacity}
	if err := e.Validate(); err != nil {
		return V41PackedRowExtent{}, err
	}
	return e, nil
}

// Validate also rejects the zero value. Device-specific integer/index limits
// remain the backend's responsibility before narrowing any checked host value.
func (e V41PackedRowExtent) Validate() error {
	if e.format != V41MainKVPer16E4M3NoGlobal || e.rows < 0 || e.capacity < e.rows || e.capacity > math.MaxInt/288 {
		return fmt.Errorf("%w: format=%d rows=%d capacity=%d", ErrV41PackedRowGeometry, e.format, e.rows, e.capacity)
	}
	return nil
}

// Format returns the immutable record format.
func (e V41PackedRowExtent) Format() V41PackedRowFormat { return e.format }

// Rows returns the initialized logical row count, not the capacity.
func (e V41PackedRowExtent) Rows() int { return e.rows }

// Capacity returns the number of reserved records.
func (e V41PackedRowExtent) Capacity() int { return e.capacity }

// LogicalWidth returns values per row, or zero for an invalid extent.
func (e V41PackedRowExtent) LogicalWidth() int {
	if e.Validate() != nil {
		return 0
	}
	return 512
}

// RecordBytes returns the byte stride, or zero for an invalid extent.
func (e V41PackedRowExtent) RecordBytes() int {
	if e.Validate() != nil {
		return 0
	}
	return 288
}

// ScaleOffsets returns row-relative byte offsets for NoPE and RoPE scales.
// Their lengths are respectively 28 and 4; payload occupies bytes [0,256).
func (e V41PackedRowExtent) ScaleOffsets() (nope, rope int, err error) {
	if err = e.Validate(); err != nil {
		return 0, 0, err
	}
	return 256, 284, nil
}

// CapacityBytes is the checked packed-record capacity, excluding allocator
// overhead. It is not evidence of actual resident memory consumption.
func (e V41PackedRowExtent) CapacityBytes() (int, error) {
	if err := e.Validate(); err != nil {
		return 0, err
	}
	return e.capacity * 288, nil
}

// ByteRange returns the byte offset and length of a logical row span. Empty
// spans at Rows are valid; spare capacity is never readable. Subtraction checks
// avoid overflowing first+rows before the already-checked byte multiplication.
func (e V41PackedRowExtent) ByteRange(first, rows int) (offset, length int, err error) {
	if err = e.Validate(); err != nil {
		return 0, 0, err
	}
	if first < 0 || rows < 0 || rows > e.rows || first > e.rows-rows {
		return 0, 0, fmt.Errorf("%w: span [%d,+%d) of %d", ErrV41PackedRowGeometry, first, rows, e.rows)
	}
	return first * 288, rows * 288, nil
}

// V41PackedRowStorage is an opaque backend-owned handle, deliberately not a
// Tensor. Its extent must remain stable for the handle's lifetime. Retention,
// release, alias identity, initialization and completion belong to the backend;
// callers cannot manufacture those facts by constructing an extent. A handle
// with a larger initialized prefix requires new metadata after completion.
type V41PackedRowStorage interface {
	V41PackedRowExtent() V41PackedRowExtent
}

// V41PackedRowBackend is an optional byte-preserving storage capability. No
// backend implementation or dispatch is supplied by this contract. It does not
// extend Backend, V41RowLogBackend, attention, indexer, Tensor or Dtype.
//
// Admission is observation only: no allocation, upload, dispatch or recovery.
// An unknown or zero format returns ErrV41PackedRowGeometry, not healthy absence.
// Only initial (false,nil) is healthy absence; a selected failure must propagate
// without F32/CPU replay. Later (false,nil) also cannot change a selected path.
// Before every copy and before side effects, implementations must validate both
// extents and ByteRanges, require the same backend owner, and verify live backing
// allocations cover CapacityBytes. Reject nil (including typed-nil), foreign,
// released or incomplete handles without panic, and enforce device integer
// limits before narrowing. Metadata alone does not prove backing ownership.
//
// Copy preserves complete records and row order without arithmetic, casting,
// re-quantization or scale changes. Both spans must be within initialized Rows
// and have identical formats. Overlap in the same underlying allocation must
// refuse, including aliases represented by different handles. Success means
// the copy completed and is visible to subsequent consumers. The caller owns
// exclusive destination access and retains both handles through completion;
// an error never authorizes publishing the destination or replaying elsewhere.
// A zero-row copy is a no-op after the same admission and handle validation.
// This interface does not allocate, grow, append, or release storage.
type V41PackedRowBackend interface {
	V41PackedRowAdmission(format V41PackedRowFormat) (supported bool, err error)
	V41PackedRowCopy(dst V41PackedRowStorage, dstRow int, src V41PackedRowStorage, srcRow, rows int) error
}
