// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"errors"
)

// Hardware alignment constants for AMD Strix Halo (GFX1151 / RDNA 3.5 / Zen 5).
const (
	// GFX1151VectorAlignment is the 16-byte alignment boundary required for
	// RDNA 3.5 vector load/store instructions (global_load_dwordx4 / global_store_dwordx4).
	GFX1151VectorAlignment = 16

	// GFX1151CacheLineAlignment is the 64-byte cache-line boundary for the Zen 5 / GFX1151 memory crossbar.
	GFX1151CacheLineAlignment = 64

	// GFX1151BurstAlignment is the 128-byte burst alignment for the 256-bit LPDDR5X memory controller.
	GFX1151BurstAlignment = 128
)

// Typed errors for zero-copy tensor slicing and view operations.
var (
	// ErrNilParentBuffer indicates a nil parent UMA buffer was provided.
	ErrNilParentBuffer = errors.New("strix/view: parent UMA buffer is nil")

	// ErrSliceOutOfBounds indicates the slice bounds exceed the parent buffer capacity.
	ErrSliceOutOfBounds = errors.New("strix/view: slice bounds exceed parent buffer capacity")

	// ErrMisalignedSliceOffset indicates the slice offset violates the required hardware alignment.
	ErrMisalignedSliceOffset = errors.New("strix/view: slice offset violates hardware alignment boundary")

	// ErrInvalidSliceLength indicates a non-positive or invalid slice byte length.
	ErrInvalidSliceLength = errors.New("strix/view: slice byte length must be positive and within bounds")

	// ErrInvalidShape indicates shape dimensions do not match the byte length of the tensor view.
	ErrInvalidShape = errors.New("strix/view: shape dimensions do not match tensor element count")

	// ErrMisalignedStride indicates a tensor stride violates hardware alignment boundaries.
	ErrMisalignedStride = errors.New("strix/view: tensor stride violates hardware alignment boundary")

	// ErrViewClosed indicates an operation was attempted on an invalid or closed view.
	ErrViewClosed = errors.New("strix/view: tensor view is closed or uninitialized")
)

// DataType defines supported tensor element data types for Strix Halo kernels.
type DataType string

const (
	DTypeFP16  DataType = "float16"
	DTypeBF16  DataType = "bfloat16"
	DTypeFP32  DataType = "float32"
	DTypeINT8  DataType = "int8"
	DTypeUINT8 DataType = "uint8"
	DTypeINT4  DataType = "int4"
	DTypeINT32 DataType = "int32"
)

// ElementSize returns the byte size of a single element of this DataType.
func (d DataType) ElementSize() int {
	switch d {
	case DTypeFP16, DTypeBF16:
		return 2
	case DTypeFP32, DTypeINT32:
		return 4
	case DTypeINT8, DTypeUINT8:
		return 1
	case DTypeINT4:
		return 1 // packed 2 elements per byte; minimum addressable byte unit is 1
	default:
		return 1
	}
}

// SliceDescriptor encapsulates the geometric slicing parameters and hardware constraints
// for deriving a zero-copy sub-tensor view from a parent UMA buffer or existing view.
type SliceDescriptor struct {
	// ByteOffset is the offset in bytes relative to the start of the source buffer/view.
	ByteOffset int `json:"byte_offset"`

	// ByteLength is the length in bytes of the desired slice.
	ByteLength int `json:"byte_length"`

	// DType is the data type of the sliced tensor elements. If empty, inherits from source.
	DType DataType `json:"dtype,omitempty"`

	// Shape specifies the tensor dimension extents [d0, d1, ..., dn].
	Shape []int `json:"shape,omitempty"`

	// ByteStrides specifies byte stride per dimension. If nil, contiguous row-major strides are computed.
	ByteStrides []int `json:"byte_strides,omitempty"`

	// AlignBytes specifies the required hardware alignment (e.g. 16 for GFX1151 vector load, 64 for cache line).
	// If 0, defaults to GFX1151VectorAlignment (16 bytes).
	AlignBytes int `json:"align_bytes,omitempty"`
}

// TensorView represents a lightweight, zero-copy header-only tensor view over an AMD Strix Halo
// Unified Memory Architecture (UMA) allocation. On GFX1151 silicon, HostPtr and DevPtr point
// to the exact same physical virtual address across the 256-bit memory crossbar (uintptr(HostPtr) == uintptr(DevPtr)).
//
// TensorView embeds a strong reference to parent *UMABuffer and optional parentView *TensorView,
// ensuring the Go garbage collector cannot finalize or evict the backing physical pages while
// this view remains live.
type TensorView struct {
	parent      *UMABuffer
	parentView  *TensorView
	hostPtr     uintptr
	devPtr      uintptr
	byteOffset  int
	byteLength  int
	dtype       DataType
	shape       []int
	byteStrides []int
	alignBytes  int
}
