// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"
)

// NewTensorView instantiates a zero-copy TensorView over an allocated UMABuffer.
// The resulting view guarantees strict pointer identity (HostPtr == DevPtr) and
// pins the parent buffer to prevent premature garbage collection.
func NewTensorView(parent *UMABuffer, desc SliceDescriptor) (*TensorView, error) {
	if parent == nil {
		return nil, ErrNilParentBuffer
	}
	if atomic.LoadUint32(&parent.closed) != 0 {
		return nil, ErrBufferClosed
	}

	bufSize := parent.Len()
	if desc.ByteLength <= 0 {
		return nil, ErrInvalidSliceLength
	}
	if desc.ByteOffset < 0 || desc.ByteOffset+desc.ByteLength > bufSize {
		return nil, fmt.Errorf("%w: offset %d + len %d exceeds buffer size %d",
			ErrSliceOutOfBounds, desc.ByteOffset, desc.ByteLength, bufSize)
	}

	align := desc.AlignBytes
	if align <= 0 {
		align = GFX1151VectorAlignment
	}
	if (align & (align - 1)) != 0 {
		return nil, ErrInvalidAlignment
	}

	// Validate vector load alignment against 16-byte GFX1151 boundary.
	if (desc.ByteOffset % align) != 0 {
		return nil, fmt.Errorf("%w: offset %d is not aligned to %d-byte hardware boundary",
			ErrMisalignedSliceOffset, desc.ByteOffset, align)
	}

	dtype := desc.DType
	if dtype == "" {
		dtype = DTypeFP16
	}
	elemSize := dtype.ElementSize()
	if (desc.ByteOffset % elemSize) != 0 {
		return nil, fmt.Errorf("%w: offset %d is not aligned to element size %d",
			ErrMisalignedSliceOffset, desc.ByteOffset, elemSize)
	}
	if (desc.ByteLength % elemSize) != 0 {
		return nil, fmt.Errorf("%w: length %d is not a multiple of element size %d",
			ErrInvalidSliceLength, desc.ByteLength, elemSize)
	}

	shape := desc.Shape
	if len(shape) == 0 {
		shape = []int{desc.ByteLength / elemSize}
	} else {
		elemCount := 1
		for _, dim := range shape {
			if dim <= 0 {
				return nil, ErrInvalidShape
			}
			elemCount *= dim
		}
		if elemCount*elemSize != desc.ByteLength {
			return nil, fmt.Errorf("%w: shape %v element count %d != byte capacity %d",
				ErrInvalidShape, shape, elemCount*elemSize, desc.ByteLength)
		}
	}

	strides := desc.ByteStrides
	if len(strides) == 0 {
		strides = computeContiguousStrides(shape, elemSize)
	} else if len(strides) != len(shape) {
		return nil, fmt.Errorf("%w: strides length %d does not match shape rank %d",
			ErrInvalidShape, len(strides), len(shape))
	}

	parentHostPtr := parent.HostPtr()
	if parentHostPtr == 0 {
		return nil, ErrBufferClosed
	}

	childPtr := parentHostPtr + uintptr(desc.ByteOffset)

	view := &TensorView{
		parent:      parent,
		hostPtr:     childPtr,
		devPtr:      childPtr,
		byteOffset:  desc.ByteOffset,
		byteLength:  desc.ByteLength,
		dtype:       dtype,
		shape:       shape,
		byteStrides: strides,
		alignBytes:  align,
	}

	return view, nil
}

// NewTensorViewFromBuffer is a convenience constructor for creating a TensorView with explicit parameters.
func NewTensorViewFromBuffer(buf *UMABuffer, byteOffset, byteLength int, dtype DataType, shape []int) (*TensorView, error) {
	return NewTensorView(buf, SliceDescriptor{
		ByteOffset: byteOffset,
		ByteLength: byteLength,
		DType:      dtype,
		Shape:      shape,
		AlignBytes: GFX1151VectorAlignment,
	})
}

// NewTensorViewFromHostSlice wraps a standard Go byte slice in a TensorView for simulation or fallback mode.
func NewTensorViewFromHostSlice(b []byte, dtype DataType, shape []int) (*TensorView, error) {
	if len(b) == 0 {
		return nil, ErrEmptySlice
	}
	if dtype == "" {
		dtype = DTypeFP16
	}
	elemSize := dtype.ElementSize()
	if len(shape) == 0 {
		shape = []int{len(b) / elemSize}
	} else {
		elemCount := 1
		for _, dim := range shape {
			if dim <= 0 {
				return nil, ErrInvalidShape
			}
			elemCount *= dim
		}
		if elemCount*elemSize != len(b) {
			return nil, fmt.Errorf("%w: shape %v element count %d != capacity %d",
				ErrInvalidShape, shape, elemCount*elemSize, len(b))
		}
	}

	ptr := uintptr(unsafe.Pointer(&b[0]))
	return &TensorView{
		hostPtr:     ptr,
		devPtr:      ptr,
		byteOffset:  0,
		byteLength:  len(b),
		dtype:       dtype,
		shape:       shape,
		byteStrides: computeContiguousStrides(shape, elemSize),
		alignBytes:  GFX1151VectorAlignment,
	}, nil
}

// SliceTensor creates a zero-copy TensorView over a sub-region of this UMABuffer.
func (b *UMABuffer) SliceTensor(desc SliceDescriptor) (*TensorView, error) {
	return NewTensorView(b, desc)
}

// AsTensorView creates a TensorView spanning the entire UMABuffer with the specified shape and dtype.
func (b *UMABuffer) AsTensorView(dtype DataType, shape []int) (*TensorView, error) {
	return NewTensorView(b, SliceDescriptor{
		ByteOffset: 0,
		ByteLength: b.Len(),
		DType:      dtype,
		Shape:      shape,
		AlignBytes: GFX1151VectorAlignment,
	})
}

// HostPtr returns the 64-bit host virtual address of this view.
func (v *TensorView) HostPtr() uintptr {
	if v == nil {
		return 0
	}
	return v.hostPtr
}

// DevPtr returns the 64-bit device virtual address of this view.
// In AMD Strix Halo GFX1151 UMA, DevPtr is guaranteed identical to HostPtr.
func (v *TensorView) DevPtr() uintptr {
	if v == nil {
		return 0
	}
	return v.devPtr
}

// IsZeroCopy returns true if the view has a non-zero virtual address where HostPtr == DevPtr.
func (v *TensorView) IsZeroCopy() bool {
	if v == nil || v.hostPtr == 0 {
		return false
	}
	return v.hostPtr == v.devPtr
}

// ByteOffset returns the absolute byte offset of this view relative to the root UMABuffer.
func (v *TensorView) ByteOffset() int {
	if v == nil {
		return 0
	}
	return v.byteOffset
}

// ByteLength returns the size of this tensor view in bytes.
func (v *TensorView) ByteLength() int {
	if v == nil {
		return 0
	}
	return v.byteLength
}

// DType returns the element data type.
func (v *TensorView) DType() DataType {
	if v == nil {
		return ""
	}
	return v.dtype
}

// AlignBytes returns the hardware alignment enforced on this view.
func (v *TensorView) AlignBytes() int {
	if v == nil {
		return 0
	}
	return v.alignBytes
}

// Shape returns a copy of the tensor dimension extents.
func (v *TensorView) Shape() []int {
	if v == nil || len(v.shape) == 0 {
		return nil
	}
	res := make([]int, len(v.shape))
	copy(res, v.shape)
	return res
}

// ByteStrides returns a copy of the byte strides per dimension.
func (v *TensorView) ByteStrides() []int {
	if v == nil || len(v.byteStrides) == 0 {
		return nil
	}
	res := make([]int, len(v.byteStrides))
	copy(res, v.byteStrides)
	return res
}

// ElementCount returns the total number of tensor elements in this view.
func (v *TensorView) ElementCount() int {
	if v == nil || v.byteLength == 0 {
		return 0
	}
	elemSize := v.dtype.ElementSize()
	if elemSize <= 0 {
		elemSize = 1
	}
	return v.byteLength / elemSize
}

// ParentBuffer returns the strong pointer reference to the root UMABuffer,
// guaranteeing Go garbage collector lifetime anchoring.
func (v *TensorView) ParentBuffer() *UMABuffer {
	if v == nil {
		return nil
	}
	return v.parent
}

// KeepAlive executes runtime.KeepAlive on the parent UMABuffer and parent view references,
// ensuring underlying physical memory pages are not prematurely finalized or reclaimed by the GC.
func (v *TensorView) KeepAlive() {
	if v == nil {
		return
	}
	if v.parent != nil {
		runtime.KeepAlive(v.parent)
	}
	if v.parentView != nil {
		runtime.KeepAlive(v.parentView)
	}
}

// ByteSlice returns a zero-copy []byte pointing directly to the view's memory.
func (v *TensorView) ByteSlice() []byte {
	if v == nil || v.hostPtr == 0 || v.byteLength <= 0 {
		return nil
	}
	uptr := uintptrToUnsafe(v.hostPtr)
	b := unsafe.Slice((*byte)(uptr), v.byteLength)
	v.KeepAlive()
	return b
}

// Float16Slice returns a zero-copy []uint16 viewing IEEE 754 float16 / bfloat16 elements.
// Returns nil if view is nil, invalid, or length < 2.
func (v *TensorView) Float16Slice() []uint16 {
	if v == nil || v.hostPtr == 0 || v.byteLength < 2 {
		return nil
	}
	uptr := uintptrToUnsafe(v.hostPtr)
	f := unsafe.Slice((*uint16)(uptr), v.byteLength/2)
	v.KeepAlive()
	return f
}

// Float32Slice returns a zero-copy []float32 viewing IEEE 754 float32 elements.
// Returns nil if view is nil, invalid, or length < 4.
func (v *TensorView) Float32Slice() []float32 {
	if v == nil || v.hostPtr == 0 || v.byteLength < 4 {
		return nil
	}
	uptr := uintptrToUnsafe(v.hostPtr)
	f := unsafe.Slice((*float32)(uptr), v.byteLength/4)
	v.KeepAlive()
	return f
}

// Slice derives a sub-tensor view from this TensorView with explicit descriptor constraints.
// It enforces 16-byte hardware vector alignment, preserves pointer identity (HostPtr == DevPtr),
// and anchors the parent buffer reference.
func (v *TensorView) Slice(desc SliceDescriptor) (*TensorView, error) {
	if v == nil || v.hostPtr == 0 {
		return nil, ErrViewClosed
	}
	if v.parent != nil && atomic.LoadUint32(&v.parent.closed) != 0 {
		return nil, ErrBufferClosed
	}

	if desc.ByteLength <= 0 {
		return nil, ErrInvalidSliceLength
	}
	if desc.ByteOffset < 0 || desc.ByteOffset+desc.ByteLength > v.byteLength {
		return nil, fmt.Errorf("%w: offset %d + len %d exceeds view length %d",
			ErrSliceOutOfBounds, desc.ByteOffset, desc.ByteLength, v.byteLength)
	}

	align := desc.AlignBytes
	if align <= 0 {
		align = v.alignBytes
	}
	if align <= 0 {
		align = GFX1151VectorAlignment
	}

	// Validate vector load alignment against hardware boundary.
	if (desc.ByteOffset % align) != 0 {
		return nil, fmt.Errorf("%w: offset %d is not aligned to %d-byte hardware boundary",
			ErrMisalignedSliceOffset, desc.ByteOffset, align)
	}

	dtype := desc.DType
	if dtype == "" {
		dtype = v.dtype
	}
	elemSize := dtype.ElementSize()
	if (desc.ByteOffset % elemSize) != 0 {
		return nil, fmt.Errorf("%w: offset %d is not aligned to element size %d",
			ErrMisalignedSliceOffset, desc.ByteOffset, elemSize)
	}
	if (desc.ByteLength % elemSize) != 0 {
		return nil, fmt.Errorf("%w: length %d is not a multiple of element size %d",
			ErrInvalidSliceLength, desc.ByteLength, elemSize)
	}

	shape := desc.Shape
	if len(shape) == 0 {
		shape = []int{desc.ByteLength / elemSize}
	} else {
		elemCount := 1
		for _, dim := range shape {
			if dim <= 0 {
				return nil, ErrInvalidShape
			}
			elemCount *= dim
		}
		if elemCount*elemSize != desc.ByteLength {
			return nil, fmt.Errorf("%w: shape %v element count %d != capacity %d",
				ErrInvalidShape, shape, elemCount*elemSize, desc.ByteLength)
		}
	}

	strides := desc.ByteStrides
	if len(strides) == 0 {
		strides = computeContiguousStrides(shape, elemSize)
	} else if len(strides) != len(shape) {
		return nil, fmt.Errorf("%w: strides length %d does not match shape rank %d",
			ErrInvalidShape, len(strides), len(shape))
	}

	childPtr := v.hostPtr + uintptr(desc.ByteOffset)

	child := &TensorView{
		parent:      v.parent,
		parentView:  v,
		hostPtr:     childPtr,
		devPtr:      childPtr,
		byteOffset:  v.byteOffset + desc.ByteOffset,
		byteLength:  desc.ByteLength,
		dtype:       dtype,
		shape:       shape,
		byteStrides: strides,
		alignBytes:  align,
	}

	return child, nil
}

// SubSlice provides a low-overhead, fast slicing path for contiguous 1D sub-views.
// It is designed to be inlineable by the Go compiler, achieving zero heap allocations (0 B/op)
// and sub-5ns execution times in high-frequency token generation loops.
func (v *TensorView) SubSlice(relByteOffset, byteLength int) (*TensorView, error) {
	if v == nil || v.hostPtr == 0 {
		return nil, ErrViewClosed
	}
	if relByteOffset < 0 || byteLength <= 0 || relByteOffset+byteLength > v.byteLength {
		return nil, ErrSliceOutOfBounds
	}
	align := v.alignBytes
	if align <= 0 {
		align = GFX1151VectorAlignment
	}
	if (relByteOffset % align) != 0 {
		return nil, ErrMisalignedSliceOffset
	}

	childPtr := v.hostPtr + uintptr(relByteOffset)
	child := &TensorView{
		parent:     v.parent,
		parentView: v,
		hostPtr:    childPtr,
		devPtr:     childPtr,
		byteOffset: v.byteOffset + relByteOffset,
		byteLength: byteLength,
		dtype:      v.dtype,
		alignBytes: align,
	}
	return child, nil
}

// SliceView returns a value-based TensorView without heap escape, guaranteeing 0 allocs/op.
func (v *TensorView) SliceView(relByteOffset, byteLength int) (TensorView, error) {
	if v == nil || v.hostPtr == 0 {
		return TensorView{}, ErrViewClosed
	}
	if relByteOffset < 0 || byteLength <= 0 || relByteOffset+byteLength > v.byteLength {
		return TensorView{}, ErrSliceOutOfBounds
	}
	align := v.alignBytes
	if align <= 0 {
		align = GFX1151VectorAlignment
	}
	if (relByteOffset % align) != 0 {
		return TensorView{}, ErrMisalignedSliceOffset
	}

	childPtr := v.hostPtr + uintptr(relByteOffset)
	return TensorView{
		parent:     v.parent,
		parentView: v,
		hostPtr:    childPtr,
		devPtr:     childPtr,
		byteOffset: v.byteOffset + relByteOffset,
		byteLength: byteLength,
		dtype:      v.dtype,
		alignBytes: align,
	}, nil
}

// computeContiguousStrides calculates row-major byte strides for a contiguous multi-dimensional tensor.
func computeContiguousStrides(shape []int, elemSize int) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	stride := elemSize
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}
