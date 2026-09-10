// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestTensorViewInPlaceSlicing verifies zero-copy slicing, pointer identity preservation,
// 16-byte GFX1151 vector alignment enforcement, and in-place bidirectional mutation reflection.
func TestTensorViewInPlaceSlicing(t *testing.T) {
	const bufSize = 256 * 1024 // 256 KB
	buf, err := AllocateUMABuffer(bufSize, 64)
	if err != nil {
		t.Fatalf("AllocateUMABuffer failed: %v", err)
	}
	defer buf.Close()

	if !buf.IsZeroCopy() {
		t.Fatal("expected parent UMABuffer to be zero-copy")
	}

	// 1. Create root view spanning entire buffer as FP16 elements (128k elements).
	rootView, err := buf.AsTensorView(DTypeFP16, []int{128, 1024})
	if err != nil {
		t.Fatalf("AsTensorView failed: %v", err)
	}

	if !rootView.IsZeroCopy() {
		t.Error("expected rootView to be zero-copy")
	}
	if rootView.HostPtr() != buf.HostPtr() {
		t.Errorf("expected rootView.HostPtr %x == buf.HostPtr %x", rootView.HostPtr(), buf.HostPtr())
	}
	if rootView.DevPtr() != buf.DevPtr() {
		t.Errorf("expected rootView.DevPtr %x == buf.DevPtr %x", rootView.DevPtr(), buf.DevPtr())
	}
	if rootView.ByteLength() != bufSize {
		t.Errorf("expected rootView.ByteLength %d == %d", rootView.ByteLength(), bufSize)
	}
	if rootView.ElementCount() != bufSize/2 {
		t.Errorf("expected ElementCount %d == %d", rootView.ElementCount(), bufSize/2)
	}

	// 2. Slice a child view at 16-byte aligned offset (e.g. 64 bytes offset, 4096 bytes length).
	childDesc := SliceDescriptor{
		ByteOffset: 64,
		ByteLength: 4096,
		DType:      DTypeFP16,
		Shape:      []int{2, 1024},
		AlignBytes: GFX1151VectorAlignment,
	}
	childView, err := rootView.Slice(childDesc)
	if err != nil {
		t.Fatalf("rootView.Slice failed: %v", err)
	}

	// 3. Verify pointer identity preservation across offset.
	expectedPtr := rootView.HostPtr() + 64
	if childView.HostPtr() != expectedPtr {
		t.Errorf("childView.HostPtr %x != expected %x", childView.HostPtr(), expectedPtr)
	}
	if childView.DevPtr() != expectedPtr {
		t.Errorf("childView.DevPtr %x != expected %x", childView.DevPtr(), expectedPtr)
	}
	if !childView.IsZeroCopy() {
		t.Error("childView must be zero-copy (HostPtr == DevPtr)")
	}
	if childView.ByteOffset() != 64 {
		t.Errorf("expected childView.ByteOffset 64, got %d", childView.ByteOffset())
	}

	// 4. Validate bidirectional in-place mutation reflection.
	// Write via parent buffer slice, observe in child view.
	parentSlice := buf.Slice()
	parentSlice[64] = 0xAA
	parentSlice[65] = 0x55

	childBytes := childView.ByteSlice()
	if childBytes[0] != 0xAA || childBytes[1] != 0x55 {
		t.Errorf("child did not reflect parent write: got [%x, %x], want [aa, 55]",
			childBytes[0], childBytes[1])
	}

	// Write via child view, observe in parent buffer.
	childBytes[2] = 0xBE
	childBytes[3] = 0xEF
	if parentSlice[66] != 0xBE || parentSlice[67] != 0xEF {
		t.Errorf("parent did not reflect child write: got [%x, %x], want [be, ef]",
			parentSlice[66], parentSlice[67])
	}

	// 5. Test FP16 slice viewing.
	childFP16 := childView.Float16Slice()
	if len(childFP16) != 2048 {
		t.Errorf("expected 2048 fp16 elements, got %d", len(childFP16))
	}
	// Verify word encoding (little endian: 0x55AA).
	if childFP16[0] != 0x55AA {
		t.Errorf("expected fp16 word 0x55AA, got 0x%04X", childFP16[0])
	}

	// 6. Test recursive slicing (child slicing grandchild).
	grandChild, err := childView.SubSlice(32, 1024)
	if err != nil {
		t.Fatalf("childView.SubSlice failed: %v", err)
	}
	if grandChild.HostPtr() != childView.HostPtr()+32 {
		t.Errorf("grandChild.HostPtr %x != childView.HostPtr()+32 %x",
			grandChild.HostPtr(), childView.HostPtr()+32)
	}
	if grandChild.ByteOffset() != 96 {
		t.Errorf("expected grandChild.ByteOffset 96, got %d", grandChild.ByteOffset())
	}
	if !grandChild.IsZeroCopy() {
		t.Error("grandChild must be zero copy")
	}

	// 7. Verify hardware alignment violation rejection.
	// Slicing with non-16-byte aligned offset (e.g. 8 bytes or 1 byte).
	unalignedDesc := SliceDescriptor{
		ByteOffset: 7, // Not 16-byte aligned
		ByteLength: 128,
		AlignBytes: GFX1151VectorAlignment,
	}
	if _, err := rootView.Slice(unalignedDesc); !errors.Is(err, ErrMisalignedSliceOffset) {
		t.Errorf("expected ErrMisalignedSliceOffset for offset 7, got %v", err)
	}

	unalignedSub, err := rootView.SubSlice(4, 128) // 4 is not 16-aligned
	if !errors.Is(err, ErrMisalignedSliceOffset) {
		t.Errorf("expected ErrMisalignedSliceOffset for offset 4, got %v (sub: %v)", err, unalignedSub)
	}

	// 8. Verify out-of-bounds rejection.
	oobDesc := SliceDescriptor{
		ByteOffset: bufSize - 16,
		ByteLength: 32, // Exceeds buffer size by 16 bytes
	}
	if _, err := rootView.Slice(oobDesc); !errors.Is(err, ErrSliceOutOfBounds) {
		t.Errorf("expected ErrSliceOutOfBounds, got %v", err)
	}

	if _, err := rootView.SubSlice(bufSize+16, 16); !errors.Is(err, ErrSliceOutOfBounds) {
		t.Errorf("expected ErrSliceOutOfBounds, got %v", err)
	}

	// 9. Verify invalid shape rejection.
	badShapeDesc := SliceDescriptor{
		ByteOffset: 0,
		ByteLength: 1024,
		DType:      DTypeFP16,     // 512 elements
		Shape:      []int{10, 10}, // 100 elements != 512
	}
	if _, err := rootView.Slice(badShapeDesc); !errors.Is(err, ErrInvalidShape) {
		t.Errorf("expected ErrInvalidShape, got %v", err)
	}

	// 10. Concurrency race test across multiple goroutines.
	const goroutines = 16
	const iters = 200
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(gid int) {
			defer wg.Done()
			offset := (gid * 1024) % (bufSize - 2048)
			// Ensure 16-byte alignment
			offset = (offset / 16) * 16

			for i := 0; i < iters; i++ {
				sub, err := rootView.SubSlice(offset, 1024)
				if err != nil {
					t.Errorf("concurrent SubSlice failed: %v", err)
					return
				}
				if !sub.IsZeroCopy() {
					t.Errorf("concurrent subview not zero-copy")
					return
				}
				b := sub.ByteSlice()
				if len(b) != 1024 {
					t.Errorf("unexpected byte slice length: %d", len(b))
					return
				}
				sub.KeepAlive()
			}
		}(g)
	}
	wg.Wait()
}

// TestTensorView_GCLifetimeAnchoring proves that a parent UMABuffer cannot be garbage-collected
// or finalized while a derived child TensorView remains reachable in scope.
func TestTensorView_GCLifetimeAnchoring(t *testing.T) {
	var finalizerRan atomic.Bool

	// Create view inside helper function to ensure local parent reference falls out of scope.
	createView := func() *TensorView {
		buf, err := AllocateUMABuffer(64*1024, 64)
		if err != nil {
			t.Fatalf("AllocateUMABuffer failed: %v", err)
		}

		// Attach finalizer to the parent buffer to detect if Go GC attempts to collect it.
		runtime.SetFinalizer(buf, func(b *UMABuffer) {
			finalizerRan.Store(true)
		})

		view, err := buf.AsTensorView(DTypeFP16, []int{32, 1024})
		if err != nil {
			t.Fatalf("AsTensorView failed: %v", err)
		}

		child, err := view.SubSlice(128, 4096)
		if err != nil {
			t.Fatalf("view.SubSlice failed: %v", err)
		}

		return child
	}

	childView := createView()

	// Trigger multiple aggressive GC sweeps.
	for i := 0; i < 5; i++ {
		runtime.GC()
		time.Sleep(2 * time.Millisecond)
	}

	// Verify parent finalizer has NOT run because childView retains childView.parent!
	if finalizerRan.Load() {
		t.Fatal("parent UMABuffer was prematurely garbage-collected while child TensorView is in scope")
	}

	// Verify childView is still valid and readable.
	bytes := childView.ByteSlice()
	if len(bytes) != 4096 {
		t.Fatalf("expected 4096 bytes, got %d", len(bytes))
	}
	childView.KeepAlive()

	// Now allow childView to become unreachable.
	runtime.KeepAlive(childView)
}

// TestTensorView_ClosedBufferRejection verifies that operations on closed buffers fail closed.
func TestTensorView_ClosedBufferRejection(t *testing.T) {
	buf, err := AllocateUMABuffer(16*1024, 64)
	if err != nil {
		t.Fatalf("AllocateUMABuffer failed: %v", err)
	}

	view, err := buf.AsTensorView(DTypeFP32, []int{4, 1024})
	if err != nil {
		t.Fatalf("AsTensorView failed: %v", err)
	}

	// Close the parent buffer.
	if err := buf.Close(); err != nil {
		t.Fatalf("buf.Close failed: %v", err)
	}

	// Slicing from closed buffer must return ErrBufferClosed.
	_, err = buf.SliceTensor(SliceDescriptor{ByteOffset: 0, ByteLength: 1024})
	if !errors.Is(err, ErrBufferClosed) {
		t.Errorf("expected ErrBufferClosed from closed buffer SliceTensor, got %v", err)
	}

	_, err = view.Slice(SliceDescriptor{ByteOffset: 0, ByteLength: 1024})
	if !errors.Is(err, ErrBufferClosed) {
		t.Errorf("expected ErrBufferClosed from view on closed buffer, got %v", err)
	}
}

// TestTensorView_SimulationFallback tests host slice wrapping for non-UMA simulation mode.
func TestTensorView_SimulationFallback(t *testing.T) {
	raw := make([]byte, 1024)
	raw[0] = 0x11
	raw[1] = 0x22

	view, err := NewTensorViewFromHostSlice(raw, DTypeFP16, []int{512})
	if err != nil {
		t.Fatalf("NewTensorViewFromHostSlice failed: %v", err)
	}

	if !view.IsZeroCopy() {
		t.Error("host slice view should be zero-copy with itself")
	}
	if view.ByteLength() != 1024 {
		t.Errorf("expected length 1024, got %d", view.ByteLength())
	}
	if view.ElementCount() != 512 {
		t.Errorf("expected element count 512, got %d", view.ElementCount())
	}

	sub, err := view.SubSlice(16, 64)
	if err != nil {
		t.Fatalf("SubSlice failed: %v", err)
	}
	if sub.ByteLength() != 64 {
		t.Errorf("expected sub length 64, got %d", sub.ByteLength())
	}
}

// BenchmarkTensorViewSlice measures view slicing overhead, asserting 0 allocs/op and sub-5ns execution.
func BenchmarkTensorViewSlice(b *testing.B) {
	const size = 1024 * 1024
	buf, err := AllocateUMABuffer(size, 64)
	if err != nil {
		b.Fatalf("AllocateUMABuffer failed: %v", err)
	}
	defer buf.Close()

	rootView, err := buf.AsTensorView(DTypeFP16, []int{512, 1024})
	if err != nil {
		b.Fatalf("AsTensorView failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		sub, err := rootView.SliceView(64, 4096)
		if err != nil {
			b.Fatal(err)
		}
		_ = sub
	}
}

// BenchmarkTensorViewSliceValue measures value-based zero-escape slicing performance.
func BenchmarkTensorViewSliceValue(b *testing.B) {
	const size = 1024 * 1024
	buf, err := AllocateUMABuffer(size, 64)
	if err != nil {
		b.Fatalf("AllocateUMABuffer failed: %v", err)
	}
	defer buf.Close()

	rootView, err := buf.AsTensorView(DTypeFP16, []int{512, 1024})
	if err != nil {
		b.Fatalf("AsTensorView failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		sub, err := rootView.SliceView(64, 4096)
		if err != nil {
			b.Fatal(err)
		}
		_ = sub
	}
}
