package strix

import (
	"encoding/binary"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

func TestUMABuffer_AlignedAllocation(t *testing.T) {
	testSizes := []int{1, 16, 63, 64, 65, 127, 128, 512, 1024, 4096, 65536}

	for _, size := range testSizes {
		buf, err := NewUMABuffer(size)
		if err != nil {
			t.Fatalf("NewUMABuffer(%d) unexpected error: %v", size, err)
		}

		ptr := buf.UnsafePointer()
		if ptr == nil {
			t.Fatalf("NewUMABuffer(%d) returned nil UnsafePointer", size)
		}

		addr := uintptr(ptr)
		if addr%UMACacheLineAlignment != 0 {
			t.Errorf("NewUMABuffer(%d) address 0x%x is not %d-byte aligned (mod %d = %d)",
				size, addr, UMACacheLineAlignment, UMACacheLineAlignment, addr%UMACacheLineAlignment)
		}

		if buf.Len() != size {
			t.Errorf("NewUMABuffer(%d) Len() = %d, expected %d", size, buf.Len(), size)
		}

		slice := buf.Slice()
		if len(slice) != size {
			t.Errorf("NewUMABuffer(%d) len(Slice()) = %d, expected %d", size, len(slice), size)
		}
		if cap(slice) != size {
			t.Errorf("NewUMABuffer(%d) cap(Slice()) = %d, expected %d", size, cap(slice), size)
		}

		sliceAddr := uintptr(unsafe.Pointer(&slice[0]))
		if sliceAddr != addr {
			t.Errorf("NewUMABuffer(%d) slice base 0x%x != UnsafePointer 0x%x", size, sliceAddr, addr)
		}

		if err := buf.Close(); err != nil {
			t.Errorf("Close() error: %v", err)
		}
	}
}

func TestUMABuffer_InvalidSizes(t *testing.T) {
	invalidSizes := []int{0, -1, -64, -4096}

	for _, size := range invalidSizes {
		buf, err := NewUMABuffer(size)
		if !errors.Is(err, ErrInvalidBufferSize) {
			t.Errorf("NewUMABuffer(%d) expected ErrInvalidBufferSize, got buf=%v, err=%v", size, buf, err)
		}
	}
}

func TestUMABuffer_ZeroCopyPointerAccess(t *testing.T) {
	buf, err := NewUMABuffer(256)
	if err != nil {
		t.Fatalf("failed to allocate buffer: %v", err)
	}
	defer buf.Close()

	if bc := buf.BytesCopied(); bc != 0 {
		t.Fatalf("expected BytesCopied() == 0, got %d", bc)
	}

	slice := buf.Slice()
	ptr := buf.UnsafePointer()

	// 1. Write via slice, read directly via unsafe.Pointer
	for i := 0; i < len(slice); i++ {
		slice[i] = byte(i & 0xFF)
	}

	bytePtr := (*[256]byte)(ptr)
	for i := 0; i < 256; i++ {
		if bytePtr[i] != byte(i&0xFF) {
			t.Fatalf("mismatch at byte %d via unsafe pointer: expected 0x%x, got 0x%x", i, byte(i&0xFF), bytePtr[i])
		}
	}

	// 2. Mutate via unsafe.Pointer, verify visible in slice immediately
	for i := 0; i < 256; i++ {
		bytePtr[i] = byte((255 - i) & 0xFF)
	}

	for i := 0; i < len(slice); i++ {
		if slice[i] != byte((255-i)&0xFF) {
			t.Fatalf("mismatch at byte %d in slice after pointer mutation: expected 0x%x, got 0x%x", i, byte((255-i)&0xFF), slice[i])
		}
	}

	// 3. Confirm BytesCopied invariant remains strictly 0
	if bc := buf.BytesCopied(); bc != 0 {
		t.Fatalf("zero-copy invariant violated: BytesCopied() = %d", bc)
	}
}

func TestUMABuffer_MemoryFenceReleaseAcquire(t *testing.T) {
	buf, err := NewUMABuffer(1024)
	if err != nil {
		t.Fatalf("failed to allocate buffer: %v", err)
	}
	defer buf.Close()

	// Simulate concurrent producer (Context-MMU) and consumer (RDNA 3.5 forward pass)
	var wg sync.WaitGroup
	var ack uint64
	const iterations = 500

	wg.Add(2)

	// Producer: writes sequential values, issues release fence
	go func() {
		defer wg.Done()
		slice := buf.Slice()
		for iter := uint64(1); iter <= iterations; iter++ {
			for atomic.LoadUint64(&ack) < iter-1 {
				runtime.Gosched()
			}
			binary.LittleEndian.PutUint32(slice[0:4], uint32(iter))
			buf.MemoryFenceRelease()
		}
	}()

	// Consumer: issues acquire fence, reads values
	go func() {
		defer wg.Done()
		slice := buf.Slice()
		for expected := uint64(1); expected <= iterations; expected++ {
			for {
				if buf.MemoryFenceAcquire() >= expected {
					break
				}
				runtime.Gosched()
			}
			val := binary.LittleEndian.Uint32(slice[0:4])
			if val != uint32(expected) {
				t.Errorf("expected %d, got %d", expected, val)
			}
			atomic.StoreUint64(&ack, expected)
		}
	}()

	wg.Wait()
}

func TestUMABuffer_InspectTokensInPlace(t *testing.T) {
	tokenCount := 64
	sizeBytes := tokenCount * 4
	buf, err := NewUMABuffer(sizeBytes)
	if err != nil {
		t.Fatalf("failed to allocate buffer: %v", err)
	}
	defer buf.Close()

	expectedTokens := make([]uint32, tokenCount)
	for i := 0; i < tokenCount; i++ {
		expectedTokens[i] = uint32(1000 + i*7)
		binary.LittleEndian.PutUint32(buf.Slice()[i*4:(i+1)*4], expectedTokens[i])
	}

	// 1. Inspect all tokens from offset 0
	tokens, err := buf.InspectTokensInPlace(0, tokenCount)
	if err != nil {
		t.Fatalf("InspectTokensInPlace failed: %v", err)
	}
	if len(tokens) != tokenCount {
		t.Fatalf("expected %d tokens, got %d", tokenCount, len(tokens))
	}

	for i := 0; i < tokenCount; i++ {
		if tokens[i] != expectedTokens[i] {
			t.Errorf("token %d mismatch: expected %d, got %d", i, expectedTokens[i], tokens[i])
		}
	}

	// 2. Verify pointer identity: tokens slice points directly to buffer
	tokensAddr := uintptr(unsafe.Pointer(&tokens[0]))
	bufAddr := uintptr(buf.UnsafePointer())
	if tokensAddr != bufAddr {
		t.Fatalf("tokens slice address 0x%x != buffer address 0x%x (copy detected)", tokensAddr, bufAddr)
	}

	// 3. Inspect with offset
	offset := 16 // 4 tokens in
	count := 8
	subTokens, err := buf.InspectTokensInPlace(offset, count)
	if err != nil {
		t.Fatalf("InspectTokensInPlace with offset failed: %v", err)
	}
	if len(subTokens) != count {
		t.Fatalf("expected %d subTokens, got %d", count, len(subTokens))
	}
	for i := 0; i < count; i++ {
		if subTokens[i] != expectedTokens[4+i] {
			t.Errorf("subToken %d mismatch: expected %d, got %d", i, expectedTokens[4+i], subTokens[i])
		}
	}

	expectedSubAddr := bufAddr + uintptr(offset)
	subAddr := uintptr(unsafe.Pointer(&subTokens[0]))
	if subAddr != expectedSubAddr {
		t.Fatalf("subTokens slice address 0x%x != expected 0x%x", subAddr, expectedSubAddr)
	}

	// 4. In-place mutation check: write to token slice, check buffer
	subTokens[0] = 999999
	updatedVal := binary.LittleEndian.Uint32(buf.Slice()[offset : offset+4])
	if updatedVal != 999999 {
		t.Fatalf("mutation in tokens slice not reflected in buffer: got %d, expected 999999", updatedVal)
	}

	// 5. Zero-copy proof
	if bc := buf.BytesCopied(); bc != 0 {
		t.Fatalf("expected 0 bytes copied, got %d", bc)
	}
}

func TestUMABuffer_InspectTokensInPlace_ZeroAlloc(t *testing.T) {
	buf, err := NewUMABuffer(256)
	if err != nil {
		t.Fatalf("failed to allocate buffer: %v", err)
	}
	defer buf.Close()

	allocs := testing.AllocsPerRun(100, func() {
		toks, err := buf.InspectTokensInPlace(0, 16)
		if err != nil || len(toks) != 16 {
			t.Fail()
		}
	})

	if allocs > 0 {
		t.Errorf("InspectTokensInPlace incurred %f heap allocations, expected 0 (zero-copy)", allocs)
	}
}

func TestUMABuffer_InspectTokensInPlace_EdgeCases(t *testing.T) {
	buf, err := NewUMABuffer(64)
	if err != nil {
		t.Fatalf("failed to allocate buffer: %v", err)
	}
	defer buf.Close()

	// 1. Negative offset
	if _, err := buf.InspectTokensInPlace(-4, 4); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("expected ErrOutOfBounds for negative offset, got %v", err)
	}

	// 2. Negative count
	if _, err := buf.InspectTokensInPlace(0, -1); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("expected ErrOutOfBounds for negative count, got %v", err)
	}

	// 3. Misaligned offset
	for _, misaligned := range []int{1, 2, 3, 5, 7, 9} {
		if _, err := buf.InspectTokensInPlace(misaligned, 2); !errors.Is(err, ErrMisalignedOffset) {
			t.Errorf("expected ErrMisalignedOffset for offset %d, got %v", misaligned, err)
		}
	}

	// 4. Out of bounds (count * 4 exceeds buffer size)
	if _, err := buf.InspectTokensInPlace(0, 17); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("expected ErrOutOfBounds when exceeding size, got %v", err)
	}
	if _, err := buf.InspectTokensInPlace(60, 2); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("expected ErrOutOfBounds when offset + count*4 exceeds size, got %v", err)
	}

	// 5. Zero count
	zeroToks, err := buf.InspectTokensInPlace(0, 0)
	if err != nil {
		t.Errorf("unexpected error for count=0: %v", err)
	}
	if len(zeroToks) != 0 {
		t.Errorf("expected empty slice for count=0, got len=%d", len(zeroToks))
	}

	// 6. Zero count with offset at boundary
	boundaryToks, err := buf.InspectTokensInPlace(64, 0)
	if err != nil {
		t.Errorf("unexpected error for count=0 at boundary: %v", err)
	}
	if len(boundaryToks) != 0 {
		t.Errorf("expected empty slice, got len=%d", len(boundaryToks))
	}

	// 7. Zero count with offset beyond boundary
	if _, err := buf.InspectTokensInPlace(68, 0); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("expected ErrOutOfBounds for offset past boundary with count=0, got %v", err)
	}
}

func TestUMABuffer_CloseAndNilSafety(t *testing.T) {
	buf, err := NewUMABuffer(128)
	if err != nil {
		t.Fatalf("failed to allocate buffer: %v", err)
	}

	if err := buf.Close(); err != nil {
		t.Fatalf("first Close() failed: %v", err)
	}

	// Idempotent Close
	if err := buf.Close(); err != nil {
		t.Fatalf("second Close() failed: %v", err)
	}

	// Post-close method behaviors
	if ptr := buf.UnsafePointer(); ptr != nil {
		t.Errorf("expected nil UnsafePointer after close, got %v", ptr)
	}
	if slice := buf.Slice(); slice != nil {
		t.Errorf("expected nil Slice after close, got %v", slice)
	}
	if length := buf.Len(); length != 0 {
		t.Errorf("expected Len() == 0 after close, got %d", length)
	}
	if bc := buf.BytesCopied(); bc != 0 {
		t.Errorf("expected BytesCopied() == 0 after close, got %d", bc)
	}
	if _, err := buf.InspectTokensInPlace(0, 4); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("expected ErrBufferClosed after close, got %v", err)
	}

	// Fences on closed buffer do not panic
	buf.MemoryFenceRelease()
	buf.MemoryFenceAcquire()

	// Nil receiver safety
	var nilBuf *UMABuffer
	if ptr := nilBuf.UnsafePointer(); ptr != nil {
		t.Errorf("expected nil UnsafePointer for nil buffer, got %v", ptr)
	}
	if slice := nilBuf.Slice(); slice != nil {
		t.Errorf("expected nil Slice for nil buffer, got %v", slice)
	}
	if length := nilBuf.Len(); length != 0 {
		t.Errorf("expected Len() == 0 for nil buffer, got %d", length)
	}
	if bc := nilBuf.BytesCopied(); bc != 0 {
		t.Errorf("expected BytesCopied() == 0 for nil buffer, got %d", bc)
	}
	if _, err := nilBuf.InspectTokensInPlace(0, 4); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("expected ErrBufferClosed for nil buffer, got %v", err)
	}
	if err := nilBuf.Close(); err != nil {
		t.Errorf("expected nil error on nil buffer Close(), got %v", err)
	}
	nilBuf.MemoryFenceRelease()
	nilBuf.MemoryFenceAcquire()
}
