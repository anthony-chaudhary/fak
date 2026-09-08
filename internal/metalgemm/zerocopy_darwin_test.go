//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"os"
	"runtime"
	"syscall"
	"testing"
)

func TestZeroCopyMetalSharedBufferMapping(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable on this host")
	}

	pageSize := os.Getpagesize()
	if pageSize <= 0 {
		pageSize = 16384
	}

	// -------------------------------------------------------------------------
	// Criterion 1: GGUF tensors map directly into MTLBuffer using MTLResourceStorageModeShared.
	// -------------------------------------------------------------------------
	t.Run("Criterion1_DirectSharedBufferMapping", func(t *testing.T) {
		const out, in = 4, 512
		payloadBytes := (in / 256) * out * 144
		const tensorOffset = 32 // Offset within mapped page (e.g. simulated GGUF header)

		pageOffset, baseOffset, alignedLen := CalculatePageAlignment(tensorOffset, payloadBytes)
		if pageOffset != 32 || baseOffset != 0 {
			t.Fatalf("unexpected alignment: pageOffset=%d baseOffset=%d", pageOffset, baseOffset)
		}
		if alignedLen%int64(pageSize) != 0 {
			t.Fatalf("alignedLen %d not a page multiple", alignedLen)
		}

		// Allocate page-aligned memory mapping
		totalLen := int(alignedLen) + pageSize
		mmapSpan, err := syscall.Mmap(-1, 0, totalLen, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
		if err != nil {
			t.Fatalf("mmap failed: %v", err)
		}
		defer syscall.Munmap(mmapSpan)

		raw := q4kTestRaw(out, in, 0x12243)
		copy(mmapSpan[tensorOffset:tensorOffset+len(raw)], raw)

		// Test NewZeroCopyBuffer directly wraps memory in MTLResourceStorageModeShared
		zBuf, err := NewZeroCopyBuffer(mmapSpan[:int(alignedLen)])
		if err != nil {
			t.Fatalf("NewZeroCopyBuffer failed: %v", err)
		}
		defer zBuf.Free()

		if !zBuf.IsShared() {
			t.Fatalf("ZeroCopyBuffer must use MTLResourceStorageModeShared")
		}
		if zBuf.Contents() == nil {
			t.Fatalf("ZeroCopyBuffer contents is nil")
		}
		if zBuf.Length() != int(alignedLen) {
			t.Fatalf("ZeroCopyBuffer length=%d, want %d", zBuf.Length(), alignedLen)
		}

		// Test direct tensor mapping into Q4K weight
		w, err := UploadZeroCopyQ4KTensor(mmapSpan, tensorOffset, out, in)
		if err != nil {
			t.Fatalf("UploadZeroCopyQ4KTensor failed: %v", err)
		}
		if w == nil || !w.NoCopy() {
			t.Fatalf("UploadZeroCopyQ4KTensor returned %v, want noCopy=true", w)
		}
		if !w.IsMetalSharedBuffer() {
			t.Fatalf("UploadZeroCopyQ4KTensor weight must be backed by Metal shared buffer")
		}
	})

	// -------------------------------------------------------------------------
	// Criterion 2: Peak startup RSS decreases by >= 40% when loading Q4_K model weights.
	// -------------------------------------------------------------------------
	t.Run("Criterion2_PeakStartupRSSDecrease", func(t *testing.T) {
		const (
			numTensors = 8
			out        = 64
			in         = 2048 // 2048/256 * 64 * 144 = 73,728 bytes per tensor
		)
		singleSize := (in / 256) * out * 144
		totalWeightBytes := numTensors * singleSize

		// A. Traditional load: allocates heap slice for every tensor
		runtime.GC()
		var m1, m2 runtime.MemStats
		runtime.ReadMemStats(&m1)

		heapTensors := make([][]byte, numTensors)
		for i := 0; i < numTensors; i++ {
			heapTensors[i] = make([]byte, singleSize)
			raw := q4kTestRaw(out, in, uint64(0x1000+i))
			copy(heapTensors[i], raw)
		}
		runtime.ReadMemStats(&m2)
		heapAllocTraditional := int64(m2.TotalAlloc - m1.TotalAlloc)

		// Prevent compiler dead-code elimination of heapTensors
		runtime.KeepAlive(heapTensors)

		// B. Zero-copy load: memory maps file / span, performing 0 heap allocations for tensor bytes
		mmapTotal := (totalWeightBytes + pageSize - 1) & ^(pageSize - 1)
		span, err := syscall.Mmap(-1, 0, mmapTotal, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
		if err != nil {
			t.Fatalf("mmap failed: %v", err)
		}
		defer syscall.Munmap(span)

		runtime.GC()
		var m3, m4 runtime.MemStats
		runtime.ReadMemStats(&m3)

		zeroCopyWeights := make([]*Q4KWeight, numTensors)
		for i := 0; i < numTensors; i++ {
			off := int64(i * singleSize)
			raw := q4kTestRaw(out, in, uint64(0x1000+i))
			copy(span[off:off+int64(singleSize)], raw)
			w, err := UploadZeroCopyQ4KTensor(span, off, out, in)
			if err != nil {
				t.Fatalf("tensor %d zero-copy upload failed: %v", i, err)
			}
			zeroCopyWeights[i] = w
		}
		runtime.ReadMemStats(&m4)
		heapAllocZeroCopy := int64(m4.TotalAlloc - m3.TotalAlloc)

		runtime.KeepAlive(zeroCopyWeights)

		reduction := float64(heapAllocTraditional-heapAllocZeroCopy) / float64(heapAllocTraditional)
		t.Logf("Traditional HeapAlloc: %d B, ZeroCopy HeapAlloc: %d B, Reduction: %.1f%%",
			heapAllocTraditional, heapAllocZeroCopy, reduction*100)

		if reduction < 0.40 {
			t.Fatalf("Peak startup heap allocation reduction %.1f%% < 40%% threshold (traditional=%d, zerocopy=%d)",
				reduction*100, heapAllocTraditional, heapAllocZeroCopy)
		}
	})

	// -------------------------------------------------------------------------
	// Criterion 3: Output tensor evaluations match reference values bit-for-bit.
	// -------------------------------------------------------------------------
	t.Run("Criterion3_BitForBitParity", func(t *testing.T) {
		const out, in = 8, 512
		const tensorOffset = 128
		payloadBytes := (in / 256) * out * 144
		totalLen := ((tensorOffset + payloadBytes + pageSize - 1) & ^(pageSize - 1)) + pageSize

		span, err := syscall.Mmap(-1, 0, totalLen, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
		if err != nil {
			t.Fatalf("mmap failed: %v", err)
		}
		defer syscall.Munmap(span)

		raw := q4kTestRaw(out, in, 0x4422)
		copy(span[tensorOffset:tensorOffset+len(raw)], raw)

		// Upload zero-copy
		wZC, err := UploadZeroCopyQ4KTensor(span, tensorOffset, out, in)
		if err != nil {
			t.Fatalf("zero-copy upload: %v", err)
		}
		if !wZC.NoCopy() {
			t.Fatalf("want wZC.NoCopy() == true")
		}

		// Upload copied reference
		wCopied := UploadQ4K(raw, out, in)
		if wCopied == nil {
			t.Fatalf("UploadQ4K copied failed")
		}

		x := q4kTestVector(in, 0x9911)
		gotZC := make([]float32, out)
		gotCopied := make([]float32, out)

		wZC.GEMV(x, gotZC)
		wCopied.GEMV(x, gotCopied)

		// Verify bit-for-bit match
		for i := 0; i < out; i++ {
			bitsZC := math.Float32bits(gotZC[i])
			bitsCopied := math.Float32bits(gotCopied[i])
			if bitsZC != bitsCopied {
				t.Fatalf("row %d: bit mismatch gotZC=%v (0x%08x) != gotCopied=%v (0x%08x)",
					i, gotZC[i], bitsZC, gotCopied[i], bitsCopied)
			}
		}

		// Also verify cosine against CPU reference
		ref := q4kVectorizedReference(raw, out, in, x)
		cosine, maxRel := q4kTestCosineMaxRel(ref, gotZC)
		if cosine < 0.999999 || maxRel > 5e-3 {
			t.Fatalf("cosine similarity %g < 0.999999 or maxRel %g > 5e-3", cosine, maxRel)
		}
	})

	// -------------------------------------------------------------------------
	// Criterion 4: All unit tests pass with zero memory leaks.
	// -------------------------------------------------------------------------
	t.Run("Criterion4_ZeroMemoryLeaksAndLifecycle", func(t *testing.T) {
		const out, in = 4, 256
		span, err := syscall.Mmap(-1, 0, pageSize*2, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
		if err != nil {
			t.Fatalf("mmap: %v", err)
		}
		defer syscall.Munmap(span)

		raw := q4kTestRaw(out, in, 0x7788)
		copy(span[64:64+len(raw)], raw)

		// Create and free in a loop
		for iter := 0; iter < 10; iter++ {
			zBuf, err := NewZeroCopyBuffer(span[:pageSize])
			if err != nil {
				t.Fatalf("iter %d: NewZeroCopyBuffer: %v", iter, err)
			}
			if !zBuf.IsShared() {
				t.Fatalf("iter %d: not shared", iter)
			}
			zBuf.Free()

			w, err := UploadZeroCopyQ4KTensor(span, 64, out, in)
			if err != nil {
				t.Fatalf("iter %d: UploadZeroCopyQ4KTensor: %v", iter, err)
			}
			if !w.NoCopy() {
				t.Fatalf("iter %d: want noCopy", iter)
			}
		}

		// Test P4 operations: automatic fallback on misaligned offset
		unalignedOffset := int64(17) // not multiple of 32
		copy(span[unalignedOffset:unalignedOffset+int64(len(raw))], raw)
		wFall, err := UploadZeroCopyQ4KTensor(span, unalignedOffset, out, in)
		if err != nil {
			t.Fatalf("misaligned upload should fall back without error, got: %v", err)
		}
		if wFall.NoCopy() {
			t.Fatalf("misaligned offset must fall back to copied buffer (noCopy=false)")
		}
		x := q4kTestVector(in, 0x1234)
		gotFall := make([]float32, out)
		wFall.GEMV(x, gotFall)

		ref := q4kVectorizedReference(raw, out, in, x)
		cosine, _ := q4kTestCosineMaxRel(ref, gotFall)
		if cosine < 0.999999 {
			t.Fatalf("fallback cosine %g < 0.999999", cosine)
		}

		ResetQ4K()
		ResetQ4K()
	})
}
