package compute

import (
	"bytes"
	"math/rand"
	"testing"
)

func TestUMAStridedCopyNumericalParity(t *testing.T) {
	guard := NewUMAStridedCopier("gfx1151")
	rng := rand.New(rand.NewSource(1337))

	const iterations = 10000
	for i := 0; i < iterations; i++ {
		width := rng.Intn(128) + 1
		height := rng.Intn(16) + 1
		srcPitch := width + rng.Intn(64)
		dstPitch := width + rng.Intn(64)

		srcSize := (height-1)*srcPitch + width + rng.Intn(32)
		dstSize := (height-1)*dstPitch + width + rng.Intn(32)

		src := make([]byte, srcSize)
		dstActual := make([]byte, dstSize)
		dstExpected := make([]byte, dstSize)

		rng.Read(src)
		rng.Read(dstActual)
		copy(dstExpected, dstActual)

		errExpected := CPURefCopy2D(dstExpected, dstPitch, src, srcPitch, width, height)
		if errExpected != nil {
			t.Fatalf("iteration %d: CPU ref failed: %v", i, errExpected)
		}

		errActual := guard.Copy2D(dstActual, dstPitch, src, srcPitch, width, height)
		if errActual != nil {
			t.Fatalf("iteration %d: UMA copy failed: %v", i, errActual)
		}

		if !bytes.Equal(dstActual, dstExpected) {
			t.Fatalf("iteration %d: mismatch! width=%d, height=%d, srcPitch=%d, dstPitch=%d",
				i, width, height, srcPitch, dstPitch)
		}
	}
}

func TestUMAStridedCopyZeroCopyIdentity(t *testing.T) {
	guard := NewUMAStridedCopier("gfx1151")
	data := make([]byte, 4096)
	rand.New(rand.NewSource(42)).Read(data)

	// In-place identity
	err := guard.Copy2D(data, 64, data, 64, 64, 64)
	if err != nil {
		t.Fatalf("identity copy failed: %v", err)
	}
	if guard.IdentityZeroCopyCount != 1 {
		t.Errorf("IdentityZeroCopyCount = %d, want 1", guard.IdentityZeroCopyCount)
	}
}

func BenchmarkUMAIdentityFastPath(b *testing.B) {
	guard := NewUMAStridedCopier("gfx1151")
	data := make([]byte, 4096)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = guard.Copy2D(data, 4096, data, 4096, 4096, 1)
	}
}

func BenchmarkUMAContiguousFastPath(b *testing.B) {
	guard := NewUMAStridedCopier("gfx1151")
	src := make([]byte, 4096)
	dst := make([]byte, 4096)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = guard.Copy2D(dst, 4096, src, 4096, 4096, 1)
	}
}

func TestUMAStridedCopyAutoregressiveGather(t *testing.T) {
	// Simulate 16-step autoregressive gather of 2D KV tiles (DeepSeek V4 / Qwen sparse attention style)
	// Single-call 2D strided gather across all 16 steps for head 3.
	guard := NewUMAStridedCopier("gfx1151")
	const steps = 16
	const headDim = 128
	const heads = 8
	const headBytes = headDim * 2 // f16
	const tokenStride = heads * headBytes

	buffer := make([]byte, steps*tokenStride)
	rng := rand.New(rand.NewSource(99))
	rng.Read(buffer)

	// Destination for gathered head 3 across 16 steps (compact layout [steps, headBytes])
	gatherDst := make([]byte, steps*headBytes)

	// Offset to head 3 within token 0
	targetHead := 3
	headOffset := targetHead * headBytes

	// Single 2D strided copy call:
	// height = steps (16 rows)
	// width = headBytes (256 bytes per head)
	// srcPitch = tokenStride (2048 bytes between consecutive tokens in buffer)
	// dstPitch = headBytes (256 bytes contiguous in destination)
	err := guard.Copy2D(gatherDst, headBytes, buffer[headOffset:], tokenStride, headBytes, steps)
	if err != nil {
		t.Fatalf("2D strided gather failed: %v", err)
	}

	// Verify each gathered step matches buffer head 3
	for step := 0; step < steps; step++ {
		expected := buffer[step*tokenStride+headOffset : step*tokenStride+headOffset+headBytes]
		actual := gatherDst[step*headBytes : (step+1)*headBytes]
		if !bytes.Equal(actual, expected) {
			t.Fatalf("step %d gather corruption detected: bit-level divergence", step)
		}
	}
}

func TestUMAStridedCopyErrors(t *testing.T) {
	guard := NewUMAStridedCopier("gfx1151")
	buf := make([]byte, 100)

	// Non-positive dimensions
	if err := guard.Copy2D(buf, 10, buf, 10, 0, 1); err != ErrInvalidDimensions {
		t.Errorf("width=0: want ErrInvalidDimensions, got %v", err)
	}
	if err := guard.Copy2D(buf, 10, buf, 10, 10, 0); err != ErrInvalidDimensions {
		t.Errorf("height=0: want ErrInvalidDimensions, got %v", err)
	}

	// Width > Pitch
	if err := guard.Copy2D(buf, 5, buf, 10, 10, 1); err == nil {
		t.Errorf("width > dstPitch: want error, got nil")
	}

	// Buffer overflow
	smallDst := make([]byte, 5)
	if err := guard.Copy2D(smallDst, 10, buf, 10, 10, 2); err == nil {
		t.Errorf("dst too small: want error, got nil")
	}
}
