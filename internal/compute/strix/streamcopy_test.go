package strix

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"testing"
	"time"
)

// TestStreamCopy_BitwiseIdentity tests exact bitwise identity between standard copy
// and StreamCopy across varying payload sizes (1 byte, 63 bytes, 64 bytes, 128 bytes,
// 4096 bytes, 65536 bytes, 1 MB) and arbitrary unaligned offsets.
func TestStreamCopy_BitwiseIdentity(t *testing.T) {
	sizes := []int{
		1,
		7,
		15,
		31,
		63,
		64,
		65,
		127,
		128,
		129,
		255,
		256,
		512,
		1024,
		4096,
		65536,
		1048576, // 1 MB
	}

	alignmentPairs := []struct {
		name   string
		srcOff int
		dstOff int
	}{
		{"Aligned_0_0", 0, 0},
		{"UnalignedSrc_1_0", 1, 0},
		{"UnalignedDst_0_1", 0, 1},
		{"SameOffset_1_1", 1, 1},
		{"SameOffset_32_32", 32, 32},
		{"DifferentOffsets_7_23", 7, 23},
		{"DifferentOffsets_63_1", 63, 1},
		{"MaxOffset_63_63", 63, 63},
	}

	modes := []struct {
		name          string
		forceFallback bool
	}{
		{"Default", false},
		{"GoFallback", true},
	}

	const pad = 128

	for _, mode := range modes {
		t.Run(mode.name, func(t *testing.T) {
			for _, size := range sizes {
				for _, align := range alignmentPairs {
					testName := fmt.Sprintf("size_%d/%s", size, align.name)
					t.Run(testName, func(t *testing.T) {
						// Allocate backing storage with padding to allow arbitrary alignment
						srcRaw := make([]byte, pad+size+pad)
						dstRaw := make([]byte, pad+size+pad)
						refDstRaw := make([]byte, pad+size+pad)

						// Deterministic source data
						for i := 0; i < size; i++ {
							srcRaw[align.srcOff+i] = byte((i*17 + 31 + size) & 0xFF)
						}

						// Guard canary pattern before and after
						for i := 0; i < pad; i++ {
							dstRaw[i] = 0xAA
							dstRaw[align.dstOff+size+i] = 0xAA
							refDstRaw[i] = 0xAA
							refDstRaw[align.dstOff+size+i] = 0xAA
						}

						srcSlice := srcRaw[align.srcOff : align.srcOff+size]
						dstSlice := dstRaw[align.dstOff : align.dstOff+size]
						refDstSlice := refDstRaw[align.dstOff : align.dstOff+size]

						// 1. Reference standard copy
						copy(refDstSlice, srcSlice)

						// 2. StreamCopy
						var telem StreamCopyTelemetry
						opts := []StreamCopyOption{
							WithTelemetry(&telem),
							WithSourceWC(true),
							WithDestWC(true),
							WithStoreFence(true),
						}
						if mode.forceFallback {
							opts = append(opts, WithForceFallback(true))
						}

						n, err := StreamCopy(dstSlice, srcSlice, opts...)
						if err != nil {
							t.Fatalf("StreamCopy failed: %v", err)
						}
						if n != int64(size) {
							t.Fatalf("StreamCopy returned %d bytes, want %d", n, size)
						}

						// 3. Bitwise identity check
						if !bytes.Equal(dstSlice, refDstSlice) {
							t.Fatalf("Bitwise mismatch between StreamCopy and standard copy (size %d, align %s)", size, align.name)
						}

						// 4. Canary check: ensure no out-of-bounds writes corrupted surrounding padding
						for i := 0; i < pad; i++ {
							if dstRaw[align.dstOff+size+i] != 0xAA {
								t.Fatalf("Tail canary corruption at offset %d", i)
							}
						}

						// 5. Verify telemetry
						if telem.BytesCopied != int64(size) {
							t.Errorf("Telemetry BytesCopied = %d, want %d", telem.BytesCopied, size)
						}
					})
				}
			}
		})
	}
}

// TestStreamCopy_OptionsAndTelemetry verifies behavior of telemetry collection,
// write-combining flags, and fencing options.
func TestStreamCopy_OptionsAndTelemetry(t *testing.T) {
	size := 4096
	src := make([]byte, size)
	dst := make([]byte, size)
	for i := range src {
		src[i] = byte(i)
	}

	var telem StreamCopyTelemetry
	n, err := StreamCopy(dst, src,
		WithSourceWC(true),
		WithDestWC(true),
		WithStoreFence(true),
		WithForceFallback(true),
		WithTelemetry(&telem),
	)
	if err != nil {
		t.Fatalf("StreamCopy failed: %v", err)
	}
	if n != int64(size) {
		t.Fatalf("copied %d, want %d", n, size)
	}

	if telem.BytesCopied != int64(size) {
		t.Errorf("telem.BytesCopied = %d, want %d", telem.BytesCopied, size)
	}
	if telem.ChunksProcessed <= 0 {
		t.Errorf("telem.ChunksProcessed = %d, want > 0", telem.ChunksProcessed)
	}
	if !telem.SourceWC || !telem.DestWC {
		t.Errorf("expected SourceWC and DestWC to be true")
	}
	if telem.Duration < 0 {
		t.Errorf("expected non-negative duration, got %v", telem.Duration)
	}
}

// TestStreamCopy_HardwareCapabilityAndOverrides verifies hardware detection,
// test overrides, and environment variable flags.
func TestStreamCopy_HardwareCapabilityAndOverrides(t *testing.T) {
	defer SetAVX512StreamingOverrideForTesting(nil)
	_ = os.Unsetenv("FAK_AVX512_STREAMING_FORCE")
	_ = os.Unsetenv("FAK_AVX512_STREAMING_DISABLE")
	_ = os.Unsetenv("FAK_AVX512_FORCE")
	_ = os.Unsetenv("FAK_AVX512_DISABLE")

	// 1. Programmatic override
	overrideTrue := true
	SetAVX512StreamingOverrideForTesting(&overrideTrue)
	if !HasAVX512Streaming() {
		t.Errorf("expected HasAVX512Streaming=true when programmatic override is true")
	}

	overrideFalse := false
	SetAVX512StreamingOverrideForTesting(&overrideFalse)
	if HasAVX512Streaming() {
		t.Errorf("expected HasAVX512Streaming=false when programmatic override is false")
	}

	SetAVX512StreamingOverrideForTesting(nil)

	// 2. Env variable disable
	_ = os.Setenv("FAK_AVX512_STREAMING_DISABLE", "1")
	ResetAVX512StreamingCapabilityForTesting()
	if HasAVX512Streaming() {
		t.Errorf("expected HasAVX512Streaming=false with FAK_AVX512_STREAMING_DISABLE")
	}
	_ = os.Unsetenv("FAK_AVX512_STREAMING_DISABLE")
	ResetAVX512StreamingCapabilityForTesting()

	// 3. Env variable force
	_ = os.Setenv("FAK_AVX512_STREAMING_FORCE", "1")
	ResetAVX512StreamingCapabilityForTesting()
	if !HasAVX512Streaming() {
		t.Errorf("expected HasAVX512Streaming=true with FAK_AVX512_STREAMING_FORCE")
	}
	_ = os.Unsetenv("FAK_AVX512_STREAMING_FORCE")
	ResetAVX512StreamingCapabilityForTesting()

	// 4. MemoryStoreFence execution
	MemoryStoreFence()
}

// TestStreamCopy_EdgeCasesAndErrors tests edge conditions, destination buffer size checks,
// and buffer overlap.
func TestStreamCopy_EdgeCasesAndErrors(t *testing.T) {
	// Zero length
	dst := make([]byte, 64)
	src := []byte{}
	n, err := StreamCopy(dst, src)
	if err != nil || n != 0 {
		t.Fatalf("expected 0, nil for empty src, got %d, %v", n, err)
	}

	// Destination too small
	src = make([]byte, 128)
	dst = make([]byte, 64)
	_, err = StreamCopy(dst, src)
	if !errors.Is(err, ErrDestinationTooSmall) {
		t.Fatalf("expected ErrDestinationTooSmall, got %v", err)
	}

	// Identical buffers
	buf := make([]byte, 256)
	for i := range buf {
		buf[i] = byte(i)
	}
	n, err = StreamCopy(buf, buf)
	if err != nil || n != int64(len(buf)) {
		t.Fatalf("expected identity copy success, got %d, %v", n, err)
	}

	// Overlapping buffers with forward copy hazard
	overlapBuf := make([]byte, 256)
	for i := range overlapBuf {
		overlapBuf[i] = byte(i)
	}
	refOverlap := make([]byte, 256)
	copy(refOverlap, overlapBuf)

	// Copy forward: dst is offset 10 into src offset 0
	copy(refOverlap[10:100], refOverlap[0:90])
	n, err = StreamCopy(overlapBuf[10:100], overlapBuf[0:90])
	if err != nil {
		t.Fatalf("overlapping StreamCopy failed: %v", err)
	}
	if n != 90 {
		t.Fatalf("overlapping StreamCopy copied %d, want 90", n)
	}
	if !bytes.Equal(overlapBuf[10:100], refOverlap[10:100]) {
		t.Fatalf("overlapping StreamCopy did not match standard copy")
	}
}

// TestStreamCopy_UMABufferIntegration tests safe integration with platform/strix/uma_mmu.go.
func TestStreamCopy_UMABufferIntegration(t *testing.T) {
	size := 65536
	uma, err := NewUMABuffer(size)
	if err != nil {
		t.Fatalf("NewUMABuffer failed: %v", err)
	}
	defer uma.Close()

	testData := make([]byte, 4096)
	for i := range testData {
		testData[i] = byte((i*13 + 7) & 0xFF)
	}

	// 1. Write via StreamCopyToUMA helper
	n, err := StreamCopyToUMA(uma, 1024, testData)
	if err != nil {
		t.Fatalf("StreamCopyToUMA failed: %v", err)
	}
	if n != int64(len(testData)) {
		t.Fatalf("StreamCopyToUMA wrote %d, want %d", n, len(testData))
	}

	// 2. Read back via StreamCopyFromUMA helper
	readBack := make([]byte, len(testData))
	rn, err := StreamCopyFromUMA(readBack, uma, 1024, len(testData))
	if err != nil {
		t.Fatalf("StreamCopyFromUMA failed: %v", err)
	}
	if rn != int64(len(testData)) {
		t.Fatalf("StreamCopyFromUMA read %d, want %d", rn, len(testData))
	}
	if !bytes.Equal(readBack, testData) {
		t.Fatalf("Read back UMA data mismatch")
	}

	// 3. Direct methods on UMABuffer: StreamWrite and StreamRead
	moreData := make([]byte, 1024)
	for i := range moreData {
		moreData[i] = byte(0xFE)
	}
	wn, err := uma.StreamWrite(2048, moreData)
	if err != nil || wn != int64(len(moreData)) {
		t.Fatalf("uma.StreamWrite failed: %v", err)
	}

	readDirect := make([]byte, len(moreData))
	rdn, err := uma.StreamRead(readDirect, 2048, len(moreData))
	if err != nil || rdn != int64(len(moreData)) {
		t.Fatalf("uma.StreamRead failed: %v", err)
	}
	if !bytes.Equal(readDirect, moreData) {
		t.Fatalf("Direct readback mismatch")
	}

	// 4. Bounds checks
	_, err = uma.StreamWrite(size-10, make([]byte, 100))
	if !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("expected ErrOutOfBounds for over-length write, got %v", err)
	}

	_, err = uma.StreamRead(make([]byte, 100), size-10, 100)
	if !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("expected ErrOutOfBounds for over-length read, got %v", err)
	}

	// 5. Closed buffer checks
	_ = uma.Close()
	_, err = uma.StreamWrite(0, testData)
	if !errors.Is(err, ErrBufferClosed) {
		t.Errorf("expected ErrBufferClosed after Close, got %v", err)
	}
	_, err = uma.StreamRead(readBack, 0, len(testData))
	if !errors.Is(err, ErrBufferClosed) {
		t.Errorf("expected ErrBufferClosed after Close, got %v", err)
	}
}

// TestStreamCopy_ConcurrencyAndRace tests concurrent StreamCopy operations across
// multiple goroutines under -race to prove thread safety and absence of data races.
func TestStreamCopy_ConcurrencyAndRace(t *testing.T) {
	const numWorkers = 16
	const iterations = 50

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		workerID := w
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(time.Now().UnixNano() + int64(workerID)*1000)))

			testSizes := []int{63, 64, 128, 512, 4096, 65536}

			for i := 0; i < iterations; i++ {
				size := testSizes[rng.Intn(len(testSizes))]
				srcAlign := rng.Intn(64)
				dstAlign := rng.Intn(64)

				srcBuf := make([]byte, 64+size)
				dstBuf := make([]byte, 64+size)
				refBuf := make([]byte, 64+size)

				src := srcBuf[srcAlign : srcAlign+size]
				dst := dstBuf[dstAlign : dstAlign+size]
				ref := refBuf[dstAlign : dstAlign+size]

				for k := range src {
					src[k] = byte((k + workerID + i) & 0xFF)
				}
				copy(ref, src)

				var telem StreamCopyTelemetry
				n, err := StreamCopy(dst, src, WithTelemetry(&telem))
				if err != nil {
					t.Errorf("worker %d iteration %d failed: %v", workerID, i, err)
					return
				}
				if n != int64(size) {
					t.Errorf("worker %d iteration %d copied %d, want %d", workerID, i, n, size)
					return
				}
				if !bytes.Equal(dst, ref) {
					t.Errorf("worker %d iteration %d bitwise mismatch", workerID, i)
					return
				}
			}
		}()
	}

	wg.Wait()
}

// BenchmarkStreamCopy benchmarks StreamCopy vs standard copy across multiple block sizes.
func BenchmarkStreamCopy(b *testing.B) {
	sizes := []struct {
		name string
		size int
	}{
		{"64B", 64},
		{"512B", 512},
		{"4KB", 4096},
		{"64KB", 65536},
		{"1MB", 1048576},
		{"16MB", 16 * 1024 * 1024},
	}

	for _, s := range sizes {
		src := make([]byte, s.size)
		dst := make([]byte, s.size)
		for i := range src {
			src[i] = byte(i & 0xFF)
		}

		b.Run("StandardCopy/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.size))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(dst, src)
			}
		})

		b.Run("StreamCopy/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.size))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = StreamCopy(dst, src)
			}
		})

		b.Run("StreamCopyFallback/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.size))
			opts := []StreamCopyOption{WithForceFallback(true)}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = StreamCopy(dst, src, opts...)
			}
		})
	}
}
