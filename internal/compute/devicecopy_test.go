package compute

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"testing"
)

func TestReadDeviceFloats(t *testing.T) {
	got := readDeviceFloats(2, func(dst []float32) { dst[0], dst[1] = 1, 2 })
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("readDeviceFloats = %v", got)
	}
	called := false
	if got := readDeviceFloats(0, func([]float32) { called = true }); len(got) != 0 || called {
		t.Fatalf("empty read = %v, called=%v", got, called)
	}
}

func TestCopyFromWriteCombined(t *testing.T) {
	sizes := []int{
		0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 65,
		127, 128, 129, 255, 256, 257, 511, 512, 1000, 1024,
		4095, 4096, 4097, 8192, 65536,
	}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			src := make([]byte, size)
			for i := range src {
				src[i] = byte((i*17 + 5) & 0xFF)
			}
			dst := make([]byte, size)

			n := CopyFromWriteCombined(dst, src)
			if n != size {
				t.Fatalf("expected %d bytes copied, got %d", size, n)
			}
			if !bytes.Equal(dst, src) {
				t.Fatalf("copied bytes do not match src for size %d", size)
			}
		})
	}
}

func TestCopyFromWriteCombined_MismatchedLengths(t *testing.T) {
	src := make([]byte, 1024)
	for i := range src {
		src[i] = byte(i & 0xFF)
	}

	// dst smaller than src
	dstSmall := make([]byte, 250)
	n := CopyFromWriteCombined(dstSmall, src)
	if n != 250 {
		t.Fatalf("expected 250 bytes copied, got %d", n)
	}
	if !bytes.Equal(dstSmall, src[:250]) {
		t.Fatalf("dstSmall does not match prefix of src")
	}

	// src smaller than dst
	dstLarge := make([]byte, 2048)
	// Fill with canary bytes
	for i := range dstLarge {
		dstLarge[i] = 0xAA
	}
	n = CopyFromWriteCombined(dstLarge, src)
	if n != 1024 {
		t.Fatalf("expected 1024 bytes copied, got %d", n)
	}
	if !bytes.Equal(dstLarge[:1024], src) {
		t.Fatalf("dstLarge prefix does not match src")
	}
	for i := 1024; i < len(dstLarge); i++ {
		if dstLarge[i] != 0xAA {
			t.Fatalf("byte at %d was modified (canary clobbered): got %x", i, dstLarge[i])
		}
	}

	// Zero length cases
	if n := CopyFromWriteCombined(nil, src); n != 0 {
		t.Fatalf("expected 0 for nil dst, got %d", n)
	}
	if n := CopyFromWriteCombined(dstLarge, nil); n != 0 {
		t.Fatalf("expected 0 for nil src, got %d", n)
	}
	if n := CopyFromWriteCombined(nil, nil); n != 0 {
		t.Fatalf("expected 0 for nil/nil, got %d", n)
	}
}

func TestCopyFromWriteCombined_UnalignedOffsets(t *testing.T) {
	const rawSize = 8192 + 128
	rawSrc := make([]byte, rawSize)
	rng := rand.New(rand.NewSource(42))
	rng.Read(rawSrc)

	offsets := []int{1, 2, 3, 5, 7, 8, 13, 17, 31, 33, 63, 64, 65, 127}
	copyLengths := []int{1, 15, 63, 64, 65, 128, 256, 1000, 4096}

	for _, srcOff := range offsets {
		for _, dstOff := range offsets {
			for _, copyLen := range copyLengths {
				if srcOff+copyLen > rawSize || dstOff+copyLen > rawSize {
					continue
				}
				src := rawSrc[srcOff : srcOff+copyLen]
				rawDst := make([]byte, rawSize)
				for i := range rawDst {
					rawDst[i] = 0x55
				}
				dst := rawDst[dstOff : dstOff+copyLen]

				n := CopyFromWriteCombined(dst, src)
				if n != copyLen {
					t.Fatalf("srcOff=%d dstOff=%d copyLen=%d: expected n=%d, got %d",
						srcOff, dstOff, copyLen, copyLen, n)
				}
				if !bytes.Equal(dst, src) {
					t.Fatalf("srcOff=%d dstOff=%d copyLen=%d: mismatch",
						srcOff, dstOff, copyLen)
				}
				// Verify before and after canary regions in rawDst
				for i := 0; i < dstOff; i++ {
					if rawDst[i] != 0x55 {
						t.Fatalf("clobbered pre-canary at %d", i)
					}
				}
				for i := dstOff + copyLen; i < rawSize; i++ {
					if rawDst[i] != 0x55 {
						t.Fatalf("clobbered post-canary at %d", i)
					}
				}
			}
		}
	}
}

func TestCopyFromWriteCombined_PathsParity(t *testing.T) {
	// Directly verify that copyFromWriteCombinedStreaming and
	// copyFromWriteCombinedFallback produce bit-identical results.
	testSizes := []int{0, 1, 15, 63, 64, 65, 127, 128, 255, 256, 512, 1024, 4096, 9000}
	for _, size := range testSizes {
		src := make([]byte, size)
		for i := range src {
			src[i] = byte(i * 3)
		}
		dstStreaming := make([]byte, size)
		dstFallback := make([]byte, size)

		nStream := copyFromWriteCombinedStreaming(dstStreaming, src)
		nFallback := copyFromWriteCombinedFallback(dstFallback, src)

		if nStream != size || nFallback != size {
			t.Fatalf("size=%d: nStream=%d, nFallback=%d", size, nStream, nFallback)
		}
		if !bytes.Equal(dstStreaming, dstFallback) {
			t.Fatalf("size=%d: streaming and fallback outputs differ", size)
		}
	}
}

func TestCopyFromWriteCombined_EnvConfig(t *testing.T) {
	origODL := os.Getenv("ODL_VERBS_WC_STREAM_COPY")
	origFAK := os.Getenv("FAK_WC_STREAM_COPY")
	origSupported := avx512StreamSupported
	defer func() {
		os.Setenv("ODL_VERBS_WC_STREAM_COPY", origODL)
		os.Setenv("FAK_WC_STREAM_COPY", origFAK)
		avx512StreamSupported = origSupported
	}()

	os.Setenv("ODL_VERBS_WC_STREAM_COPY", "1")
	if !HasAVX512StreamSupport() {
		t.Errorf("expected HasAVX512StreamSupport()=true with ODL_VERBS_WC_STREAM_COPY=1")
	}
	avx512StreamSupported = HasAVX512StreamSupport()
	buf := make([]byte, 128)
	CopyFromWriteCombined(buf, buf)

	os.Setenv("ODL_VERBS_WC_STREAM_COPY", "0")
	if HasAVX512StreamSupport() {
		t.Errorf("expected HasAVX512StreamSupport()=false with ODL_VERBS_WC_STREAM_COPY=0")
	}
	avx512StreamSupported = HasAVX512StreamSupport()
	CopyFromWriteCombined(buf, buf)

	os.Unsetenv("ODL_VERBS_WC_STREAM_COPY")
	os.Setenv("FAK_WC_STREAM_COPY", "1")
	if !HasAVX512StreamSupport() {
		t.Errorf("expected HasAVX512StreamSupport()=true with FAK_WC_STREAM_COPY=1")
	}
	avx512StreamSupported = HasAVX512StreamSupport()
	CopyFromWriteCombined(buf, buf)

	os.Setenv("FAK_WC_STREAM_COPY", "0")
	if HasAVX512StreamSupport() {
		t.Errorf("expected HasAVX512StreamSupport()=false with FAK_WC_STREAM_COPY=0")
	}
	avx512StreamSupported = HasAVX512StreamSupport()
	CopyFromWriteCombined(buf, buf)
}

func TestCopyFromWriteCombined_Overlapping(t *testing.T) {
	buf := []byte("0123456789abcdefghijklmnopqrstuvwxyz")
	// Overlapping: dst starts inside src
	CopyFromWriteCombined(buf[4:20], buf[0:16])
	want := []byte("01230123456789abcdefklmnopqrstuvwxyz")
	if !bytes.Equal(buf, want) {
		t.Fatalf("overlap copy mismatch: got %q, want %q", buf, want)
	}
}

func TestCopyFromWriteCombined_Hook(t *testing.T) {
	hookCalled := false
	avx512StreamCopyHook = func(dst, src []byte) int {
		hookCalled = true
		return copy(dst, src)
	}
	defer func() {
		avx512StreamCopyHook = nil
	}()

	src := []byte{1, 2, 3}
	dst := make([]byte, 3)
	n := CopyFromWriteCombined(dst, src)
	if !hookCalled || n != 3 || !bytes.Equal(dst, src) {
		t.Fatalf("hook not called properly: called=%v, n=%d", hookCalled, n)
	}
}

func BenchmarkCopyFromWriteCombined(b *testing.B) {
	const size = 1024 * 1024 // 1 MB buffer
	src := make([]byte, size)
	dst := make([]byte, size)
	for i := range src {
		src[i] = byte(i)
	}
	b.SetBytes(size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CopyFromWriteCombined(dst, src)
	}
}

func BenchmarkStandardCopy(b *testing.B) {
	const size = 1024 * 1024 // 1 MB buffer
	src := make([]byte, size)
	dst := make([]byte, size)
	for i := range src {
		src[i] = byte(i)
	}
	b.SetBytes(size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(dst, src)
	}
}
