package compute

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestStreamingLoadWC_Sizes(t *testing.T) {
	sizes := []int{
		0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 65,
		127, 128, 129, 255, 256, 257, 511, 512, 1000, 1024,
		4095, 4096, 4097, 8192, 65536,
	}

	modes := []struct {
		name string
		env  string
	}{
		{"streaming", "1"},
		{"fallback", "0"},
	}

	for _, m := range modes {
		t.Run(m.name, func(t *testing.T) {
			t.Setenv("FAK_AVX512_STREAMING_LOAD", m.env)

			for _, size := range sizes {
				t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
					src := make([]byte, size)
					for i := range src {
						src[i] = byte((i*31 + 7) & 0xFF)
					}
					dst := make([]byte, size)

					n := StreamingLoadWC(dst, src)
					if n != size {
						t.Fatalf("expected %d bytes copied, got %d", size, n)
					}
					if !bytes.Equal(dst, src) {
						t.Fatalf("copied data does not match src for size %d", size)
					}
				})
			}
		})
	}
}

func TestStreamingLoadWC_Alignments(t *testing.T) {
	const rawSize = 8192 + 256
	rawSrc := make([]byte, rawSize)
	rng := rand.New(rand.NewSource(99))
	rng.Read(rawSrc)

	// Specified alignment offsets in issue #11242: 0, 1, 3, 7, 31
	offsets := []int{0, 1, 3, 7, 31, 63, 64}
	lengths := []int{0, 15, 64, 127, 128, 256, 1000, 4096}

	for _, srcOff := range offsets {
		for _, dstOff := range offsets {
			for _, copyLen := range lengths {
				if srcOff+copyLen > rawSize || dstOff+copyLen > rawSize {
					continue
				}

				src := rawSrc[srcOff : srcOff+copyLen]
				rawDst := make([]byte, rawSize)
				const canary byte = 0xAA
				for i := range rawDst {
					rawDst[i] = canary
				}
				dst := rawDst[dstOff : dstOff+copyLen]

				n := StreamingLoadWC(dst, src)
				if n != copyLen {
					t.Fatalf("srcOff=%d dstOff=%d len=%d: expected n=%d, got %d",
						srcOff, dstOff, copyLen, copyLen, n)
				}
				if !bytes.Equal(dst, src) {
					t.Fatalf("srcOff=%d dstOff=%d len=%d: byte mismatch",
						srcOff, dstOff, copyLen)
				}

				// Verify canary region before dst
				for i := 0; i < dstOff; i++ {
					if rawDst[i] != canary {
						t.Fatalf("clobbered pre-canary byte at %d: got 0x%02x, want 0x%02x", i, rawDst[i], canary)
					}
				}
				// Verify canary region after dst
				for i := dstOff + copyLen; i < rawSize; i++ {
					if rawDst[i] != canary {
						t.Fatalf("clobbered post-canary byte at %d: got 0x%02x, want 0x%02x", i, rawDst[i], canary)
					}
				}
			}
		}
	}
}

func TestStreamingLoadWC_MismatchedLengthsAndEmpty(t *testing.T) {
	src := make([]byte, 1024)
	for i := range src {
		src[i] = byte(i & 0xFF)
	}

	// 1. Destination smaller than source
	dstSmall := make([]byte, 250)
	n := StreamingLoadWC(dstSmall, src)
	if n != 250 {
		t.Fatalf("expected 250 bytes copied, got %d", n)
	}
	if !bytes.Equal(dstSmall, src[:250]) {
		t.Fatalf("dstSmall does not match prefix of src")
	}

	// 2. Source smaller than destination (with canary bytes)
	dstLarge := make([]byte, 2048)
	const canary byte = 0x7E
	for i := range dstLarge {
		dstLarge[i] = canary
	}
	n = StreamingLoadWC(dstLarge, src)
	if n != 1024 {
		t.Fatalf("expected 1024 bytes copied, got %d", n)
	}
	if !bytes.Equal(dstLarge[:1024], src) {
		t.Fatalf("dstLarge prefix does not match src")
	}
	for i := 1024; i < len(dstLarge); i++ {
		if dstLarge[i] != canary {
			t.Fatalf("canary byte clobbered at index %d: got %x, want %x", i, dstLarge[i], canary)
		}
	}

	// 3. Empty and nil slices
	if n := StreamingLoadWC(nil, src); n != 0 {
		t.Fatalf("expected 0 for nil dst, got %d", n)
	}
	if n := StreamingLoadWC(dstLarge, nil); n != 0 {
		t.Fatalf("expected 0 for nil src, got %d", n)
	}
	if n := StreamingLoadWC(nil, nil); n != 0 {
		t.Fatalf("expected 0 for nil/nil, got %d", n)
	}
	if n := StreamingLoadWC(dstLarge[:0], src); n != 0 {
		t.Fatalf("expected 0 for empty dst, got %d", n)
	}
	if n := StreamingLoadWC(dstLarge, src[:0]); n != 0 {
		t.Fatalf("expected 0 for empty src, got %d", n)
	}
}

func TestStreamingLoadWC_Overlapping(t *testing.T) {
	orig := []byte("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz")
	buf := make([]byte, len(orig))
	copy(buf, orig)
	want := make([]byte, len(orig))
	copy(want, orig)

	// Forward overlap: dst starts inside src
	StreamingLoadWC(buf[10:30], buf[0:20])
	copy(want[10:30], want[0:20])
	if !bytes.Equal(buf, want) {
		t.Fatalf("forward overlapping copy mismatch:\ngot:  %s\nwant: %s", buf, want)
	}

	// Backward overlap: src starts inside dst
	copy(buf, orig)
	copy(want, orig)
	StreamingLoadWC(buf[0:20], buf[10:30])
	copy(want[0:20], want[10:30])
	if !bytes.Equal(buf, want) {
		t.Fatalf("backward overlapping copy mismatch:\ngot:  %s\nwant: %s", buf, want)
	}
}

func TestStreamingLoadWC_PathsParity(t *testing.T) {
	sizes := []int{0, 1, 15, 63, 64, 65, 127, 128, 255, 256, 511, 512, 1024, 4096, 65536}
	for _, size := range sizes {
		src := make([]byte, size)
		for i := range src {
			src[i] = byte((i * 13) ^ 0x5C)
		}
		dstStreaming := make([]byte, size)
		dstFallback := make([]byte, size)

		nStream := streamingLoadStreaming(dstStreaming, src)
		nFallback := streamingLoadFallback(dstFallback, src)

		if nStream != size || nFallback != size {
			t.Fatalf("size %d: nStream=%d, nFallback=%d", size, nStream, nFallback)
		}
		if !bytes.Equal(dstStreaming, dstFallback) {
			t.Fatalf("size %d: streaming and fallback outputs differ", size)
		}
	}
}

func TestHasAVX512StreamingLoad_Env(t *testing.T) {
	envKeys := []string{
		"FAK_AVX512_STREAMING_LOAD",
		"FAK_WC_STREAM_LOAD",
		"ODL_VERBS_WC_STREAM_LOAD",
		"FAK_WC_STREAM_COPY",
		"ODL_VERBS_WC_STREAM_COPY",
	}

	for _, key := range envKeys {
		t.Run(key+"_true", func(t *testing.T) {
			t.Setenv(key, "1")
			if !HasAVX512StreamingLoad() {
				t.Fatalf("expected HasAVX512StreamingLoad()=true with %s=1", key)
			}
		})
		t.Run(key+"_false", func(t *testing.T) {
			t.Setenv(key, "0")
			if HasAVX512StreamingLoad() {
				t.Fatalf("expected HasAVX512StreamingLoad()=false with %s=0", key)
			}
		})
	}
}

func TestHasAVX512StreamingLoad_Procfs(t *testing.T) {
	t.Setenv("FAK_AVX512_STREAMING_LOAD", "")
	t.Setenv("FAK_WC_STREAM_LOAD", "")
	t.Setenv("ODL_VERBS_WC_STREAM_LOAD", "")
	t.Setenv("FAK_WC_STREAM_COPY", "")
	t.Setenv("ODL_VERBS_WC_STREAM_COPY", "")

	tmpDir := t.TempDir()
	origPath := procCPUInfoPath
	defer func() { procCPUInfoPath = origPath }()

	// 1. AMD Strix Halo model name
	strixHaloPath := filepath.Join(tmpDir, "cpuinfo_strix")
	strixContent := "processor\t: 0\nmodel name\t: AMD Ryzen AI MAX+ 395 with Radeon 8060S\nflags\t: fpu vme sse sse2\n"
	if err := os.WriteFile(strixHaloPath, []byte(strixContent), 0644); err != nil {
		t.Fatal(err)
	}
	procCPUInfoPath = strixHaloPath
	if !checkProcCPUInfoAVX512() {
		t.Errorf("expected Strix Halo cpuinfo to detect AVX-512 streaming support")
	}

	// 2. Explicit avx512f flag
	avx512Path := filepath.Join(tmpDir, "cpuinfo_avx512")
	avxContent := "processor\t: 0\nmodel name\t: Generic AMD CPU\nflags\t: fpu vme sse sse2 avx avx2 avx512f avx512cd\n"
	if err := os.WriteFile(avx512Path, []byte(avxContent), 0644); err != nil {
		t.Fatal(err)
	}
	procCPUInfoPath = avx512Path
	if !checkProcCPUInfoAVX512() {
		t.Errorf("expected avx512f flag in cpuinfo to detect AVX-512 streaming support")
	}

	// 3. Negative case: no AVX-512 flags and legacy model name
	noAVXPath := filepath.Join(tmpDir, "cpuinfo_noavx")
	noAVXContent := "processor\t: 0\nmodel name\t: Legacy CPU\nflags\t: fpu vme sse sse2 avx avx2\n"
	if err := os.WriteFile(noAVXPath, []byte(noAVXContent), 0644); err != nil {
		t.Fatal(err)
	}
	procCPUInfoPath = noAVXPath
	if checkProcCPUInfoAVX512() {
		t.Errorf("expected non-AVX512 cpuinfo to return false")
	}
}

func TestHasAVX512StreamingLoad_CPUIDEmulation(t *testing.T) {
	origChecker := cpuidFeatureChecker
	defer func() { cpuidFeatureChecker = origChecker }()

	t.Setenv("FAK_AVX512_STREAMING_LOAD", "")
	t.Setenv("FAK_WC_STREAM_LOAD", "")
	t.Setenv("ODL_VERBS_WC_STREAM_LOAD", "")
	t.Setenv("FAK_WC_STREAM_COPY", "")
	t.Setenv("ODL_VERBS_WC_STREAM_COPY", "")

	cpuidFeatureChecker = func() bool { return true }
	if !HasAVX512StreamingLoad() {
		t.Errorf("expected cpuidFeatureChecker()=true to report HasAVX512StreamingLoad()=true")
	}

	cpuidFeatureChecker = func() bool { return false }
	// When hook returns false and env is unset, it proceeds to syscpu / procfs
}

func TestStreamingLoadWC_AsmHook(t *testing.T) {
	origHook := streamingLoadAsmHook
	defer func() { streamingLoadAsmHook = origHook }()

	hookCalled := false
	streamingLoadAsmHook = func(dst, src []byte) int {
		hookCalled = true
		return copy(dst, src)
	}

	src := []byte{10, 20, 30, 40}
	dst := make([]byte, 4)
	n := StreamingLoadWC(dst, src)
	if !hookCalled || n != 4 || !bytes.Equal(dst, src) {
		t.Fatalf("hook not called properly: hookCalled=%v, n=%d, dst=%v", hookCalled, n, dst)
	}
}

func BenchmarkStreamingLoadWC(b *testing.B) {
	const size = 1024 * 1024 // 1 MB buffer
	src := make([]byte, size)
	dst := make([]byte, size)
	for i := range src {
		src[i] = byte(i)
	}
	b.SetBytes(size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		StreamingLoadWC(dst, src)
	}
}

func BenchmarkStreamingLoadWC_ChunkSizes(b *testing.B) {
	sizes := []int{64, 512, 4096, 65536, 1024 * 1024}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dB", size), func(b *testing.B) {
			src := make([]byte, size)
			dst := make([]byte, size)
			for i := range src {
				src[i] = byte(i)
			}
			b.SetBytes(int64(size))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				StreamingLoadWC(dst, src)
			}
		})
	}
}
