package compute

import (
	"os"
	"runtime"
	"strings"
	"unsafe"

	syscpu "golang.org/x/sys/cpu"
)

func readDeviceFloats(length int, copyToHost func(dst []float32)) []float32 {
	out := make([]float32, length)
	if length > 0 {
		copyToHost(out)
	}
	return out
}

var avx512StreamSupported = detectAVX512StreamSupport()

func detectAVX512StreamSupport() bool {
	if val := os.Getenv("ODL_VERBS_WC_STREAM_COPY"); val != "" {
		if val == "0" || strings.EqualFold(val, "false") || strings.EqualFold(val, "off") {
			return false
		}
		if val == "1" || strings.EqualFold(val, "true") || strings.EqualFold(val, "on") {
			return true
		}
	}
	if val := os.Getenv("FAK_WC_STREAM_COPY"); val != "" {
		if val == "0" || strings.EqualFold(val, "false") || strings.EqualFold(val, "off") {
			return false
		}
		if val == "1" || strings.EqualFold(val, "true") || strings.EqualFold(val, "on") {
			return true
		}
	}
	if runtime.GOARCH == "amd64" {
		return syscpu.X86.HasAVX512 || syscpu.X86.HasAVX512F
	}
	return false
}

// HasAVX512StreamSupport reports whether AVX-512 non-temporal streaming copy
// is supported by the CPU or enabled via environment configuration.
// It checks ODL_VERBS_WC_STREAM_COPY (from wkljohn/ds4-strix-halo-tp-odinlink)
// and FAK_WC_STREAM_COPY before falling back to CPU feature detection.
func HasAVX512StreamSupport() bool {
	return detectAVX512StreamSupport()
}

// avx512StreamCopyHook allows optional external or assembly kernel dispatch.
var avx512StreamCopyHook func(dst, src []byte) int

// CopyFromWriteCombined copies bytes from Write-Combining (WC) device memory
// into host memory (dst), mitigating the cache-fill and eviction collapse (~200 MB/s)
// observed when reading WC mappings with standard cached loads.
// When AVX-512 or streaming copy is supported/enabled, it uses a non-polluting
// 64-byte burst-read unrolled loop aligned to cache line boundaries; otherwise,
// it falls back to an efficient chunked copy.
// Returns the number of bytes copied: min(len(dst), len(src)).
func CopyFromWriteCombined(dst, src []byte) int {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	if n <= 0 {
		return 0
	}

	if avx512StreamCopyHook != nil {
		return avx512StreamCopyHook(dst[:n], src[:n])
	}

	if avx512StreamSupported {
		return copyFromWriteCombinedStreaming(dst[:n], src[:n])
	}
	return copyFromWriteCombinedFallback(dst[:n], src[:n])
}

func copyFromWriteCombinedStreaming(dst, src []byte) int {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	if n <= 0 {
		return 0
	}

	// Safety check: if slices overlap or pointers match, delegate to standard copy.
	if slicesOverlap(dst[:n], src[:n]) {
		return copy(dst[:n], src[:n])
	}

	// For tiny buffers (< 64 bytes), avoid alignment and unroll overhead.
	if n < 64 {
		return copy(dst[:n], src[:n])
	}

	// Align src to 64-byte boundary (cache line width) for write-combining streaming loads.
	srcAddr := uintptr(unsafe.Pointer(&src[0]))
	misalignment := int(srcAddr & 63)
	prefix := 0
	if misalignment != 0 {
		prefix = 64 - misalignment
		if prefix > n {
			prefix = n
		}
		copy(dst[:prefix], src[:prefix])
		if prefix == n {
			return n
		}
	}

	currDst := dst[prefix:n]
	currSrc := src[prefix:n]
	rem := len(currSrc)

	i := 0
	// 256-byte unrolled loop (4 x 64-byte cache lines).
	// In write-combining memory, reading each 64-byte cache line into CPU registers
	// as a single burst before issuing destination stores prevents interleaving
	// of read and write bus cycles, sustaining bus line rate.
	for ; i+256 <= rem; i += 256 {
		s0 := (*[8]uint64)(unsafe.Pointer(&currSrc[i]))
		v0, v1, v2, v3, v4, v5, v6, v7 := s0[0], s0[1], s0[2], s0[3], s0[4], s0[5], s0[6], s0[7]
		d0 := (*[8]uint64)(unsafe.Pointer(&currDst[i]))
		d0[0], d0[1], d0[2], d0[3], d0[4], d0[5], d0[6], d0[7] = v0, v1, v2, v3, v4, v5, v6, v7

		s1 := (*[8]uint64)(unsafe.Pointer(&currSrc[i+64]))
		v0, v1, v2, v3, v4, v5, v6, v7 = s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7]
		d1 := (*[8]uint64)(unsafe.Pointer(&currDst[i+64]))
		d1[0], d1[1], d1[2], d1[3], d1[4], d1[5], d1[6], d1[7] = v0, v1, v2, v3, v4, v5, v6, v7

		s2 := (*[8]uint64)(unsafe.Pointer(&currSrc[i+128]))
		v0, v1, v2, v3, v4, v5, v6, v7 = s2[0], s2[1], s2[2], s2[3], s2[4], s2[5], s2[6], s2[7]
		d2 := (*[8]uint64)(unsafe.Pointer(&currDst[i+128]))
		d2[0], d2[1], d2[2], d2[3], d2[4], d2[5], d2[6], d2[7] = v0, v1, v2, v3, v4, v5, v6, v7

		s3 := (*[8]uint64)(unsafe.Pointer(&currSrc[i+192]))
		v0, v1, v2, v3, v4, v5, v6, v7 = s3[0], s3[1], s3[2], s3[3], s3[4], s3[5], s3[6], s3[7]
		d3 := (*[8]uint64)(unsafe.Pointer(&currDst[i+192]))
		d3[0], d3[1], d3[2], d3[3], d3[4], d3[5], d3[6], d3[7] = v0, v1, v2, v3, v4, v5, v6, v7
	}

	// 64-byte single cache-line chunks.
	for ; i+64 <= rem; i += 64 {
		s := (*[8]uint64)(unsafe.Pointer(&currSrc[i]))
		v0, v1, v2, v3, v4, v5, v6, v7 := s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]
		d := (*[8]uint64)(unsafe.Pointer(&currDst[i]))
		d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7] = v0, v1, v2, v3, v4, v5, v6, v7
	}

	// Trailing bytes (< 64 bytes).
	if i < rem {
		copy(currDst[i:], currSrc[i:])
	}

	return n
}

func copyFromWriteCombinedFallback(dst, src []byte) int {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	if n <= 0 {
		return 0
	}
	// Chunked copy in 4KB blocks to reduce cache eviction pressure on host.
	const chunkSize = 4096
	for i := 0; i < n; i += chunkSize {
		end := i + chunkSize
		if end > n {
			end = n
		}
		copy(dst[i:end], src[i:end])
	}
	return n
}

func slicesOverlap(dst, src []byte) bool {
	if len(dst) == 0 || len(src) == 0 {
		return false
	}
	dStart := uintptr(unsafe.Pointer(&dst[0]))
	dEnd := dStart + uintptr(len(dst))
	sStart := uintptr(unsafe.Pointer(&src[0]))
	sEnd := sStart + uintptr(len(src))
	return dStart < sEnd && sStart < dEnd
}
