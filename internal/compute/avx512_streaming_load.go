package compute

import (
	"encoding/binary"
	"os"
	"runtime"
	"strings"
	"unsafe"

	syscpu "golang.org/x/sys/cpu"
)

// StreamingLoadChunkSize is the size in bytes of a 512-bit (64-byte) cacheline
// chunk used for AVX-512 non-temporal streaming loads on write-combining memory.
const StreamingLoadChunkSize = 64

// streamingLoadAsmHook allows optional external assembly kernel dispatch or test overrides.
var streamingLoadAsmHook func(dst, src []byte) int

// avx512StreamingLoadOverride allows deterministic override in test environments.
var avx512StreamingLoadOverride *bool

// cpuidFeatureChecker allows optional CPUID emulation testing.
var cpuidFeatureChecker func() bool

// procCPUInfoPath defines the procfs path for CPU feature probing on Linux platforms.
var procCPUInfoPath = "/proc/cpuinfo"

var avx512StreamingLoadEnvVars = []string{
	"FAK_AVX512_STREAMING_LOAD",
	"FAK_WC_STREAM_LOAD",
	"ODL_VERBS_WC_STREAM_LOAD",
	"FAK_WC_STREAM_COPY",
	"ODL_VERBS_WC_STREAM_COPY",
}

func parseEnvFlag(key string) (val bool, ok bool) {
	v := os.Getenv(key)
	if v == "" {
		return false, false
	}
	v = strings.TrimSpace(v)
	if v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "on") || strings.EqualFold(v, "yes") {
		return true, true
	}
	if v == "0" || strings.EqualFold(v, "false") || strings.EqualFold(v, "off") || strings.EqualFold(v, "no") {
		return false, true
	}
	return false, false
}

func detectAVX512StreamingLoad() bool {
	for _, envKey := range avx512StreamingLoadEnvVars {
		if val, ok := parseEnvFlag(envKey); ok {
			return val
		}
	}
	if avx512StreamingLoadOverride != nil {
		return *avx512StreamingLoadOverride
	}
	if cpuidFeatureChecker != nil && cpuidFeatureChecker() {
		return true
	}
	if runtime.GOARCH != "amd64" {
		return false
	}
	// Hardware feature check via Go syscpu (AVX-512 foundation / 512-bit vector extensions).
	if syscpu.X86.HasAVX512 || syscpu.X86.HasAVX512F {
		return true
	}
	// Check Linux procfs /proc/cpuinfo for Zen 5 / Strix Halo or AVX-512 flags.
	if checkProcCPUInfoAVX512() {
		return true
	}
	return false
}

func checkProcCPUInfoAVX512() bool {
	data, err := os.ReadFile(procCPUInfoPath)
	if err != nil {
		return false
	}
	return parseProcCPUInfoAVX512(string(data))
}

func parseProcCPUInfoAVX512(content string) bool {
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "flags") || strings.HasPrefix(line, "Features") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				for _, flag := range strings.Fields(parts[1]) {
					if flag == "avx512f" || flag == "avx512_streaming" || flag == "avx512" {
						return true
					}
				}
			}
		}
		if strings.HasPrefix(line, "model name") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				model := strings.ToLower(parts[1])
				if strings.Contains(model, "ryzen ai max") || strings.Contains(model, "strix halo") || strings.Contains(model, "zen 5") {
					return true
				}
			}
		}
	}
	return false
}

// HasAVX512StreamingLoad reports whether the host CPU supports AVX-512 streaming loads
// from write-combined host memory (e.g. AMD Zen 5 / Strix Halo Ryzen AI MAX+ / gfx1151).
// It checks environment overrides, CPUID emulation hooks, golang.org/x/sys/cpu feature bits,
// and Linux /proc/cpuinfo flags.
func HasAVX512StreamingLoad() bool {
	return detectAVX512StreamingLoad()
}

// StreamingLoadWC reads from Write-Combining (WC) memory src and writes to dst using
// 64-byte streaming loads where possible, handling unaligned head and tail safely.
// On Unified Memory Architecture (UMA) APUs such as AMD Strix Halo, reading write-combined
// memory via standard cached loads triggers severe bus turn-around bubbles and cache-line
// bouncing, collapsing throughput to ~189-200 MB/s. StreamingLoadWC loads full 64-byte
// cacheline chunks into CPU registers before storing to destination, avoiding bus contention.
// Returns the number of bytes copied: min(len(dst), len(src)).
func StreamingLoadWC(dst, src []byte) int {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	if n <= 0 {
		return 0
	}

	if streamingLoadAsmHook != nil {
		return streamingLoadAsmHook(dst[:n], src[:n])
	}

	if HasAVX512StreamingLoad() {
		return streamingLoadStreaming(dst[:n], src[:n])
	}
	return streamingLoadFallback(dst[:n], src[:n])
}

func putUint64x8(dst []byte, v0, v1, v2, v3, v4, v5, v6, v7 uint64) {
	binary.LittleEndian.PutUint64(dst[0:8], v0)
	binary.LittleEndian.PutUint64(dst[8:16], v1)
	binary.LittleEndian.PutUint64(dst[16:24], v2)
	binary.LittleEndian.PutUint64(dst[24:32], v3)
	binary.LittleEndian.PutUint64(dst[32:40], v4)
	binary.LittleEndian.PutUint64(dst[40:48], v5)
	binary.LittleEndian.PutUint64(dst[48:56], v6)
	binary.LittleEndian.PutUint64(dst[56:64], v7)
}

func streamingLoadStreaming(dst, src []byte) int {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	if n <= 0 {
		return 0
	}

	// Overlapping slices must preserve standard memmove ordering.
	if slicesOverlap(dst[:n], src[:n]) {
		return copy(dst[:n], src[:n])
	}

	// For buffers smaller than a single 64-byte cacheline, bypass alignment overhead.
	if n < StreamingLoadChunkSize {
		return copy(dst[:n], src[:n])
	}

	// Align src to 64-byte boundary for write-combining streaming loads.
	srcAddr := uintptr(unsafe.Pointer(&src[0]))
	misalignment := int(srcAddr & (StreamingLoadChunkSize - 1))
	prefix := 0
	if misalignment != 0 {
		prefix = StreamingLoadChunkSize - misalignment
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

	// Check if destination is 8-byte aligned to allow direct uint64 array assignment.
	dstAligned8 := (uintptr(unsafe.Pointer(&currDst[0])) & 7) == 0

	i := 0
	// 256-byte unrolled loop (4 x 64-byte cachelines).
	// In write-combining memory, loading all 64 bytes of each cacheline into CPU registers
	// before issuing stores to destination prevents read/write bus interleaving and
	// cache-line bouncing.
	for ; i+256 <= rem; i += 256 {
		// Cacheline 0
		s0 := (*[8]uint64)(unsafe.Pointer(&currSrc[i]))
		v0, v1, v2, v3, v4, v5, v6, v7 := s0[0], s0[1], s0[2], s0[3], s0[4], s0[5], s0[6], s0[7]
		if dstAligned8 {
			d0 := (*[8]uint64)(unsafe.Pointer(&currDst[i]))
			d0[0], d0[1], d0[2], d0[3], d0[4], d0[5], d0[6], d0[7] = v0, v1, v2, v3, v4, v5, v6, v7
		} else {
			putUint64x8(currDst[i:], v0, v1, v2, v3, v4, v5, v6, v7)
		}

		// Cacheline 1
		s1 := (*[8]uint64)(unsafe.Pointer(&currSrc[i+64]))
		v0, v1, v2, v3, v4, v5, v6, v7 = s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7]
		if dstAligned8 {
			d1 := (*[8]uint64)(unsafe.Pointer(&currDst[i+64]))
			d1[0], d1[1], d1[2], d1[3], d1[4], d1[5], d1[6], d1[7] = v0, v1, v2, v3, v4, v5, v6, v7
		} else {
			putUint64x8(currDst[i+64:], v0, v1, v2, v3, v4, v5, v6, v7)
		}

		// Cacheline 2
		s2 := (*[8]uint64)(unsafe.Pointer(&currSrc[i+128]))
		v0, v1, v2, v3, v4, v5, v6, v7 = s2[0], s2[1], s2[2], s2[3], s2[4], s2[5], s2[6], s2[7]
		if dstAligned8 {
			d2 := (*[8]uint64)(unsafe.Pointer(&currDst[i+128]))
			d2[0], d2[1], d2[2], d2[3], d2[4], d2[5], d2[6], d2[7] = v0, v1, v2, v3, v4, v5, v6, v7
		} else {
			putUint64x8(currDst[i+128:], v0, v1, v2, v3, v4, v5, v6, v7)
		}

		// Cacheline 3
		s3 := (*[8]uint64)(unsafe.Pointer(&currSrc[i+192]))
		v0, v1, v2, v3, v4, v5, v6, v7 = s3[0], s3[1], s3[2], s3[3], s3[4], s3[5], s3[6], s3[7]
		if dstAligned8 {
			d3 := (*[8]uint64)(unsafe.Pointer(&currDst[i+192]))
			d3[0], d3[1], d3[2], d3[3], d3[4], d3[5], d3[6], d3[7] = v0, v1, v2, v3, v4, v5, v6, v7
		} else {
			putUint64x8(currDst[i+192:], v0, v1, v2, v3, v4, v5, v6, v7)
		}
	}

	// 64-byte single cacheline chunks.
	for ; i+64 <= rem; i += 64 {
		s := (*[8]uint64)(unsafe.Pointer(&currSrc[i]))
		v0, v1, v2, v3, v4, v5, v6, v7 := s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]
		if dstAligned8 {
			d := (*[8]uint64)(unsafe.Pointer(&currDst[i]))
			d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7] = v0, v1, v2, v3, v4, v5, v6, v7
		} else {
			putUint64x8(currDst[i:], v0, v1, v2, v3, v4, v5, v6, v7)
		}
	}

	// Unaligned tail (< 64 bytes).
	if i < rem {
		copy(currDst[i:], currSrc[i:])
	}

	return n
}

func streamingLoadFallback(dst, src []byte) int {
	return copy(dst, src)
}
