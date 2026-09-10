//go:build amd64

package strix

import (
	"os"
	"runtime"
	"strings"
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:noescape
func streamCopyAVX512Kernel(dst, src unsafe.Pointer, chunks int64)

//go:noescape
func streamCopyAVX512StoreOnly(dst, src unsafe.Pointer, chunks int64)

//go:noescape
func streamCopyAVX512LoadOnly(dst, src unsafe.Pointer, chunks int64)

//go:noescape
func sfenceAsm()

// canExecuteAVX512 checks whether the physical CPU actually supports AVX-512 execution.
// This is an absolute hardware safety gate: even if HasAVX512Streaming() is overridden
// or forced via environment variables, code will never jump to AVX-512 assembly
// instructions unless the hardware CPUID reports native AVX-512 support.
func canExecuteAVX512() bool {
	return cpu.X86.HasAVX512F || cpu.X86.HasAVX512
}

// hardwareHasAVX512 probes the physical hardware for AVX-512 support.
func hardwareHasAVX512() bool {
	if cpu.X86.HasAVX512F || cpu.X86.HasAVX512 {
		return true
	}
	if runtime.GOOS == "linux" {
		data, err := os.ReadFile("/proc/cpuinfo")
		if err == nil && strings.Contains(string(data), "avx512f") {
			return true
		}
	}
	return false
}
