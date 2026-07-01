//go:build windows

package harnessres

import (
	"os"
	"syscall"
	"time"
	"unsafe"
)

// Windows resource readers via the kernel32/psapi LazyDLL idiom the repo already uses
// (see cmd/modelbench/rss_windows.go): GetProcessTimes for CPU, GetProcessMemoryInfo
// for working-set + peak working-set, GetProcessIoCounters for I/O transfer bytes.
var (
	modKernel32              = syscall.NewLazyDLL("kernel32.dll")
	modPsapi                 = syscall.NewLazyDLL("psapi.dll")
	procGetCurrentProcess    = modKernel32.NewProc("GetCurrentProcess")
	procGetProcessTimes      = modKernel32.NewProc("GetProcessTimes")
	procGetProcessIoCounters = modKernel32.NewProc("GetProcessIoCounters")
	procGetProcessMemoryInfo = modPsapi.NewProc("GetProcessMemoryInfo")
)

// fileTime mirrors the Win32 FILETIME: a 64-bit count of 100-ns intervals split
// across two 32-bit words. For GetProcessTimes the kernel/user fields are elapsed
// CPU time (not a wall-clock date).
type fileTime struct {
	low  uint32
	high uint32
}

func (f fileTime) duration() time.Duration {
	ticks := uint64(f.high)<<32 | uint64(f.low)
	return time.Duration(ticks) * 100 * time.Nanosecond
}

// ioCounters mirrors Win32 IO_COUNTERS (six ULONGLONG fields). ReadTransferCount /
// WriteTransferCount are byte totals (they include file, pipe, and device I/O).
type ioCounters struct {
	readOps    uint64
	writeOps   uint64
	otherOps   uint64
	readBytes  uint64
	writeBytes uint64
	otherBytes uint64
}

// processMemoryCounters mirrors Win32 PROCESS_MEMORY_COUNTERS (matches rss_windows.go).
type processMemoryCounters struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
}

func readProcSelf() procSample {
	var s procSample
	handle, _, _ := procGetCurrentProcess.Call()

	var creation, exit, kernelT, userT fileTime
	if ok, _, _ := procGetProcessTimes.Call(handle,
		uintptr(unsafe.Pointer(&creation)),
		uintptr(unsafe.Pointer(&exit)),
		uintptr(unsafe.Pointer(&kernelT)),
		uintptr(unsafe.Pointer(&userT))); ok != 0 {
		s.cpuUser = userT.duration()
		s.cpuSys = kernelT.duration()
		s.haveCPU = true
	}

	var mem processMemoryCounters
	mem.cb = uint32(unsafe.Sizeof(mem))
	if ok, _, _ := procGetProcessMemoryInfo.Call(handle,
		uintptr(unsafe.Pointer(&mem)), uintptr(mem.cb)); ok != 0 {
		s.rss, s.haveRSS = uint64(mem.workingSetSize), true
		s.peakRSS, s.havePeakRSS = uint64(mem.peakWorkingSetSize), true
	}

	var io ioCounters
	if ok, _, _ := procGetProcessIoCounters.Call(handle,
		uintptr(unsafe.Pointer(&io))); ok != 0 {
		s.ioRead, s.ioWrite, s.haveIO = io.readBytes, io.writeBytes, true
	}
	return s
}

// foldChildRusage is a no-op on Windows: a reaped child exposes no Maxrss. Child CPU
// is folded cross-platform via ProcessState.UserTime/SystemTime; child peak RSS is
// covered by the continuous per-PID sampler (#2048), not the exit state.
func foldChildRusage(h *Half, ps *os.ProcessState) {}
