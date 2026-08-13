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

// Access rights for the per-PID fleet reader. QUERY_LIMITED_INFORMATION is the
// narrowest right that answers GetProcessTimes and (since Windows 7) psapi's memory
// query; VM_READ is additionally needed for the memory counters. Neither grants any
// ability to write to, or terminate, the target — this rung only reads.
const (
	processQueryLimitedInformation = 0x1000
	processVMRead                  = 0x0010
)

// processMemoryCountersEx extends PROCESS_MEMORY_COUNTERS with PrivateUsage — the
// process's private committed bytes, i.e. what it costs that another copy of the same
// binary does NOT share. Resident/working set is the number an operator sees first;
// private is the number a pooling argument has to be made against (#6552).
type processMemoryCountersEx struct {
	base         processMemoryCounters
	privateUsage uintptr
}

// readProcPID reads ANOTHER process's CPU + working-set + private bytes, the per-PID
// half of the fleet walk (#6557). ok is false when the process cannot be opened at all
// (it exited between census and sample, or this token may not query it) — the caller
// reports that as `unreadable` rather than folding it in as a zero-cost process.
//
// It opens with the narrowest useful rights and degrades one step: without VM_READ the
// memory counters are refused but GetProcessTimes still answers, so a process that
// yields CPU-only is a real, partial reading rather than a lost one.
func readProcPID(pid int) (procSample, bool) {
	var s procSample
	if pid <= 0 {
		return s, false
	}
	h, err := syscall.OpenProcess(processQueryLimitedInformation|processVMRead, false, uint32(pid))
	if err != nil || h == 0 {
		h, err = syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
		if err != nil || h == 0 {
			return s, false
		}
	}
	defer syscall.CloseHandle(h)

	var creation, exit, kernelT, userT fileTime
	if ok, _, _ := procGetProcessTimes.Call(uintptr(h),
		uintptr(unsafe.Pointer(&creation)),
		uintptr(unsafe.Pointer(&exit)),
		uintptr(unsafe.Pointer(&kernelT)),
		uintptr(unsafe.Pointer(&userT))); ok != 0 {
		s.cpuUser = userT.duration()
		s.cpuSys = kernelT.duration()
		s.haveCPU = true
	}

	var mem processMemoryCountersEx
	mem.base.cb = uint32(unsafe.Sizeof(mem))
	if ok, _, _ := procGetProcessMemoryInfo.Call(uintptr(h),
		uintptr(unsafe.Pointer(&mem)), uintptr(mem.base.cb)); ok != 0 {
		s.rss, s.haveRSS = uint64(mem.base.workingSetSize), true
		s.peakRSS, s.havePeakRSS = uint64(mem.base.peakWorkingSetSize), true
		s.private, s.havePrivate = uint64(mem.privateUsage), true
	}
	return s, true
}
