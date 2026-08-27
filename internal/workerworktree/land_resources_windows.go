//go:build windows

package workerworktree

import (
	"strings"
	"syscall"
	"time"
	"unsafe"
)

var (
	landKernel32             = syscall.NewLazyDLL("kernel32.dll")
	landPSAPI                = syscall.NewLazyDLL("psapi.dll")
	landGetCurrentProcess    = landKernel32.NewProc("GetCurrentProcess")
	landGetProcessTimes      = landKernel32.NewProc("GetProcessTimes")
	landGetProcessMemoryInfo = landPSAPI.NewProc("GetProcessMemoryInfo")
)

// processMemoryCounters is PROCESS_MEMORY_COUNTERS from psapi.h. SIZE_T fields
// are uintptr so the layout follows the running Windows architecture.
type processMemoryCounters struct {
	CB                         uint32
	PageFaultCount             uint32
	PeakWorkingSetSize         uintptr
	WorkingSetSize             uintptr
	QuotaPeakPagedPoolUsage    uintptr
	QuotaPagedPoolUsage        uintptr
	QuotaPeakNonPagedPoolUsage uintptr
	QuotaNonPagedPoolUsage     uintptr
	PagefileUsage              uintptr
	PeakPagefileUsage          uintptr
}

func filetimeDuration(kernel, user syscall.Filetime) time.Duration {
	kernelTicks := uint64(kernel.HighDateTime)<<32 | uint64(kernel.LowDateTime)
	userTicks := uint64(user.HighDateTime)<<32 | uint64(user.LowDateTime)
	return time.Duration(kernelTicks+userTicks) * 100 * time.Nanosecond
}

func currentLandResources() landResourceSample {
	handle, _, _ := landGetCurrentProcess.Call()
	var creation, exit, kernel, user syscall.Filetime
	timesOK, _, timesErr := landGetProcessTimes.Call(
		handle,
		uintptr(unsafe.Pointer(&creation)),
		uintptr(unsafe.Pointer(&exit)),
		uintptr(unsafe.Pointer(&kernel)),
		uintptr(unsafe.Pointer(&user)),
	)
	counters := processMemoryCounters{CB: uint32(unsafe.Sizeof(processMemoryCounters{}))}
	memoryOK, _, memoryErr := landGetProcessMemoryInfo.Call(
		handle,
		uintptr(unsafe.Pointer(&counters)),
		uintptr(counters.CB),
	)
	sample := landResourceSample{
		cpuAvailable: timesOK != 0,
		rssAvailable: memoryOK != 0,
	}
	if sample.cpuAvailable {
		sample.cpuTime = filetimeDuration(kernel, user)
	}
	if sample.rssAvailable {
		sample.peakRSSBytes = int64(counters.PeakWorkingSetSize)
	}
	var reasons []string
	if !sample.cpuAvailable {
		reasons = append(reasons, "GetProcessTimes failed: "+windowsCallError(timesErr))
	}
	if !sample.rssAvailable {
		reasons = append(reasons, "GetProcessMemoryInfo failed: "+windowsCallError(memoryErr))
	}
	sample.reason = strings.Join(reasons, "; ")
	return sample
}

func windowsCallError(err error) string {
	if err == nil || err == syscall.Errno(0) {
		return "no Windows error code"
	}
	return err.Error()
}
