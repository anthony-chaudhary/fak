package main

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	kernel32                 = syscall.NewLazyDLL("kernel32.dll")
	psapi                    = syscall.NewLazyDLL("psapi.dll")
	procGetCurrentProcess    = kernel32.NewProc("GetCurrentProcess")
	procGetProcessMemoryInfo = psapi.NewProc("GetProcessMemoryInfo")
)

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

func peakRSSBytes() (uint64, error) {
	handle, _, _ := procGetCurrentProcess.Call()
	var counters processMemoryCounters
	counters.cb = uint32(unsafe.Sizeof(counters))
	ok, _, err := procGetProcessMemoryInfo.Call(
		handle,
		uintptr(unsafe.Pointer(&counters)),
		uintptr(counters.cb),
	)
	if ok == 0 {
		if err != syscall.Errno(0) {
			return 0, err
		}
		return 0, fmt.Errorf("GetProcessMemoryInfo failed")
	}
	return uint64(counters.peakWorkingSetSize), nil
}
