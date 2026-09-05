//go:build windows

package harnessres

import (
	"syscall"
	"time"
	"unsafe"
)

var (
	k32GovDll                   = syscall.NewLazyDLL("kernel32.dll")
	procGlobalMemoryStatusExGov = k32GovDll.NewProc("GlobalMemoryStatusEx")
)

type memoryStatusExGov struct {
	cbSize               uint32
	memoryLoad           uint32
	totalPhys            uint64
	availPhys            uint64
	totalPageFile        uint64
	availPageFile        uint64
	totalVirtual         uint64
	availVirtual         uint64
	availExtendedVirtual uint64
}

func init() {
	platformHostSampleReader = readHostSampleWindows
}

func readHostSampleWindows() (HostSample, bool) {
	var s HostSample
	s.Timestamp = time.Now()
	var ms memoryStatusExGov
	ms.cbSize = uint32(unsafe.Sizeof(ms))
	r1, _, _ := procGlobalMemoryStatusExGov.Call(uintptr(unsafe.Pointer(&ms)))
	if r1 == 0 || ms.totalPhys == 0 {
		return s, false
	}
	s.TotalRAMBytes = ms.totalPhys
	s.AvailRAMBytes = ms.availPhys
	s.HaveRAM = true
	s.TotalSwapBytes = ms.totalPageFile
	s.AvailSwapBytes = ms.availPageFile
	s.HaveSwap = true
	s.HavePSI = false
	return s, true
}
