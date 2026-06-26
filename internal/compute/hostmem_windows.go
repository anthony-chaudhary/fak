//go:build windows

package compute

import (
	"syscall"
	"unsafe"
)

type memoryStatusEx struct {
	dwLength                uint32
	dwMemoryLoad            uint32
	ullTotalPhys            uint64
	ullAvailPhys            uint64
	ullTotalPageFile        uint64
	ullAvailPageFile        uint64
	ullTotalVirtual         uint64
	ullAvailVirtual         uint64
	ullAvailExtendedVirtual uint64
}

var (
	kernel32GlobalMemoryStatusEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")
)

func hostSystemMemory() (total, free int64, known bool) {
	var m memoryStatusEx
	m.dwLength = uint32(unsafe.Sizeof(m))
	ok, _, _ := kernel32GlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&m)))
	if ok == 0 || m.ullTotalPhys == 0 {
		return 0, FreeUnknown, false
	}
	return uint64ToCapInt64(m.ullTotalPhys), uint64ToCapInt64(m.ullAvailPhys), true
}
