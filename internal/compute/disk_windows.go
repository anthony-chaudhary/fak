//go:build windows

package compute

import (
	"syscall"
	"unsafe"
)

var (
	kernel32GetDiskFreeSpaceEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GetDiskFreeSpaceExW")
)

// diskInfo reports disk total/free bytes for path on Windows using GetDiskFreeSpaceEx.
func diskInfo(path string) (total, free int64, known bool) {
	pathPtr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, FreeUnknown, false
	}
	var freeBytes, totalBytes, availBytes uint64
	ok, _, _ := kernel32GetDiskFreeSpaceEx.Call(uintptr(unsafe.Pointer(pathPtr)),
		uintptr(unsafe.Pointer(&freeBytes)),
		uintptr(unsafe.Pointer(&totalBytes)),
		uintptr(unsafe.Pointer(&availBytes)))
	if ok == 0 || totalBytes == 0 {
		return 0, FreeUnknown, false
	}
	return uint64ToCapInt64(totalBytes), uint64ToCapInt64(freeBytes), true
}