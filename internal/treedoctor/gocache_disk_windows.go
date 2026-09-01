//go:build windows

package treedoctor

import (
	"syscall"
	"unsafe"
)

var getDiskFreeSpaceExW = syscall.NewLazyDLL("kernel32.dll").NewProc("GetDiskFreeSpaceExW")

func GoCacheFreeBytes(path string) (int64, error) {
	ptr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	var available, total, free uint64
	r1, _, callErr := getDiskFreeSpaceExW.Call(uintptr(unsafe.Pointer(ptr)), uintptr(unsafe.Pointer(&available)), uintptr(unsafe.Pointer(&total)), uintptr(unsafe.Pointer(&free)))
	if r1 == 0 {
		if callErr != syscall.Errno(0) {
			return 0, callErr
		}
		return 0, syscall.EINVAL
	}
	return int64(available), nil
}
