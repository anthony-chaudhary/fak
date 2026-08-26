//go:build windows

package hostdiag

import (
	"syscall"
	"time"
	"unsafe"
)

const processQueryLimitedInformation = 0x1000

var queryFullProcessImageNameW = syscall.NewLazyDLL("kernel32.dll").NewProc("QueryFullProcessImageNameW")

func processImage(pid int) (string, bool) {
	if pid <= 0 {
		return "", false
	}
	handle, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil || handle == 0 {
		return "", false
	}
	defer syscall.CloseHandle(handle)
	buffer := make([]uint16, 32768)
	size := uint32(len(buffer))
	result, _, _ := queryFullProcessImageNameW.Call(uintptr(handle), 0, uintptr(unsafe.Pointer(&buffer[0])), uintptr(unsafe.Pointer(&size)))
	if result == 0 || size == 0 {
		return "", false
	}
	return syscall.UTF16ToString(buffer[:size]), true
}

func processStartedAt(pid int) (time.Time, bool) {
	if pid <= 0 {
		return time.Time{}, false
	}
	handle, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil || handle == 0 {
		return time.Time{}, false
	}
	defer syscall.CloseHandle(handle)
	var creation, exit, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(handle, &creation, &exit, &kernel, &user); err != nil {
		return time.Time{}, false
	}
	ns := creation.Nanoseconds()
	if ns <= 0 {
		return time.Time{}, false
	}
	return time.Unix(0, ns), true
}
