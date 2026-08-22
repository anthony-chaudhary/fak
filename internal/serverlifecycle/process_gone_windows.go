//go:build windows

package serverlifecycle

import (
	"errors"
	"syscall"
	"unsafe"
)

const (
	processQueryLimitedInformation = 0x1000
	processStillActive             = 259
	processNotFound                = syscall.Errno(87)
)

var getExitCodeProcessProc = syscall.NewLazyDLL("kernel32.dll").NewProc("GetExitCodeProcess")

func processDefinitelyGone(pid int) bool {
	if pid <= 0 {
		return false
	}
	handle, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil || handle == 0 {
		return errors.Is(err, processNotFound)
	}
	defer syscall.CloseHandle(handle)

	var code uint32
	result, _, _ := getExitCodeProcessProc.Call(uintptr(handle), uintptr(unsafe.Pointer(&code)))
	return result != 0 && code != processStillActive
}
