//go:build windows

package dispatchaudit

import (
	"syscall"
	"unsafe"
)

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	const (
		processQueryLimitedInformation = 0x1000
		stillActive                    = 259
	)
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil || h == 0 {
		return false
	}
	defer syscall.CloseHandle(h)

	var code uint32
	r, _, _ := procGetExitCodeProcess.Call(uintptr(h), uintptr(unsafe.Pointer(&code)))
	if r == 0 {
		return true
	}
	return code == stillActive
}

var (
	modkernel32            = syscall.NewLazyDLL("kernel32.dll")
	procGetExitCodeProcess = modkernel32.NewProc("GetExitCodeProcess")
)
