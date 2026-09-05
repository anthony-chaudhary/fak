//go:build windows

// Package processalive provides the shared, no-spawn process liveness probe.
package processalive

import (
	"errors"
	"syscall"
	"unsafe"
)

const (
	processQueryLimitedInformation = 0x1000
	stillActive                    = 259
	// ERROR_INVALID_PARAMETER is the Win32 error code 87 returned by OpenProcess
	// when the target PID does not exist.
	ERROR_INVALID_PARAMETER = syscall.Errno(87)
)

var (
	kernel32           = syscall.NewLazyDLL("kernel32.dll")
	getExitCodeProcess = kernel32.NewProc("GetExitCodeProcess")
	openProcess        = syscall.OpenProcess
)

// Check reports whether pid identifies a running process. If Windows allows the
// process handle but not its exit code, or if the process owner is inaccessible
// or indeterminate, Check conservatively treats it as alive.
func Check(pid int) bool {
	if pid <= 0 {
		return false
	}
	h, err := openProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil {
		if errors.Is(err, ERROR_INVALID_PARAMETER) {
			return false
		}
		return true
	}
	if h == 0 {
		return false
	}
	defer syscall.CloseHandle(h)

	var code uint32
	r, _, _ := getExitCodeProcess.Call(uintptr(h), uintptr(unsafe.Pointer(&code)))
	if r == 0 || code == stillActive {
		return true
	}
	return false
}
