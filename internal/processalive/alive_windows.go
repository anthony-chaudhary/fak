//go:build windows

// Package processalive provides the shared, no-spawn process liveness probe.
package processalive

import (
	"syscall"
	"unsafe"
)

const (
	processQueryLimitedInformation = 0x1000
	stillActive                    = 259
)

var (
	kernel32           = syscall.NewLazyDLL("kernel32.dll")
	getExitCodeProcess = kernel32.NewProc("GetExitCodeProcess")
)

// Check reports whether pid identifies a running process. If Windows allows the
// process handle but not its exit code, Check conservatively treats it as alive.
func Check(pid int) bool {
	if pid <= 0 {
		return false
	}
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil || h == 0 {
		return false
	}
	defer syscall.CloseHandle(h)

	var code uint32
	r, _, _ := getExitCodeProcess.Call(uintptr(h), uintptr(unsafe.Pointer(&code)))
	return r == 0 || code == stillActive
}
