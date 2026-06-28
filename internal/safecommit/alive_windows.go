//go:build windows

package safecommit

import (
	"syscall"
	"unsafe"
)

// processAlive reports whether a process with the given pid is currently running.
//
// On Windows, OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) succeeds for a live
// process and fails for one that has exited. A still-open handle to a zombie (exited
// but not yet reaped) reports a known exit code, so we additionally check
// GetExitCodeProcess: STILL_ACTIVE (259) means running, anything else means the holder
// is gone. Any error resolving the pid is treated as "not alive" — a pid we cannot
// confirm is live must not keep a stale commit lock wedged.
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
		// Could not read the exit code; conservatively treat the handle's existence as
		// "alive" rather than reap a lock we are unsure about.
		return true
	}
	return code == stillActive
}

var (
	modkernel32            = syscall.NewLazyDLL("kernel32.dll")
	procGetExitCodeProcess = modkernel32.NewProc("GetExitCodeProcess")
)
