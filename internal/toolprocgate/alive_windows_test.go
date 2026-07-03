//go:build windows

package toolprocgate

import (
	"syscall"
	"unsafe"
)

// pidAlive reports whether a process with the given pid is currently running.
// It is the OS-liveness oracle the descendant-containment witness polls: it must
// answer from the kernel's own process table, never from a self-report.
//
// On Windows, OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) succeeds for a live
// process and fails for one that has exited. A still-open handle to a zombie
// (exited but not yet reaped) reports a known exit code, so we additionally check
// GetExitCodeProcess: STILL_ACTIVE (259) means running, anything else means the
// process is gone. A taskkill /F termination yields a non-259 exit code, so a
// reaped grandchild reads as dead here. Any error resolving the pid is treated as
// "not alive" — a pid we cannot confirm is live is reported dead.
func pidAlive(pid int) bool {
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
	r, _, _ := procGetExitCodeProcessAlive.Call(uintptr(h), uintptr(unsafe.Pointer(&code)))
	if r == 0 {
		// Could not read the exit code; conservatively treat the handle's
		// existence as "alive" rather than assert a kill we are unsure about.
		return true
	}
	return code == stillActive
}

var (
	modkernel32Alive            = syscall.NewLazyDLL("kernel32.dll")
	procGetExitCodeProcessAlive = modkernel32Alive.NewProc("GetExitCodeProcess")
)
