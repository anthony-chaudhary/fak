//go:build windows

package main

import (
	"os/exec"
	"syscall"
	"unsafe"
)

func configureDispatchSpawn(cmd *exec.Cmd) {
	configureDispatchHelperCommand(cmd)
}

// dispatchPIDAlive reports whether a pid is currently running — WITHOUT spawning
// a process. It previously shelled `tasklist /FI` once per call, and the dispatch
// tick calls it 50-80 times per tick across its scan loops (livescan, preflight,
// witness). On a multi-session box that per-pid `tasklist` fan-out is a dominant
// process-spawn storm — each cold `tasklist` start is a burst of soft faults,
// context switches and syscalls that stalls the whole machine while CPU/RAM read
// low (see `fak stallscan`). Replacing it with the same
// OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)+GetExitCodeProcess syscall pair
// that internal/safecommit and internal/dispatchaudit already use turns each of
// those 50-80 spawns per tick into a cheap in-process syscall.
//
// OpenProcess succeeds for a live pid and fails once it has exited. A still-open
// handle to a not-yet-reaped zombie reports its exit code, so we also check
// GetExitCodeProcess: STILL_ACTIVE (259) means running. Any error resolving the
// pid is treated as "not alive" — a pid we cannot confirm live must not keep a
// worker slot or lock wedged.
func dispatchPIDAlive(pid int) bool {
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
	r, _, _ := procDispatchGetExitCodeProcess.Call(uintptr(h), uintptr(unsafe.Pointer(&code)))
	if r == 0 {
		// Could not read the exit code; conservatively treat the open handle as
		// "alive" rather than free a slot we are unsure about.
		return true
	}
	return code == stillActive
}

var (
	modDispatchKernel32            = syscall.NewLazyDLL("kernel32.dll")
	procDispatchGetExitCodeProcess = modDispatchKernel32.NewProc("GetExitCodeProcess")
)
