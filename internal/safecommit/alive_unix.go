//go:build !windows

package safecommit

import "syscall"

// processAlive reports whether a process with the given pid is currently running.
// signal 0 probes the process without delivering anything: nil means the process
// exists, ESRCH means it is gone, and EPERM (a live process we do not own) still
// confirms existence.
func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	if err == nil {
		return true
	}
	return err == syscall.EPERM
}
