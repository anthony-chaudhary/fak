//go:build !windows

package toolprocgate

import "syscall"

// pidAlive reports whether a process with the given pid is currently running.
// It is the OS-liveness oracle the descendant-containment witness polls: it must
// answer from the kernel, never from a self-report.
//
// On POSIX, signal 0 performs the kernel's existence + permission check without
// delivering a signal: nil means the process exists and we may signal it, EPERM
// means it exists but is owned by another user (still alive). ESRCH (no such
// process) is the dead verdict. A SIGKILL-reaped grandchild reads as dead here.
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || err == syscall.EPERM
}
