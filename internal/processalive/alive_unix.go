//go:build !windows

// Package processalive provides the shared, no-spawn process liveness probe.
package processalive

import "syscall"

// Check reports whether pid identifies a running process.
func Check(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || err == syscall.EPERM
}
