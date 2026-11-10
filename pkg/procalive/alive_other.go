//go:build !windows

package procalive

import (
	"errors"
	"os"
	"syscall"
)

// Alive reports whether pid names a running process. EPERM means the process
// exists under another user, so it reads as alive.
func Alive(pid int) bool {
	if pid <= 0 {
		return false
	}
	if pid == os.Getpid() {
		return true
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	defer proc.Release()
	err = proc.Signal(syscall.Signal(0))
	return err == nil || errors.Is(err, syscall.EPERM)
}
