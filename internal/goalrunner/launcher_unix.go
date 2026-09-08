//go:build !windows

package goalrunner

import (
	"errors"
	"syscall"
)

func detachedSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{
		Setsid: true,
	}
}

func isProcessLive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	if err == nil {
		return true
	}
	if errors.Is(err, syscall.EPERM) {
		return true
	}
	return false
}
