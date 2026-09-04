//go:build !windows

package procguard

import (
	"fmt"
	"syscall"
)

func suspendProcess(pid int) error {
	if pid <= 0 {
		return fmt.Errorf("invalid pid: %d", pid)
	}
	return syscall.Kill(pid, syscall.SIGSTOP)
}

func resumeProcess(pid int) error {
	if pid <= 0 {
		return fmt.Errorf("invalid pid: %d", pid)
	}
	return syscall.Kill(pid, syscall.SIGCONT)
}
