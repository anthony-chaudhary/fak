//go:build !windows

package serverlifecycle

import (
	"errors"
	"syscall"
)

func processDefinitelyGone(pid int) bool {
	if pid <= 0 {
		return false
	}
	return errors.Is(syscall.Kill(pid, 0), syscall.ESRCH)
}
