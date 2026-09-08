//go:build !windows

package goalrunner

import (
	"errors"
	"fmt"
	"os"
	"strings"
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
	if err != nil {
		return errors.Is(err, syscall.EPERM)
	}
	// On Linux, a killed child whose parent hasn't reaped it remains in the process
	// table as a zombie ('Z' or 'X'). A zombie is not live.
	if data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid)); err == nil {
		str := string(data)
		if idx := strings.LastIndex(str, ")"); idx >= 0 && idx+2 < len(str) {
			state := str[idx+2]
			if state == 'Z' || state == 'X' {
				return false
			}
		}
	}
	return true
}
