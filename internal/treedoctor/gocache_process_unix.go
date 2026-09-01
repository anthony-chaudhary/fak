//go:build !windows

package treedoctor

import (
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
)

func ActiveGoBuild() (bool, error) {
	out, err := exec.Command("ps", "-axo", "comm=").Output()
	if err != nil {
		return false, fmt.Errorf("list processes: %w", err)
	}
	for _, line := range strings.Split(string(out), "\n") {
		name := strings.ToLower(filepath.Base(strings.TrimSpace(line)))
		switch name {
		case "go", "compile", "link", "asm", "cgo":
			return true, nil
		}
	}
	return false, nil
}

func goCacheProcessAlive(pid int) (bool, error) {
	if pid <= 0 {
		return false, fmt.Errorf("invalid pid %d", pid)
	}
	err := syscall.Kill(pid, 0)
	switch {
	case err == nil, errors.Is(err, syscall.EPERM):
		return true, nil
	case errors.Is(err, syscall.ESRCH):
		return false, nil
	default:
		return false, err
	}
}
