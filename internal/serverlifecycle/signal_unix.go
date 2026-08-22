//go:build !windows

package serverlifecycle

import (
	"os"
	"syscall"
)

func signalProcess(pid int) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return process.Signal(syscall.SIGTERM)
}
