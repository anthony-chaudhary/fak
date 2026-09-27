//go:build darwin

package main

import (
	"os"
	"os/exec"
	"syscall"
)

// spawnUpServiceLapseWaker starts the `fak up off --for` waker as its own
// session so it outlives the terminal (and a `go run`) that ran `off`. It
// returns the waker pid; the waker logs to logPath.
func spawnUpServiceLapseWaker(exe string, args []string, logPath string) (int, error) {
	logf, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return 0, err
	}
	defer logf.Close()
	cmd := exec.Command(exe, args...)
	cmd.Stdout = logf
	cmd.Stderr = logf
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	pid := cmd.Process.Pid
	_ = cmd.Process.Release()
	return pid, nil
}
