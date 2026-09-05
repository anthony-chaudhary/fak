//go:build !windows

package mcpbroker

import (
	"os/exec"
	"syscall"
	"time"
)

// setProcessGroup configures the command to run in its own process group on POSIX systems.
func setProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

func configureDispatchHelperCommand(_ *exec.Cmd) {}

// terminateProcessTree terminates the child process and its process group.
// It issues SIGTERM first, allows up to gracePeriod for the process to exit cleanly,
// and issues SIGKILL if it has not exited when the grace period expires.
func terminateProcessTree(cmd *exec.Cmd, gracePeriod time.Duration) error {
	if cmd == nil || cmd.Process == nil || cmd.Process.Pid <= 0 {
		return nil
	}

	pid := cmd.Process.Pid
	pgid := -pid

	// Try process group SIGTERM first, falling back to direct process signal.
	if err := syscall.Kill(pgid, syscall.SIGTERM); err != nil {
		_ = syscall.Kill(pid, syscall.SIGTERM)
	}

	// Poll for exit within grace period.
	deadline := time.Now().Add(gracePeriod)
	for time.Now().Before(deadline) {
		// syscall.Kill with signal 0 checks if process still exists.
		if err := syscall.Kill(pid, 0); err != nil {
			// Process has exited
			return nil
		}
		time.Sleep(25 * time.Millisecond)
	}

	// Grace period expired without exit: force termination with SIGKILL
	if err := syscall.Kill(pgid, syscall.SIGKILL); err != nil {
		_ = syscall.Kill(pid, syscall.SIGKILL)
	}

	return nil
}

// killProcessTree immediately sends SIGKILL to the process group.
func killProcessTree(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil || cmd.Process.Pid <= 0 {
		return nil
	}
	pid := cmd.Process.Pid
	pgid := -pid
	if err := syscall.Kill(pgid, syscall.SIGKILL); err != nil {
		return syscall.Kill(pid, syscall.SIGKILL)
	}
	return nil
}
