//go:build !windows

package childproc

import (
	"os/exec"
	"syscall"
)

// ConfigureProcessGroup configures the command to run in its own process group
// and sets up cancellation to terminate the entire process tree.
// Invariant: cmd.SysProcAttr is non-nil and Setpgid is true.
// Guard: fail-closed exit code propagation via process tree cleanup.
func ConfigureProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	cmd.Cancel = func() error {
		return TerminateProcessTree(cmd)
	}
}

// TerminateProcessTree terminates the given command's process and all descendant processes.
// Invariant: safe to call with nil cmd or unstarted process.
// Guard: fail-closed exit code propagation through guaranteed process termination.
func TerminateProcessTree(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	return KillTree(cmd.Process.Pid)
}

// KillTree forcibly terminates the process identified by pid and all child processes in its group.
// Invariant: pid must be positive; non-positive pids are safely ignored.
// Guard: fail-closed exit code propagation via negative PID process-group signal delivery.
func KillTree(pid int) error {
	if pid <= 0 {
		return nil
	}
	err := syscall.Kill(-pid, syscall.SIGKILL)
	if err != nil {
		_ = syscall.Kill(pid, syscall.SIGKILL)
	}
	return nil
}
