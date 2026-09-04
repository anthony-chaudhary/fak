//go:build windows

package childproc

import (
	"os"
	"os/exec"
	"strconv"
	"syscall"
)

// ConfigureProcessGroup configures the command to run in its own process group
// and sets up cancellation to terminate the entire process tree.
// Invariant: cmd.SysProcAttr is non-nil and includes CREATE_NEW_PROCESS_GROUP.
// Guard: fail-closed exit code propagation via process tree cleanup.
func ConfigureProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP
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

// KillTree forcibly terminates the process identified by pid and all child processes.
// Invariant: pid must be positive; non-positive pids are safely ignored.
// Guard: fail-closed exit code propagation via forced termination.
func KillTree(pid int) error {
	if pid <= 0 {
		return nil
	}
	// On Windows, taskkill.exe /T (tree) /F (force) /PID <pid> tears down the entire process tree.
	killCmd := exec.Command("taskkill.exe", "/T", "/F", "/PID", strconv.Itoa(pid))
	ConfigureBackgroundCommand(killCmd)
	_ = killCmd.Run()

	if p, err := os.FindProcess(pid); err == nil {
		_ = p.Kill()
	}
	return nil
}

// ConfigureBackgroundCommand suppresses visible console window creation on Windows.
func ConfigureBackgroundCommand(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags |= 0x08000000
}
