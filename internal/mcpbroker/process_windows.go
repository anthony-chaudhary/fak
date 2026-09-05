//go:build windows

package mcpbroker

import (
	"os/exec"
	"strconv"
	"syscall"
	"time"
)

// setProcessGroup configures the command to run in its own process group on Windows
// with console window creation suppressed.
func setProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= 0x08000200 // CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
	cmd.SysProcAttr.HideWindow = true
}

// terminateProcessTree terminates the child process and its process tree on Windows.
func terminateProcessTree(cmd *exec.Cmd, gracePeriod time.Duration) error {
	return killProcessTree(cmd)
}

func configureDispatchHelperCommand(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= 0x08000000 // CREATE_NO_WINDOW
	cmd.SysProcAttr.HideWindow = true
}

// killProcessTree terminates the child process tree on Windows using taskkill /PID <pid> /T /F
// with fallback to direct process termination.
func killProcessTree(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil || cmd.Process.Pid <= 0 {
		return nil
	}
	pid := cmd.Process.Pid
	killCmd := exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T", "/F")
	configureDispatchHelperCommand(killCmd)
	if err := killCmd.Run(); err != nil {
		return cmd.Process.Kill()
	}
	return nil
}
