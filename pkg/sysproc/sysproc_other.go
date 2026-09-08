//go:build !windows

package sysproc

import (
	"os/exec"
	"syscall"
)

// ConfigureBackground is a no-op on non-Windows platforms.
func ConfigureBackground(_ *exec.Cmd) {}

// ConfigureDetached configures cmd to create a new session (setsid) on Unix-like systems.
func ConfigureDetached(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setsid = true
}

// ConfigureProcessGroup configures cmd to create a new process group (setpgid) on Unix-like systems.
func ConfigureProcessGroup(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}
