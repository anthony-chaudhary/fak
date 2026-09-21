//go:build !windows

package sysproc

import (
	"os/exec"
	"syscall"
)

// ConfigureDurableChild configures cmd so the child outlives its launcher. Off
// Windows the POSIX setsid already detaches the child from the launcher's
// controlling terminal and session, so this is the same configuration as
// ConfigureDetached and carries no per-child console cost (fak#13468).
func ConfigureDurableChild(cmd *exec.Cmd) {
	ConfigureDetached(cmd)
}

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
