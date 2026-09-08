//go:build windows

package sysproc

import (
	"os/exec"
	"syscall"
)

// Windows process creation flags for console suppression and process groups.
const (
	CreateNoWindow        = 0x08000000
	CreateNewProcessGroup = 0x00000200
	DetachedProcess       = 0x00000008
)

// ConfigureBackground configures cmd to run in the background without creating
// a visible console window on Windows.
func ConfigureBackground(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags |= CreateNoWindow
}

// ConfigureDetached configures cmd to run as a detached process without any console
// attached. CreateNoWindow is cleared because Windows ignores CREATE_NO_WINDOW
// when DETACHED_PROCESS is set.
func ConfigureDetached(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags &^= CreateNoWindow
	cmd.SysProcAttr.CreationFlags |= DetachedProcess
}

// ConfigureProcessGroup configures cmd to run in the background as the root of a new
// process group on Windows.
func ConfigureProcessGroup(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	ConfigureBackground(cmd)
	cmd.SysProcAttr.CreationFlags |= CreateNewProcessGroup
}
