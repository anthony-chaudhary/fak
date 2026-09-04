//go:build windows

package studylink

import (
	"os/exec"
	"syscall"
)

// ConfigureBackgroundCommand ensures spawned background processes on Windows
// do not allocate a visible console window (CREATE_NO_WINDOW | HideWindow).
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
