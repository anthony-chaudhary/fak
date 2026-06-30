//go:build windows

package windowgate

import (
	"os/exec"
	"syscall"
)

// CreateNoWindow is the Windows process creation flag that prevents a console
// child spawned by a windowless parent from allocating a visible conhost window.
const CreateNoWindow = 0x08000000

// ConfigureBackgroundCommand marks cmd as a background/helper process. It must
// not create a user-visible console window when the parent has no console of its
// own, which is the common shape for scheduled fak maintenance tasks.
func ConfigureBackgroundCommand(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags |= CreateNoWindow
}
