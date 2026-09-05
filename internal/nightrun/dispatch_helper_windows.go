//go:build windows

package nightrun

import (
	"os/exec"
	"syscall"
)

func configureDispatchHelperCommand(cmd *exec.Cmd) {
	if cmd != nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	}
}
