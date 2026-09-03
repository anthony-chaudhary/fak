//go:build windows

package codedebt

import (
	"os/exec"
	"syscall"
)

func configureDispatchHelperCommand(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
}
