//go:build windows

package main

import (
	"os/exec"
	"syscall"
)

const dispatchCreateNoWindow = 0x08000000

func configureDispatchHelperCommand(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= dispatchCreateNoWindow
}
