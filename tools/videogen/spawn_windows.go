//go:build windows

package main

import (
	"os/exec"
	"syscall"
)

// configureDispatchSpawn prevents the facade's nested Go tool from opening a
// fresh console when videogen runs under a windowless agent or hook process.
func configureDispatchSpawn(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
}
