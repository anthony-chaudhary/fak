//go:build windows

package selfupdatecmd

import (
	"os/exec"
	"syscall"
)

func configureSelfUpdateSuccessor(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}
