//go:build windows

package sessionjournal

import (
	"os/exec"
	"syscall"
)

func hideAppendHelperWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
}
