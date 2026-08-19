//go:build windows

package codetools

import (
	"fmt"
	"os/exec"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func configureProcessTree(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		killCmd := exec.Command("taskkill.exe", "/T", "/F", "/PID", fmt.Sprint(cmd.Process.Pid))
		windowgate.ConfigureBackgroundCommand(killCmd)
		_ = killCmd.Run()
		return cmd.Process.Kill()
	}
	cmd.WaitDelay = 2 * time.Second
}
