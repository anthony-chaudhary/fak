//go:build !windows

package procguard

import (
	"os/exec"
	"syscall"
)

// ConfigureProcessTreeCancel puts cmd in its own process group and wires Cancel
// to kill that whole group. It is for unattended launchers whose direct child may
// spawn grandchildren that must not survive a timeout.
func ConfigureProcessTreeCancel(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
			return cmd.Process.Kill()
		}
		return nil
	}
}
