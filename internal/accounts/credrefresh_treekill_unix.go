//go:build !windows

package accounts

import (
	"os/exec"
	"syscall"
)

// setRefreshSysProcAttr puts the refresh spawn in its own process group so the
// deadline kill can reap claude's whole subtree with one group signal. It mirrors
// procguard.ConfigureProcessTreeCancel, which this package cannot import:
// procguard -> dispatchtick -> accounts is a committed import cycle (#3105).
func setRefreshSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// refreshKillTree SIGKILLs pid's process group — the whole claude/node subtree,
// since setRefreshSysProcAttr made the spawn its own group leader — falling back
// to the single pid when no such group exists.
func refreshKillTree(pid int) error {
	if pid <= 0 {
		return nil
	}
	if err := syscall.Kill(-pid, syscall.SIGKILL); err == nil {
		return nil
	}
	return syscall.Kill(pid, syscall.SIGKILL)
}
