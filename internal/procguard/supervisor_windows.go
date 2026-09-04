//go:build windows

package procguard

import (
	"os/exec"
	"syscall"
)

// getPGID returns pid on Windows where process groups do not follow POSIX pgid semantics.
func getPGID(pid int) int {
	return pid
}

// osDescendantPIDs returns nil on Windows where KillPID uses the native Job Object/Toolhelp32 tree reaper.
func osDescendantPIDs(root int) ([]int, string) {
	return nil, ""
}

// configureSysProcAttr sets Windows creation flags for headless execution without GUI popups or console windows.
func configureSysProcAttr(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	// CREATE_NO_WINDOW (0x08000000) | DETACHED_PROCESS (0x00000008)
	cmd.SysProcAttr.CreationFlags |= 0x08000000 | 0x00000008
}

// killProcessGroup is a no-op on Windows; KillPID handles tree termination natively.
func killProcessGroup(pgid int) error {
	return nil
}

// reapChildZombie is a no-op on Windows (no Unix zombie semantics).
func reapChildZombie(pid int) {}
