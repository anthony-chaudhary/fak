//go:build !windows

package procguard

import "syscall"

// killSignal sends SIGKILL on POSIX. The Windows reaper goes through taskkill in
// KillPID, so this file is POSIX-only.
func killSignal(pid int) error {
	return syscall.Kill(pid, syscall.SIGKILL)
}
