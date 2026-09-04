//go:build !windows

package procguard

import (
	"os/exec"
	"syscall"
	"time"
)

// getPGID returns the process group ID for pid on POSIX systems,
// excluding the supervisor's own process group to prevent friendly fire.
func getPGID(pid int) int {
	if pid <= 0 {
		return 0
	}
	if pgid, err := syscall.Getpgid(pid); err == nil {
		if pgid <= 1 || pgid == syscall.Getpgrp() || pgid == syscall.Getpid() {
			return 0
		}
		return pgid
	}
	return 0
}

// osDescendantPIDs returns descendant PIDs using the POSIX relation census.
func osDescendantPIDs(root int) ([]int, string) {
	return descendantPIDs(root)
}

// configureSysProcAttr sets POSIX process group attributes.
func configureSysProcAttr(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// killProcessGroup sends SIGKILL to the POSIX process group.
func killProcessGroup(pgid int) error {
	if pgid <= 1 || pgid == syscall.Getpgrp() || pgid == syscall.Getpid() {
		return nil
	}
	return syscall.Kill(-pgid, syscall.SIGKILL)
}

// reapChildZombie attempts to reap zombie process status if pid is an immediate child.
func reapChildZombie(pid int) {
	if pid <= 0 {
		return
	}
	var wstatus syscall.WaitStatus
	for i := 0; i < 20; i++ {
		wpid, err := syscall.Wait4(pid, &wstatus, syscall.WNOHANG, nil)
		if wpid == pid || err != nil {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
}
