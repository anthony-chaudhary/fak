//go:build linux

package main

import "syscall"

// armParentDeathSignal uses SIGKILL, not a catchable signal: the parent is
// already gone when it fires, so there is nothing left to shut down gracefully.
func armParentDeathSignal() error {
	if _, _, errno := syscall.RawSyscall6(syscall.SYS_PRCTL, syscall.PR_SET_PDEATHSIG, uintptr(syscall.SIGKILL), 0, 0, 0, 0); errno != 0 {
		return errno
	}
	return nil
}
