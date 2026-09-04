//go:build windows

package procguard

import (
	"fmt"
	"syscall"
)

const (
	processSuspendResume = 0x0800
	statusSuccess        = 0x00000000
)

var (
	ntdllSuspend         = syscall.NewLazyDLL("ntdll.dll")
	procNtSuspendProcess = ntdllSuspend.NewProc("NtSuspendProcess")
	procNtResumeProcess  = ntdllSuspend.NewProc("NtResumeProcess")
)

func openProcessForSuspendResume(pid int) (syscall.Handle, error) {
	if pid <= 0 || pid == 4 {
		return 0, fmt.Errorf("invalid pid: %d", pid)
	}
	if !pidAliveNative(pid) {
		return 0, syscall.ESRCH
	}
	h, err := syscall.OpenProcess(processSuspendResume|processQueryLimitedInformation, false, uint32(pid))
	if err != nil || h == 0 {
		if !pidAliveNative(pid) {
			return 0, syscall.ESRCH
		}
		if err != nil {
			return 0, err
		}
		return 0, fmt.Errorf("open process %d: handle is 0", pid)
	}
	return h, nil
}

func suspendProcess(pid int) error {
	h, err := openProcessForSuspendResume(pid)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(h)

	r, _, _ := procNtSuspendProcess.Call(uintptr(h))
	if uint32(r) != statusSuccess {
		if !pidAliveNative(pid) {
			return syscall.ESRCH
		}
		return fmt.Errorf("NtSuspendProcess failed: NTSTATUS 0x%08X", uint32(r))
	}
	return nil
}

func resumeProcess(pid int) error {
	h, err := openProcessForSuspendResume(pid)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(h)

	r, _, _ := procNtResumeProcess.Call(uintptr(h))
	if uint32(r) != statusSuccess {
		if !pidAliveNative(pid) {
			return syscall.ESRCH
		}
		return fmt.Errorf("NtResumeProcess failed: NTSTATUS 0x%08X", uint32(r))
	}
	return nil
}
