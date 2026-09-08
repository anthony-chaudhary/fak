//go:build windows

package goalrunner

import "syscall"

func detachedSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{
		CreationFlags: 0x08000200, // CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
		HideWindow:    true,
	}
}

const stillActive = 259

func isProcessLive(pid int) bool {
	if pid <= 0 {
		return false
	}
	const access = 0x1000 | 0x00100000 // PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE
	h, err := syscall.OpenProcess(access, false, uint32(pid))
	if err != nil {
		h, err = syscall.OpenProcess(syscall.PROCESS_QUERY_INFORMATION, false, uint32(pid))
		if err != nil {
			return false
		}
	}
	defer syscall.CloseHandle(h)

	event, err := syscall.WaitForSingleObject(h, 0)
	if err == nil {
		return event == syscall.WAIT_TIMEOUT
	}

	var code uint32
	if err := syscall.GetExitCodeProcess(h, &code); err != nil {
		return false
	}
	return code == stillActive
}
