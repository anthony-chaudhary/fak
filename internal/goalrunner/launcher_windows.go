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
	h, err := syscall.OpenProcess(syscall.PROCESS_QUERY_INFORMATION, false, uint32(pid))
	if err != nil {
		h, err = syscall.OpenProcess(0x1000, false, uint32(pid)) // PROCESS_QUERY_LIMITED_INFORMATION
		if err != nil {
			return false
		}
	}
	defer syscall.CloseHandle(h)

	var code uint32
	if err := syscall.GetExitCodeProcess(h, &code); err != nil {
		return false
	}
	return code == stillActive
}
