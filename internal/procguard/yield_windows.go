//go:build windows

package procguard

import (
	"syscall"
)

var (
	psapiYield          = syscall.NewLazyDLL("psapi.dll")
	procEmptyWorkingSet = psapiYield.NewProc("EmptyWorkingSet")
)

const (
	processQueryInformation = 0x0400
)

func emptyWorkingSet(h syscall.Handle) bool {
	if h == 0 {
		return false
	}
	r, _, _ := procEmptyWorkingSet.Call(uintptr(h))
	return r != 0
}

func yieldWorkingSets(pids ...int) {
	if h, err := syscall.GetCurrentProcess(); err == nil && h != 0 {
		emptyWorkingSet(h)
	}
	for _, pid := range pids {
		if pid <= 0 {
			continue
		}
		h, err := syscall.OpenProcess(processSetQuota|processQueryLimitedInformation, false, uint32(pid))
		if err != nil || h == 0 {
			h, err = syscall.OpenProcess(processSetQuota|processQueryInformation, false, uint32(pid))
		}
		if err != nil || h == 0 {
			continue
		}
		emptyWorkingSet(h)
		_ = syscall.CloseHandle(h)
	}
}
