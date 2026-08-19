//go:build windows

package processstart

import (
	"syscall"
	"time"
)

const processQueryLimitedInformation = 0x1000

// Start returns the kernel creation time for pid.
func Start(pid int) (time.Time, bool) {
	if pid <= 0 {
		return time.Time{}, false
	}
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil || h == 0 {
		return time.Time{}, false
	}
	defer syscall.CloseHandle(h)
	var creation, exit, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(h, &creation, &exit, &kernel, &user); err != nil {
		return time.Time{}, false
	}
	return time.Unix(0, creation.Nanoseconds()).UTC(), true
}
