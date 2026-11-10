//go:build windows

package procalive

import (
	"time"

	"golang.org/x/sys/windows"
)

// StartTime returns pid's creation time. ok is false when the process is gone
// or its times cannot be read; a caller must then fall back to Alive alone.
func StartTime(pid int) (time.Time, bool) {
	if pid <= 0 {
		return time.Time{}, false
	}
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return time.Time{}, false
	}
	defer windows.CloseHandle(h)
	var code uint32
	if err := windows.GetExitCodeProcess(h, &code); err != nil || code != stillActive {
		return time.Time{}, false
	}
	var creation, exit, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(h, &creation, &exit, &kernel, &user); err != nil || (creation.HighDateTime == 0 && creation.LowDateTime == 0) {
		return time.Time{}, false
	}
	return time.Unix(0, creation.Nanoseconds()), true
}
