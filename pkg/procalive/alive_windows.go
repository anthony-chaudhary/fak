//go:build windows

package procalive

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

// stillActive is STILL_ACTIVE, which golang.org/x/sys/windows does not export.
const stillActive = 259

// Alive reports whether pid names a running process. Only ERROR_INVALID_PARAMETER
// (no such pid) or an exit code other than STILL_ACTIVE proves it gone; access
// denied and every other error read as alive.
func Alive(pid int) bool {
	if pid <= 0 {
		return false
	}
	if pid == os.Getpid() {
		return true
	}
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return !errors.Is(err, windows.ERROR_INVALID_PARAMETER)
	}
	defer windows.CloseHandle(h)
	var code uint32
	if err := windows.GetExitCodeProcess(h, &code); err != nil {
		return true
	}
	return code == stillActive
}
