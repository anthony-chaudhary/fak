//go:build windows

package gitbroker

import "syscall"

// stillActive is GetExitCodeProcess's "has not exited yet" sentinel.
const stillActive = 259

// gitProcessGone reports whether pid names no live process.
//
// The portable-looking os.FindProcess + Signal(0) idiom is WRONG on Windows and
// silently so: os.FindProcess succeeds, and (*os.Process).Signal returns
// ErrProcessDone only when this process already waited on that child. For a
// grandchild we never waited on, it returns EWINDOWS whether the process is
// alive or long gone — so the idiom reports "alive" forever and an orphan test
// built on it can never fail. Ask the OS directly instead.
func gitProcessGone(pid int) bool {
	h, err := syscall.OpenProcess(syscall.PROCESS_QUERY_INFORMATION, false, uint32(pid))
	if err != nil {
		// The pid cannot be opened at all: it names nothing.
		return true
	}
	defer syscall.CloseHandle(h)
	var code uint32
	if err := syscall.GetExitCodeProcess(h, &code); err != nil {
		return true
	}
	return code != stillActive
}
