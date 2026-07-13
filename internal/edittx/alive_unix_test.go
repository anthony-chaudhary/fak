//go:build !windows

package edittx

import (
	"fmt"
	"os"
	"strings"
	"syscall"
)

// pidAlive reports whether a process with the given pid is currently running. It is
// the OS-liveness oracle the descendant-containment witness (#3108) polls: it must
// answer from the kernel, never from a self-report.
//
// On POSIX, signal 0 performs the kernel's existence + permission check without
// delivering a signal: nil means the process exists and we may signal it, EPERM means
// it exists but is owned by another user (still alive). ESRCH (no such process) is the
// dead verdict. A SIGKILL-reaped grandchild reads as dead here. On Linux, a still-
// present zombie/dead entry in /proc is treated as dead so a not-yet-reaped exit does
// not read as alive.
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	if procState, ok := linuxProcState(pid); ok {
		return procState != "Z" && procState != "X"
	}
	err := syscall.Kill(pid, 0)
	return err == nil || err == syscall.EPERM
}

func linuxProcState(pid int) (string, bool) {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return "", false
	}
	text := string(raw)
	closeParen := strings.LastIndex(text, ")")
	if closeParen < 0 || closeParen+2 >= len(text) {
		return "", false
	}
	return text[closeParen+2 : closeParen+3], true
}
