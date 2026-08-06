//go:build !windows

package gitbroker

import (
	"errors"
	"os"
	"syscall"
)

// gitProcessGone reports whether pid names no live process. On POSIX signal 0
// is the liveness question: delivered means alive, ESRCH means gone, and EPERM
// means alive but owned by somebody else.
func gitProcessGone(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return true
	}
	err = p.Signal(syscall.Signal(0))
	switch {
	case err == nil, errors.Is(err, syscall.EPERM):
		return false
	default:
		return true
	}
}
