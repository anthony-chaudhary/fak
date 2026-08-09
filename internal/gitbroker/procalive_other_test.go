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

// killGitProcessTree terminates the git this package started. Here that is one
// kill, because `git` on PATH is the git binary itself: POSIX has no equivalent
// of the Git-for-Windows launcher `.exe` that re-execs the real git as a child
// and survives its parent's death, which is why the Windows build of this
// helper has to walk descendants. See procalive_windows_test.go.
func killGitProcessTree(pid int) error {
	p, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return p.Kill()
}
