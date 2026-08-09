//go:build windows

package gitbroker

import (
	"syscall"
	"unsafe"
)

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

// killGitProcessTree terminates pid AND every process it launched, which on
// Windows is what "kill this git" has to mean.
//
// TerminateProcess reaches exactly one process, and for git that is often not
// the one doing the work: the `git` every non-Git-Bash shell finds first on
// PATH (`C:\Program Files\Git\cmd\git.exe`, ~46 KB) is a launcher that re-execs
// the real `git.exe` (~4 MB, under `mingw64\bin`) as a CHILD, handing it the
// inherited stdio handles. Kill only the launcher and the real
// `cat-file --batch` keeps reading our stdin pipe and answering correctly — so
// a test that kills the pool "out from under" the caller would not have killed
// the pool at all. Git Bash puts `mingw64\bin` first and so hits the worker
// directly, which is why this is invisible from that one shell and live
// everywhere else on the same host.
//
// The POSIX build of this helper is a single kill: there is no launcher shim
// there, so the process this package started IS the worker.
func killGitProcessTree(pid int) error {
	for _, child := range gitChildPIDs(pid) {
		_ = killGitProcessTree(child)
	}
	return terminateGitPID(pid)
}

// gitChildPIDs lists the direct children of pid from a Toolhelp32 snapshot. It
// is taken BEFORE pid is terminated, so a recycled pid cannot make an unrelated
// process look like a descendant.
func gitChildPIDs(pid int) []int {
	h, err := syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return nil
	}
	defer syscall.CloseHandle(h)

	var e syscall.ProcessEntry32
	e.Size = uint32(unsafe.Sizeof(e))
	if err := syscall.Process32First(h, &e); err != nil {
		return nil
	}
	var kids []int
	for {
		if int(e.ParentProcessID) == pid && int(e.ProcessID) != pid {
			kids = append(kids, int(e.ProcessID))
		}
		if err := syscall.Process32Next(h, &e); err != nil {
			break // ERROR_NO_MORE_FILES ends the walk
		}
	}
	return kids
}

// terminateGitPID kills one process by pid. A pid that can no longer be opened
// has already exited, which is success for a caller that wants it gone.
func terminateGitPID(pid int) error {
	h, err := syscall.OpenProcess(syscall.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		return nil
	}
	defer syscall.CloseHandle(h)
	return syscall.TerminateProcess(h, 1)
}
