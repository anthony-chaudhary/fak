//go:build darwin

package processstart

import (
	"errors"
	"time"

	"golang.org/x/sys/unix"
)

// Start returns the kernel-recorded process start time for pid. Reading the
// kern.proc.pid record directly avoids a helper process whose own lifecycle
// could otherwise be mistaken for the target's identity.
func Start(pid int) (time.Time, bool) {
	return darwinProcessStart(pid, readDarwinProcessInfo)
}

func readDarwinProcessInfo(pid int) (darwinProcessInfo, error) {
	info, err := unix.SysctlKinfoProc("kern.proc.pid", pid)
	if err != nil {
		return darwinProcessInfo{}, err
	}
	if info == nil {
		return darwinProcessInfo{}, errors.New("kern.proc.pid returned no record")
	}
	return darwinProcessInfo{
		pid:       int64(info.Proc.P_pid),
		startSec:  info.Proc.P_starttime.Sec,
		startUsec: int64(info.Proc.P_starttime.Usec),
	}, nil
}
