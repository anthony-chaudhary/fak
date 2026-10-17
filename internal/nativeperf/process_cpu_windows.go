//go:build windows

package nativeperf

import (
	"syscall"
	"time"
)

func readProcessCPUTime() (time.Duration, bool) {
	h, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0, false
	}
	var created, exited, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(h, &created, &exited, &kernel, &user); err != nil {
		return 0, false
	}
	ticks := func(f syscall.Filetime) int64 { return int64(f.HighDateTime)<<32 | int64(f.LowDateTime) }
	return time.Duration((ticks(kernel) + ticks(user)) * 100), true
}
