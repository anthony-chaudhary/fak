//go:build linux || darwin || freebsd || netbsd || openbsd || dragonfly

package nativeperf

import (
	"syscall"
	"time"
)

func readProcessCPUTime() (time.Duration, bool) {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0, false
	}
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano()), true
}
