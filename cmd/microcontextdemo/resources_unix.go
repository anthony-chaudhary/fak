//go:build unix

package main

import (
	"syscall"
	"time"
)

type processCPU struct {
	user   time.Duration
	system time.Duration
	ok     bool
}

func readProcessCPU() processCPU {
	var r syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &r); err != nil {
		return processCPU{}
	}
	return processCPU{
		user:   time.Duration(r.Utime.Sec)*time.Second + time.Duration(r.Utime.Usec)*time.Microsecond,
		system: time.Duration(r.Stime.Sec)*time.Second + time.Duration(r.Stime.Usec)*time.Microsecond,
		ok:     true,
	}
}
