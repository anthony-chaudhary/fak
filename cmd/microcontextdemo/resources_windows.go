//go:build windows

package main

import (
	"time"

	"golang.org/x/sys/windows"
)

type processCPU struct {
	user   time.Duration
	system time.Duration
	ok     bool
}

func readProcessCPU() processCPU {
	var created, exited, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(windows.CurrentProcess(), &created, &exited, &kernel, &user); err != nil {
		return processCPU{}
	}
	return processCPU{
		user:   time.Duration(user.Nanoseconds()),
		system: time.Duration(kernel.Nanoseconds()),
		ok:     true,
	}
}
