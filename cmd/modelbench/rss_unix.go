//go:build linux || darwin

package main

import (
	"runtime"
	"syscall"
)

func peakRSSBytes() (uint64, error) {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0, err
	}
	rss := uint64(ru.Maxrss)
	if runtime.GOOS == "linux" {
		rss *= 1024
	}
	return rss, nil
}
