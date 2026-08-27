//go:build linux || darwin || freebsd || netbsd || openbsd

package workerworktree

import (
	"runtime"
	"syscall"
	"time"
)

func currentLandResources() landResourceSample {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return landResourceSample{reason: "getrusage failed: " + err.Error()}
	}
	peak := int64(usage.Maxrss)
	if runtime.GOOS != "darwin" {
		peak *= 1024 // Linux and the supported BSDs report KiB; Darwin reports bytes.
	}
	return landResourceSample{
		cpuTime:      time.Duration(syscall.TimevalToNsec(usage.Utime) + syscall.TimevalToNsec(usage.Stime)),
		peakRSSBytes: peak,
		cpuAvailable: true,
		rssAvailable: true,
	}
}
