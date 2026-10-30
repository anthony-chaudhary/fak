//go:build linux || darwin || freebsd || netbsd || openbsd

package workerworktree

import (
	"runtime"
	"syscall"
	"time"
)

func currentLandResources() landResourceSample {
	sample := landResourceSample{}
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		sample.reason = "getrusage failed: " + err.Error()
	} else {
		peak := int64(usage.Maxrss)
		if runtime.GOOS != "darwin" {
			peak *= 1024 // Linux and the supported BSDs report KiB; Darwin reports bytes.
		}
		sample.cpuTime = time.Duration(syscall.TimevalToNsec(usage.Utime) + syscall.TimevalToNsec(usage.Stime))
		sample.peakRSSBytes = peak
		sample.cpuAvailable, sample.rssAvailable = true, true
	}
	// Other Unix child-accounting semantics are not qualified here. Keep their
	// self metrics, but do not invent available child axes from zero values.
	if runtime.GOOS != "linux" {
		sample.childReason = "waited-child resource sampling is unsupported on " + runtime.GOOS
		return sample
	}
	var children syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_CHILDREN, &children); err != nil {
		sample.childReason = "getrusage children failed: " + err.Error()
		return sample
	}
	// Linux getrusage(2): waited children (and recursively waited descendants),
	// with Maxrss in KiB for the largest child, not the concurrent tree peak.
	// https://man7.org/linux/man-pages/man2/getrusage.2.html
	sample.childCPUTime = time.Duration(syscall.TimevalToNsec(children.Utime) + syscall.TimevalToNsec(children.Stime))
	sample.childPeakRSSBytes = int64(children.Maxrss) * 1024
	sample.childCPUAvailable, sample.childRSSAvailable = true, true
	return sample
}
