//go:build linux || darwin

package harnessres

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// readProcSelf reads the current process's resource use on unix via Getrusage (CPU +
// peak RSS) plus /proc on Linux (current RSS + I/O bytes). Darwin gets CPU + peak RSS
// from Getrusage; its current-RSS / per-process-IO axes stay absent here (folded live
// by the per-PID sampler in #2048).
func readProcSelf() procSample {
	var s procSample
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err == nil {
		s.cpuUser = timevalDuration(ru.Utime)
		s.cpuSys = timevalDuration(ru.Stime)
		s.haveCPU = true
		if peak := maxrssBytes(ru.Maxrss); peak > 0 {
			s.peakRSS, s.havePeakRSS = peak, true
		}
	}
	if rss, ok := currentRSSBytes(); ok {
		s.rss, s.haveRSS = rss, true
	}
	if r, w, ok := selfIOBytes(); ok {
		s.ioRead, s.ioWrite, s.haveIO = r, w, true
	}
	return s
}

// foldChildRusage folds the reaped child's peak RSS from its Rusage (unix only). CPU
// is folded cross-platform in FoldChildExit via ProcessState.UserTime/SystemTime.
func foldChildRusage(h *Half, ps *os.ProcessState) {
	ru, ok := ps.SysUsage().(*syscall.Rusage)
	if !ok || ru == nil {
		return
	}
	if peak := maxrssBytes(ru.Maxrss); peak > 0 {
		h.PeakRSSBytes, h.HavePeakRSS = peak, true
	}
}

func timevalDuration(tv syscall.Timeval) time.Duration {
	return time.Duration(tv.Sec)*time.Second + time.Duration(tv.Usec)*time.Microsecond
}

// maxrssBytes normalizes Rusage.Maxrss to bytes: Linux reports KiB, darwin bytes.
func maxrssBytes(maxrss int64) uint64 {
	if maxrss <= 0 {
		return 0
	}
	if runtime.GOOS == "linux" {
		return uint64(maxrss) * 1024
	}
	return uint64(maxrss)
}

func currentRSSBytes() (uint64, bool) {
	if runtime.GOOS != "linux" {
		return 0, false
	}
	b, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0, false
	}
	fields := strings.Fields(string(b))
	if len(fields) < 2 {
		return 0, false
	}
	pages, err := strconv.ParseUint(fields[1], 10, 64) // field 2 = resident pages
	if err != nil {
		return 0, false
	}
	return pages * uint64(os.Getpagesize()), true
}

func selfIOBytes() (read, write uint64, ok bool) {
	if runtime.GOOS != "linux" {
		return 0, 0, false
	}
	b, err := os.ReadFile("/proc/self/io")
	if err != nil {
		return 0, 0, false
	}
	var haveR, haveW bool
	for _, line := range strings.Split(string(b), "\n") {
		key, val, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		n, err := strconv.ParseUint(strings.TrimSpace(val), 10, 64)
		if err != nil {
			continue
		}
		switch strings.TrimSpace(key) {
		case "read_bytes":
			read, haveR = n, true
		case "write_bytes":
			write, haveW = n, true
		}
	}
	return read, write, haveR && haveW
}
