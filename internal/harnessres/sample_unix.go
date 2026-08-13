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

// clockTicksPerSecond is the USER_HZ the /proc/<pid>/stat CPU fields are counted in.
// The kernel fixes it at 100 for userspace on every Linux port fak runs on (it is the
// value glibc's sysconf(_SC_CLK_TCK) returns); there is no stdlib reader for it, and
// guessing a different constant would silently scale every fleet CPU number.
const clockTicksPerSecond = 100

// readProcPID reads ANOTHER process's CPU + resident + private bytes for the fleet walk
// (#6557), the per-PID sibling of readProcSelf. ok is false when the process cannot be
// read at all — it exited between census and sample, or this platform has no reader —
// which the caller reports as `unreadable` rather than folding in as a free process.
//
// Linux only: the numbers come from /proc/<pid>/{stat,statm}. Darwin exposes no
// equivalent per-PID view to the stdlib (its answer is libproc, which is cgo), so it
// reports unreadable rather than a fabricated zero.
func readProcPID(pid int) (procSample, bool) {
	var s procSample
	if runtime.GOOS != "linux" || pid <= 0 {
		return s, false
	}
	dir := "/proc/" + strconv.Itoa(pid)
	stat, err := os.ReadFile(dir + "/stat")
	if err != nil {
		return s, false
	}
	if u, k, ok := parseProcStatCPU(string(stat)); ok {
		s.cpuUser, s.cpuSys, s.haveCPU = u, k, true
	}
	if b, err := os.ReadFile(dir + "/statm"); err == nil {
		if rss, private, ok := parseProcStatm(string(b), uint64(os.Getpagesize())); ok {
			s.rss, s.haveRSS = rss, true
			s.private, s.havePrivate = private, true
		}
	}
	return s, true
}

// parseProcStatCPU pulls utime/stime out of a /proc/<pid>/stat line. The comm field is
// parenthesized and may itself contain spaces and parens, so the scan starts after the
// LAST ')' — splitting the whole line on spaces is the classic way to misread this file
// for any process whose name contains one.
func parseProcStatCPU(line string) (user, sys time.Duration, ok bool) {
	close := strings.LastIndex(line, ")")
	if close < 0 {
		return 0, 0, false
	}
	// After the comm field the next token is state (field 3), so field N is at index N-3.
	fields := strings.Fields(line[close+1:])
	const utimeIdx, stimeIdx = 14 - 3, 15 - 3
	if len(fields) <= stimeIdx {
		return 0, 0, false
	}
	ut, err1 := strconv.ParseUint(fields[utimeIdx], 10, 64)
	st, err2 := strconv.ParseUint(fields[stimeIdx], 10, 64)
	if err1 != nil || err2 != nil {
		return 0, 0, false
	}
	tick := time.Second / clockTicksPerSecond
	return time.Duration(ut) * tick, time.Duration(st) * tick, true
}

// parseProcStatm folds /proc/<pid>/statm into resident and private bytes. Field 2 is
// resident pages and field 3 is the shared (file-backed) subset of them, so
// resident-minus-shared is the process's private cost — the number that does not
// double-count pages N copies of one binary already share.
func parseProcStatm(line string, pageSize uint64) (rss, private uint64, ok bool) {
	fields := strings.Fields(line)
	if len(fields) < 3 || pageSize == 0 {
		return 0, 0, false
	}
	resident, err1 := strconv.ParseUint(fields[1], 10, 64)
	shared, err2 := strconv.ParseUint(fields[2], 10, 64)
	if err1 != nil || err2 != nil {
		return 0, 0, false
	}
	if shared > resident {
		shared = resident
	}
	return resident * pageSize, (resident - shared) * pageSize, true
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
