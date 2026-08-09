//go:build !windows

package safecommit

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// procClockTicksPerSecond is the USER_HZ that /proc/<pid>/stat's starttime field is
// denominated in. It is 100 on every Linux ABI Go targets — the kernel reports process
// times to userspace in USER_HZ whatever CONFIG_HZ is built as — and reading it properly
// needs sysconf(_SC_CLK_TCK), which is unavailable without cgo. A wrong divisor could only
// shift the derived start time, and both the caller's skew grace and its fail-safe
// direction absorb that: a mis-stated start time at worst declines to prove PID reuse.
const procClockTicksPerSecond = 100

// processStartTime resolves when the process at pid started, reporting ok=false when that
// cannot be established. It is the identity half of the PID-reuse guard (issue #5892): an
// image NAME can only ever say "this looks like something a committer might run", which on
// a fleet host is nearly every process, whereas a start time can PROVE the process at a
// recycled PID is not the one that took the lock.
//
// Linux derives it from procfs: field 22 of /proc/<pid>/stat is the process's start time in
// clock ticks since boot, and /proc/stat's btime is the boot wall-clock. Platforms without
// procfs (macOS, the BSDs) yield ok=false, which leaves the caller on the pre-existing image
// heuristic rather than inventing a verdict — the same direction processImageName already
// takes there.
func processStartTime(pid int) (time.Time, bool) {
	if pid <= 0 {
		return time.Time{}, false
	}
	stat, err := os.ReadFile("/proc/" + itoa(pid) + "/stat")
	if err != nil {
		return time.Time{}, false // no procfs, or the process is already gone
	}
	ticks, ok := procStartTicks(string(stat))
	if !ok {
		return time.Time{}, false
	}
	boot, ok := procBootTime()
	if !ok {
		return time.Time{}, false
	}
	// Split the whole seconds from the remainder rather than multiplying ticks by a
	// nanosecond scale: ticks is unbounded in uptime, and ticks*1e9 overflows int64 well
	// inside plausible host lifetimes.
	whole := ticks / procClockTicksPerSecond
	frac := ticks % procClockTicksPerSecond
	return boot.Add(time.Duration(whole)*time.Second +
		time.Duration(frac)*time.Second/procClockTicksPerSecond), true
}

// procStartTicks extracts field 22 (starttime, in clock ticks since boot) from a
// /proc/<pid>/stat body, or ok=false when the body is not shaped like one.
//
// Field 2 (comm) is the reason this cannot be a plain Fields() index: it is parenthesised
// and carries the executable's own name, which may contain both spaces and parentheses. The
// only safe anchor is the LAST ')' in the record — every field after it is
// whitespace-separated, starting with field 3 (state).
func procStartTicks(stat string) (int64, bool) {
	end := strings.LastIndexByte(stat, ')')
	if end < 0 {
		return 0, false
	}
	fields := strings.Fields(stat[end+1:])
	const startTimeIndex = 22 - 3 // field 22, counting field 3 as index 0
	if len(fields) <= startTimeIndex {
		return 0, false
	}
	ticks, err := strconv.ParseInt(fields[startTimeIndex], 10, 64)
	if err != nil || ticks < 0 {
		return 0, false
	}
	return ticks, true
}

// procBootTime reads the kernel's boot wall-clock from /proc/stat's btime line, or
// ok=false when it is absent or unparseable.
func procBootTime() (time.Time, bool) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return time.Time{}, false
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		rest, ok := strings.CutPrefix(line, "btime ")
		if !ok {
			continue
		}
		secs, err := strconv.ParseInt(strings.TrimSpace(rest), 10, 64)
		if err != nil || secs <= 0 {
			return time.Time{}, false
		}
		return time.Unix(secs, 0), true
	}
	return time.Time{}, false
}
