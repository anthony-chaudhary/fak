//go:build !windows

package sessionjournal

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// BootTime returns the machine's last boot instant and the source. On Linux it reads the
// exact boot epoch from /proc/stat's "btime <unix-seconds>" line; on any other OS (macOS,
// etc.) it degrades to a zero time and "unknown", which makes Classify skip the
// MACHINE_REBOOT verdict and fall back to the PID / stale-beat signals — never a false
// crash. The now argument is unused here (the kernel already holds the boot epoch);
// Windows derives it from now-uptime. A sysctl kern.boottime path for macOS is a follow-on.
func BootTime(now time.Time) (time.Time, string) {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return time.Time{}, "unknown"
	}
	for _, line := range strings.Split(string(b), "\n") {
		if rest, ok := strings.CutPrefix(line, "btime "); ok {
			if secs, err := strconv.ParseInt(strings.TrimSpace(rest), 10, 64); err == nil && secs > 0 {
				return time.Unix(secs, 0).UTC(), "proc-stat-btime"
			}
		}
	}
	return time.Time{}, "unknown"
}
