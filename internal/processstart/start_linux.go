//go:build linux

package processstart

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Start returns the /proc start time. Linux exposes start ticks in USER_HZ (100).
func Start(pid int) (time.Time, bool) {
	if pid <= 0 {
		return time.Time{}, false
	}
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return time.Time{}, false
	}
	end := strings.LastIndex(string(raw), ")")
	if end < 0 {
		return time.Time{}, false
	}
	fields := strings.Fields(string(raw)[end+1:])
	if len(fields) <= 19 {
		return time.Time{}, false
	}
	ticks, err := strconv.ParseInt(fields[19], 10, 64)
	if err != nil {
		return time.Time{}, false
	}
	stat, err := os.ReadFile("/proc/stat")
	if err != nil {
		return time.Time{}, false
	}
	var boot int64
	for _, line := range strings.Split(string(stat), "\n") {
		if strings.HasPrefix(line, "btime ") {
			boot, _ = strconv.ParseInt(strings.TrimSpace(strings.TrimPrefix(line, "btime ")), 10, 64)
			break
		}
	}
	if boot == 0 {
		return time.Time{}, false
	}
	return time.Unix(boot, 0).Add(time.Duration(ticks) * time.Second / 100).UTC(), true
}
