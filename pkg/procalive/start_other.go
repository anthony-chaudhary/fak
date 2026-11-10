//go:build !windows

package procalive

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// linuxUserHZ is USER_HZ, the unit of /proc/<pid>/stat starttime.
const linuxUserHZ = 100

// StartTime returns pid's start time. ok is false when the process is gone,
// the platform is not Linux, or /proc cannot be parsed; a caller must then
// fall back to Alive alone.
func StartTime(pid int) (time.Time, bool) {
	if runtime.GOOS != "linux" || pid <= 0 {
		return time.Time{}, false
	}
	data, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return time.Time{}, false
	}
	stat := string(data)
	end := strings.LastIndexByte(stat, ')')
	if end < 0 {
		return time.Time{}, false
	}
	fields := strings.Fields(stat[end+1:])
	if len(fields) < 20 || fields[0] == "Z" || fields[0] == "X" {
		return time.Time{}, false
	}
	ticks, err := strconv.ParseInt(fields[19], 10, 64)
	if err != nil {
		return time.Time{}, false
	}
	boot, ok := linuxBootTime()
	if !ok {
		return time.Time{}, false
	}
	return boot.Add(time.Duration(ticks) * time.Second / linuxUserHZ), true
}

func linuxBootTime() (time.Time, bool) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return time.Time{}, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		if rest, ok := strings.CutPrefix(line, "btime "); ok {
			sec, err := strconv.ParseInt(strings.TrimSpace(rest), 10, 64)
			if err != nil || sec <= 0 {
				return time.Time{}, false
			}
			return time.Unix(sec, 0), true
		}
	}
	return time.Time{}, false
}
