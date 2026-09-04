//go:build linux

package config

import (
	"os"
	"strconv"
	"strings"
	"syscall"
)

// nofileLimit returns the current soft RLIMIT_NOFILE (max open files).
// Returns 0 on error.
func nofileLimit() int {
	var rlim syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rlim); err != nil {
		return 0
	}
	return int(rlim.Cur)
}

// systemMemoryGB reads total system RAM from /proc/meminfo and returns GB.
// Returns 0 on error.
func systemMemoryGB() float64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && strings.TrimSuffix(fields[0], ":") == "MemTotal" {
			kb, err := strconv.ParseInt(fields[1], 10, 64)
			if err != nil {
				return 0
			}
			return float64(kb) / (1024 * 1024)
		}
	}
	return 0
}
