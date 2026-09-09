//go:build darwin

package sessionjournal

import (
	"time"

	"golang.org/x/sys/unix"
)

// BootTime returns the machine's last boot instant and the source on macOS.
// It queries sysctl kern.boottime directly from the kernel.
func BootTime(now time.Time) (time.Time, string) {
	tv, err := unix.SysctlTimeval("kern.boottime")
	if err != nil || tv == nil || tv.Sec <= 0 {
		return time.Time{}, "unknown"
	}
	return time.Unix(int64(tv.Sec), int64(tv.Usec)*1000).UTC(), "sysctl-kern-boottime"
}
