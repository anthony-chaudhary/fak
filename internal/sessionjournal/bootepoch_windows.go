//go:build windows

package sessionjournal

import (
	"syscall"
	"time"
)

// The boot epoch on Windows, dependency-free: the module is stdlib-only (no
// golang.org/x/sys), so we call GetTickCount64 through a LazyDLL — the same kernel32
// idiom internal/procguard and dispatch_tick_os_windows.go already use.
var (
	modKernel32SJ      = syscall.NewLazyDLL("kernel32.dll")
	procGetTickCount64 = modKernel32SJ.NewProc("GetTickCount64")
)

// BootTime returns the machine's last boot instant and the source, or a zero time and
// "unknown" if it cannot be read. GetTickCount64 is the elapsed milliseconds since the
// system started (it advances across sleep), so now-uptime approximates the wall-clock
// boot instant — enough for the started-before-boot crash test. The exact value (WMI
// LastBootUpTime + a persisted marker, immune to sleep/NTP drift) is C6 (#3790).
func BootTime(now time.Time) (time.Time, string) {
	r, _, _ := procGetTickCount64.Call()
	ms := uint64(r)
	if ms == 0 {
		return time.Time{}, "unknown"
	}
	return now.Add(-time.Duration(ms) * time.Millisecond).UTC(), "gettickcount64"
}
