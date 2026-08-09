//go:build windows

package safecommit

import (
	"syscall"
	"time"
)

// processStartTime resolves when the process at pid started, reporting ok=false when that
// cannot be established. It is the identity half of the PID-reuse guard (issue #5892): an
// image NAME can only ever say "this looks like something a committer might run", which on
// a fleet host is nearly every process, whereas a start time can PROVE the process at a
// recycled PID is not the one that took the lock.
//
// On Windows the creation time comes from GetProcessTimes as a FILETIME, opened with the
// same PROCESS_QUERY_LIMITED_INFORMATION right the liveness and image probes use so no new
// privilege is required. Any failure — the process exited between the liveness check and
// here, or we lack the right — yields ok=false, which leaves the caller on the pre-existing
// image heuristic rather than inventing a verdict.
func processStartTime(pid int) (time.Time, bool) {
	if pid <= 0 {
		return time.Time{}, false
	}
	const processQueryLimitedInformation = 0x1000
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil || h == 0 {
		return time.Time{}, false
	}
	defer syscall.CloseHandle(h)

	var creation, exit, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(h, &creation, &exit, &kernel, &user); err != nil {
		return time.Time{}, false
	}
	// Filetime.Nanoseconds converts the 1601-epoch FILETIME to nanoseconds since the Unix
	// epoch. A non-positive result means the field was never populated, which is not a
	// start time we may reason about.
	ns := creation.Nanoseconds()
	if ns <= 0 {
		return time.Time{}, false
	}
	return time.Unix(0, ns), true
}
