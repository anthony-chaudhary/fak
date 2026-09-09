//go:build darwin

package safecommit

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/processstart"
)

// processStartTime resolves when the process at pid started, reporting ok=false when that
// cannot be established. On Darwin it delegates to processstart.Start, which queries the
// kernel's kern.proc.pid sysctl directly.
func processStartTime(pid int) (time.Time, bool) {
	if pid <= 0 {
		return time.Time{}, false
	}
	return processstart.Start(pid)
}
