//go:build !windows

package processalive

import "time"

// StartTime is unavailable through a portable standard-library primitive.
// Host-crash resurrection is Windows-only; other platforms fail closed.
func StartTime(int) (time.Time, bool) { return time.Time{}, false }
