//go:build !windows && !linux && !darwin

package processstart

import "time"

func Start(pid int) (time.Time, bool) { return time.Time{}, false }
