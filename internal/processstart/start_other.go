//go:build !windows && !linux

package processstart

import "time"

func Start(pid int) (time.Time, bool) { return time.Time{}, false }
