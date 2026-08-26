package processstart

import (
	"math"
	"time"
)

type darwinProcessInfo struct {
	pid       int64
	startSec  int64
	startUsec int64
}

type darwinProcessInfoReader func(pid int) (darwinProcessInfo, error)

func darwinProcessStart(pid int, read darwinProcessInfoReader) (time.Time, bool) {
	if pid <= 0 || int64(pid) > math.MaxInt32 {
		return time.Time{}, false
	}
	info, err := read(pid)
	if err != nil || info.pid != int64(pid) || info.startSec <= 0 || info.startUsec < 0 || info.startUsec >= 1_000_000 {
		return time.Time{}, false
	}
	return time.Unix(info.startSec, info.startUsec*int64(time.Microsecond)).UTC(), true
}
