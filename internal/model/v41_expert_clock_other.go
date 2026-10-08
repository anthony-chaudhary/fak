//go:build !windows

package model

import "time"

var v41ExpertClockEpoch = time.Now()

func defaultV41ExpertClockNanos() int64 {
	elapsed := time.Since(v41ExpertClockEpoch).Nanoseconds()
	const maxInt64 = int64(^uint64(0) >> 1)
	if elapsed < 0 || elapsed == maxInt64 {
		return 0
	}
	return elapsed + 1
}
