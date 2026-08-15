package main

import (
	"io"
	"time"
)

// runInfoWorkDoneQuery is the integration-facing read seam. It emits the same accounting
// object the TUI renders; bounded mode differs only by sampling two cumulative snapshots and
// applying the reset-aware delta fold in guardInfoBoundedWorkDoneQuery.
func runInfoWorkDoneQuery(stdout, stderr io.Writer, c *claudeMacDebugClient, window time.Duration) int {
	before, ok := fetchGuardInfoVars(c, stderr)
	if !ok {
		return 1
	}
	start := time.Now().UTC()
	if window <= 0 {
		return encodeJSONOrFail(stdout, stderr, guardInfoSessionWorkDoneQuery(before, start), "fak info --work-done-json")
	}
	time.Sleep(window)
	after, ok := fetchGuardInfoVars(c, stderr)
	if !ok {
		return 1
	}
	end := time.Now().UTC()
	return encodeJSONOrFail(stdout, stderr, guardInfoBoundedWorkDoneQuery(before, after, start, end), "fak info --work-done-json")
}
