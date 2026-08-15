package main

import (
	"fmt"
	"io"
	"os"
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

func runInfoWorkDoneHistoryQuery(stdout, stderr io.Writer, c *claudeMacDebugClient, window time.Duration, historyPath, workloadKey, runKey string) int {
	var q guardInfoWorkDoneQuery
	before, ok := fetchGuardInfoVars(c, stderr)
	if !ok {
		return 1
	}
	start := time.Now().UTC()
	if window <= 0 {
		q = guardInfoSessionWorkDoneQuery(before, start)
	} else {
		time.Sleep(window)
		after, ok := fetchGuardInfoVars(c, stderr)
		if !ok {
			return 1
		}
		q = guardInfoBoundedWorkDoneQuery(before, after, start, time.Now().UTC())
	}
	if historyPath == "" {
		return encodeJSONOrFail(stdout, stderr, q, "fak info --work-done-json")
	}
	records, err := guardInfoReadWorkHistory(historyPath)
	if err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(stderr, "fak info --work-done-history: %v\n", err)
		return 1
	}
	record := guardInfoWorkHistoryRecordFromQuery(q, workloadKey, runKey, time.Now().UTC())
	comparison := guardInfoCompareWorkHistory(record, records)
	if err := guardInfoAppendWorkHistory(historyPath, record); err != nil {
		fmt.Fprintf(stderr, "fak info --work-done-history: %v\n", err)
		return 1
	}
	export := guardInfoWorkHistoryExport{Schema: guardInfoWorkHistorySchema, Records: guardInfoComparedHistoryRecords(record, records), Comparison: comparison}
	return encodeJSONOrFail(stdout, stderr, export, "fak info --work-done-json")
}
