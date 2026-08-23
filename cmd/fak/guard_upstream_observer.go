package main

import (
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountobs"
	"github.com/anthony-chaudhary/fak/internal/goalpark"
)

func guardUpstreamObserver(quotaHarvester *accountobs.Harvester, parkStore goalpark.Store, parkGoal string, parkTemplate goalpark.Record, log io.Writer) func(int, http.Header) {
	longRetryParked := false
	return func(status int, header http.Header) {
		// Passive, zero-request harvest; persistence failures never fail the response path.
		_ = quotaHarvester.Observe(status, header)
		if parkGoal == "" || longRetryParked {
			return
		}
		parked, err := parkStore.RecordLongRetry(status, header, time.Now(), parkTemplate)
		if err != nil {
			fmt.Fprintf(log, "fak guard: long Retry-After park failed open: %v\n", err)
			return
		}
		if !parked {
			return
		}
		longRetryParked = true
		if rec, err := parkStore.Load(parkGoal); err == nil {
			fmt.Fprintf(log, "fak guard: PARKED goal=%q parked_until=%d reason=%s account=%q pool=%q next=%q\n", rec.Goal, rec.ParkedUntil, rec.Reason, rec.Account, rec.Pool, rec.NextAction)
		}
	}
}
