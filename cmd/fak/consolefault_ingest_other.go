//go:build !windows

package main

import "time"

// gatherWinConsoleFaultRecords is Windows-only: the console-host fault class the
// #2170 audit witnessed is a Windows event-log signal (pwsh / .NET Runtime Event
// 1026). On other platforms the ingest refuses so the cross-platform build stays
// green and the operator gets an honest "wrong OS" message rather than a silent
// empty report.
func gatherWinConsoleFaultRecords(_ time.Duration) ([]winEventRecord, string) {
	return nil, "console-fault ingest requires Windows (reads the Windows Application event log)"
}
