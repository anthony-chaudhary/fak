//go:build !windows

package main

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostfault"
)

// gatherWinHostFaultRecords is Windows-only: the host-fault classes the #2170
// audit witnessed are Windows event-log signals (WindowsUpdateClient Event 20,
// WER Event 1001). On other platforms the ingest refuses so the cross-platform
// build stays green and the operator gets an honest "wrong OS" message rather
// than a silent empty report.
func gatherWinHostFaultRecords(_ time.Duration, _ int) ([]hostfault.WinFaultRecord, string) {
	return nil, "host-fault ingest requires Windows (reads the Windows System + Application event logs)"
}
