package main

import (
	"strings"
	"testing"
	"time"
)

// TestRunDispatchAuditWindowSkipsOldLogs is the #3478 regression: the scheduled
// `fak dispatch audit` must NOT re-read the entire never-reaped runs dir every
// cadence. With the default retrospective window an out-of-window log is
// skipped (both the worker scan and the signature scan); --window-h=0 is the
// explicit scan-everything opt-out that restores the legacy full scan.
func TestRunDispatchAuditWindowSkipsOldLogs(t *testing.T) {
	runsDir := t.TempDir()

	const oldName = "resolve-101-20200101-000000.log"
	writeDispatchAuditFixture(t, runsDir, oldName,
		"# fak-spawn 20200101-000000 issue=101 lane=cmd backend=claude argv0=claude\n"+
			"panic: oldboom\n")

	recentStamp := time.Now().UTC().Add(-time.Hour).Format("20060102-150405")
	recentName := "resolve-202-" + recentStamp + ".log"
	writeDispatchAuditFixture(t, runsDir, recentName,
		"# fak-spawn "+recentStamp+" issue=202 lane=cmd backend=claude argv0=claude\n"+
			"panic: recentboom\n")

	// Default window (168h): the 2020 log is out of window and must be skipped.
	var stdout, stderr strings.Builder
	code := runDispatchAudit(&stdout, &stderr, []string{"--runs-dir", runsDir, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	out := stdout.String()
	if strings.Contains(out, oldName) {
		t.Fatalf("windowed audit still scanned the out-of-window log %s:\n%s", oldName, out)
	}
	if !strings.Contains(out, recentName) {
		t.Fatalf("windowed audit dropped the in-window log %s:\n%s", recentName, out)
	}

	// --window-h=0: the explicit opt-out scans everything, old log included.
	stdout.Reset()
	stderr.Reset()
	code = runDispatchAudit(&stdout, &stderr, []string{"--runs-dir", runsDir, "--json", "--window-h", "0"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	out = stdout.String()
	if !strings.Contains(out, oldName) || !strings.Contains(out, recentName) {
		t.Fatalf("--window-h=0 must scan every log (want both %s and %s):\n%s", oldName, recentName, out)
	}
}
