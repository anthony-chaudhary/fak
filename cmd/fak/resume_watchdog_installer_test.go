package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestResumeWatchdogInstallerDefaultsToLive pins the #3321 contract on
// tools/register_resume_watchdog.ps1: a no-flag install must produce a LIVE
// task (action carries -Live) with StartWhenAvailable, and dry-run must be an
// explicit opt-in. The old DRY-RUN default meant a habitual no-flag reinstall
// silently downgraded the fleet's auto-resume layer to log-only; this test
// makes that regression a red suite instead of a quiet outage.
func TestResumeWatchdogInstallerDefaultsToLive(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "tools", "register_resume_watchdog.ps1"))
	if err != nil {
		t.Fatalf("read installer: %v", err)
	}
	s := string(b)
	for _, want := range []string{
		// dry-run is an explicit, named opt-out — not the default
		"[switch]$DryRun",
		// the live-by-default resolution itself
		"if (-not $DryRun) { $Live = $true }",
		// contradictory flags fail loud instead of picking a winner
		"if ($Live -and $DryRun) { throw",
		// a tick missed while the box slept fires on wake (New-ScheduledTaskSettingsSet switch)
		"-StartWhenAvailable",
		// the host-side doctor probe for an installed LIVE action
		"'install','remove','status','assert-live'",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("installer missing %q", want)
		}
	}
	if strings.Contains(s, "DRY-RUN (safe default)") {
		t.Errorf("installer still advertises DRY-RUN as the default install mode")
	}
}
