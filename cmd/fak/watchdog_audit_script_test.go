package main

import (
	"os"
	"strings"
	"testing"
)

func TestWatchdogAuditIgnoresDisabledTowerTasks(t *testing.T) {
	raw, err := os.ReadFile("../../tools/watchdog_watchdog_audit.ps1")
	if err != nil {
		t.Fatal(err)
	}
	script := string(raw)
	for _, want := range []string{"$enabled = [bool]$t.Settings.Enabled", "$down = ($enabled -and", "$latent = ($enabled -and"} {
		if !strings.Contains(script, want) {
			t.Errorf("audit missing state-aware tower contract %q", want)
		}
	}
}
