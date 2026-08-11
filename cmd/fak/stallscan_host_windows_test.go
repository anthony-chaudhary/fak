//go:build windows

package main

import (
	"strings"
	"testing"
	"time"
)

func TestParseStallRawCarriesPostCrashHostIdentity(t *testing.T) {
	raw, errText := parseStallRaw(`{"timestamp":"2026-08-11T03:00:00Z","boot_time":"2026-08-11T00:48:38.5Z","commit_bytes":39637938176,"commit_limit":289432506368,"available_bytes":242432000000,"faults":1}`)
	if errText != "" {
		t.Fatal(errText)
	}
	if raw.CommitBytes != 39637938176 || raw.CommitLimit != 289432506368 || raw.AvailableBytes != 242432000000 {
		t.Fatalf("host counters not preserved: %+v", raw)
	}
	boot, err := time.Parse(time.RFC3339Nano, raw.BootTime)
	if err != nil || boot.IsZero() {
		t.Fatalf("boot identity invalid: %q %v", raw.BootTime, err)
	}
}

func TestStallProbeAddsHostCountersWithoutAnotherProcessEnumeration(t *testing.T) {
	if got := strings.Count(stallPS, "$snap = { Get-CimInstance Win32_Process"); got != 1 {
		t.Fatalf("process enumeration count=%d, want 1 script site", got)
	}
	for _, counter := range []string{`\Memory\Committed Bytes`, `\Memory\Commit Limit`, `\Memory\Available MBytes`} {
		if !strings.Contains(stallPS, counter) {
			t.Fatalf("missing counter %s", counter)
		}
	}
	if !strings.Contains(stallPS, "Win32_OperatingSystem") {
		t.Fatal("missing boot identity query")
	}
}
