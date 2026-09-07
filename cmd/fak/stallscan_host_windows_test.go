//go:build windows

package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stallscan"
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

func TestParseStallRawCarriesWDDMCounters(t *testing.T) {
	raw, errText := parseStallRaw(`{"timestamp":"2026-09-06T12:00:00Z","boot_time":"2026-09-06T00:00:00Z","commit_bytes":39637938176,"commit_limit":289432506368,"available_bytes":242432000000,"vram_committed_bytes":7800000000,"vram_total_bytes":8589934592,"vram_shared_bytes":524288000,"faults":1}`)
	if errText != "" {
		t.Fatal(errText)
	}
	if raw.VRAMCommittedBytes != 7800000000 || raw.VRAMTotalBytes != 8589934592 || raw.VRAMSharedBytes != 524288000 {
		t.Fatalf("WDDM VRAM counters not preserved: %+v", raw)
	}
}

func TestStallProbeIncludesWDDMCounters(t *testing.T) {
	for _, counter := range []string{`\GPU Adapter Memory(*)\Dedicated Usage`, `\GPU Adapter Memory(*)\Shared Usage`} {
		if !strings.Contains(stallPS, counter) {
			t.Fatalf("missing WDDM counter %s", counter)
		}
	}
	if !strings.Contains(stallPS, "Win32_VideoController") {
		t.Fatal("missing Win32_VideoController query for VRAM capacity")
	}
}

func TestStallscanWindowsAlertsOnVRAMApproachingCapacity(t *testing.T) {
	s := stallscan.Sample{
		AvailableMB:        32000,
		VRAMTotalBytes:     8 * 1024 * 1024 * 1024,
		VRAMCommittedBytes: 7800 * 1024 * 1024, // ~95.2%
		VRAMSharedBytes:    512 * 1024 * 1024,
	}
	v := stallscan.Classify(s, stallscan.DefaultThresholds())
	if v.Level != stallscan.LevelStall || v.Cause != stallscan.CauseGPUMemPressure {
		t.Fatalf("expected LevelStall with CauseGPUMemPressure, got level=%s cause=%s", v.Level, v.Cause)
	}

	var buf bytes.Buffer
	renderStallFingerprint(&buf, s, v, 6)
	out := buf.String()
	if !strings.Contains(out, "vram") {
		t.Fatalf("render output missing vram line:\n%s", out)
	}
	if !strings.Contains(out, "WARNING: VRAM committed approaches capacity") {
		t.Fatalf("render output missing VRAM paging warning:\n%s", out)
	}
}
