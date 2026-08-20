package ggufload

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestLoadProgressReporter verifies the streaming load-progress lines: throttled by a
// percent step, with the first and last tick always emitted, and carrying the percent,
// tensor count, and GB read so a multi-minute large-model load is observable.
func TestLoadProgressReporter(t *testing.T) {
	pinHostMemStatus(t, compute.HostMemStatus{})
	var buf bytes.Buffer
	p := NewLoadProfiler()
	p.Progress = &buf
	p.ProgressEvery = 25 // a line roughly every 25%
	p.SetTotal(8)

	const gb = int64(1) << 30
	for i := 0; i < 8; i++ {
		p.Tick(gb) // 1 GB per tensor
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	// first (1/8=12.5%), then ~25/50/75 steps, then last (8/8=100%) — bounded, not one-per-tensor.
	if len(lines) < 3 || len(lines) > 6 {
		t.Fatalf("expected a handful of throttled lines, got %d:\n%s", len(lines), buf.String())
	}
	first, last := lines[0], lines[len(lines)-1]
	if !strings.Contains(first, "1/8 tensors") {
		t.Errorf("first line should report tensor 1/8: %q", first)
	}
	if !strings.Contains(last, "100%") || !strings.Contains(last, "8/8 tensors") {
		t.Errorf("last line should report 100%% 8/8: %q", last)
	}
	if !strings.Contains(last, "GB") || !strings.Contains(last, "elapsed") {
		t.Errorf("progress line should carry GB + elapsed: %q", last)
	}
}

// TestLoadProgressNilSafe confirms the progress hooks are no-ops without a Progress writer
// (the default) and on a nil profiler, so the loader's hot path is unaffected when off.
func TestLoadProgressNilSafe(t *testing.T) {
	var nilP *LoadProfiler
	nilP.SetTotal(10) // must not panic
	nilP.Tick(123)    // must not panic

	p := NewLoadProfiler() // Progress unset
	p.SetTotal(10)
	for i := 0; i < 10; i++ {
		p.Tick(1)
	}
	// nothing to assert beyond "did not panic and emitted nothing"; cumBytes still tracked.
	if p.cumBytes != 10 {
		t.Fatalf("cumBytes = %d, want 10", p.cumBytes)
	}
}

// TestLoadAlertsDoNotReenablePercentageProgress is the startup-screen witness: a
// production-style profiler may keep the safety-alert channel armed while ordinary
// percentage updates remain byte-silent. The alert is retained in the final profile.
func TestLoadAlertsDoNotReenablePercentageProgress(t *testing.T) {
	const gib = int64(1) << 30
	pinHostMemStatus(t, compute.HostMemStatus{
		Constrained: true,
		PolicyNodes: "1",
		PolicyLabel: "membind",
		PolicyFree:  32 * gib,
		HostAvail:   64 * gib,
	})
	var screen bytes.Buffer
	p := NewLoadProfiler()
	p.AlertWriter = &screen
	p.SetTotal(2)
	p.Tick(gib)
	p.Tick(gib)

	got := screen.String()
	if strings.Contains(got, "loading model") || strings.Contains(got, "% (") {
		t.Fatalf("alerts-only startup flashed percentage progress:\n%s", got)
	}
	if !strings.Contains(got, "memory preflight") {
		t.Fatalf("alerts-only startup lost the safety notice:\n%s", got)
	}
	profile := p.Snapshot("test", "fixture.gguf", 1)
	if len(profile.Alerts) != 1 || profile.Alerts[0].Kind != "memory-preflight" {
		t.Fatalf("durable alerts = %+v, want one memory-preflight", profile.Alerts)
	}
}
