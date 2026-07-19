package main

// microbench_test.go — witnesses for `fak microbench` (#2008): the in-process
// density cell measures a real steady-state fleet (all N live at once), the
// JSONL row carries the acceptance numbers under the stable schema tag, and the
// baseline-vs-in-process delta folds to a number (with the honest floor clamp).
// No child processes and no provider spend: the cell runs the Mock planner.

import (
	"encoding/json"
	"runtime"
	"strings"
	"testing"
)

// TestMicrobenchInProcessCellMeasuresDensity runs one small in-process cell and
// checks it measured a genuinely-concurrent fleet: every agent retired done,
// steady-state goroutines cover the N parked Step drivers, and (on platforms
// harnessres can read) the RSS witness fields are populated rather than zero.
func TestMicrobenchInProcessCellMeasuresDensity(t *testing.T) {
	const n = 4
	row, err := runMicrobenchCell(n, 2)
	if err != nil {
		t.Fatalf("runMicrobenchCell: %v", err)
	}
	if row.Kind != "inprocess" || row.Schema != microbenchSchema {
		t.Fatalf("row identity = kind %q schema %q, want inprocess/%s", row.Kind, row.Schema, microbenchSchema)
	}
	if row.N != n || row.Workers != n || row.Turns != 2 {
		t.Fatalf("row shape = N %d workers %d turns %d, want %d/%d/2", row.N, row.Workers, row.Turns, n, n)
	}
	if row.Done != n || row.Failed != 0 {
		t.Fatalf("fleet outcome = %d done / %d failed, want %d/0", row.Done, row.Failed, n)
	}
	if row.GoroutinesSteady < n {
		t.Errorf("goroutines at steady state = %d, want >= %d (one parked Step driver per live agent)", row.GoroutinesSteady, n)
	}
	if row.WallSeconds <= 0 {
		t.Errorf("wall_seconds = %v, want > 0", row.WallSeconds)
	}
	switch runtime.GOOS {
	case "windows", "linux", "darwin":
		if row.RSSSteadyBytes == 0 || row.RSSBeforeBytes == 0 {
			t.Errorf("RSS witness empty on %s: before=%d steady=%d — the density number would be a claim, not a measurement",
				runtime.GOOS, row.RSSBeforeBytes, row.RSSSteadyBytes)
		}
		if row.HostPeakRSSBytes < row.RSSSteadyBytes {
			t.Errorf("host peak rss %d < steady rss %d — peak must dominate any sampled reading", row.HostPeakRSSBytes, row.RSSSteadyBytes)
		}
	}
}

// TestMicrobenchRowJSONLShape pins the machine-readable contract: the schema
// tag and the acceptance's key fields survive marshaling under stable names.
func TestMicrobenchRowJSONLShape(t *testing.T) {
	row := microbenchRow{
		Schema:           microbenchSchema,
		Kind:             "inprocess",
		N:                100,
		RSSPerAgentBytes: 4096,
		RSSSteadyBytes:   1 << 20,
		HostPeakRSSBytes: 2 << 20,
	}
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, want := range []string{
		`"schema":"fak-microbench/1"`,
		`"rss_per_agent_bytes":4096`,
		`"host_peak_rss_bytes":2097152`,
		`"n":100`,
	} {
		if !strings.Contains(string(b), want) {
			t.Errorf("JSONL row missing %s in %s", want, b)
		}
	}
}

// TestMicrobenchDeltaFactor pins the delta arithmetic: a plain ratio when both
// sides measured, and the disclosed 1-byte floor clamp instead of a fabricated
// infinity when the in-process side reads below the floor.
func TestMicrobenchDeltaFactor(t *testing.T) {
	if f, note := microbenchDeltaFactor(2<<20, 8<<10); f != 256 || note != "" {
		t.Errorf("plain ratio = (%v, %q), want (256, \"\")", f, note)
	}
	if f, note := microbenchDeltaFactor(2<<20, 0); f != float64(2<<20) || note == "" {
		t.Errorf("floor clamp = (%v, %q), want factor vs 1 B floor with a disclosure note", f, note)
	}
	if f, note := microbenchDeltaFactor(0, 8<<10); f != 0 || note == "" {
		t.Errorf("unreadable baseline = (%v, %q), want (0, disclosure note)", f, note)
	}
}

// TestMicrobenchDeltaRowBindsLargestCell checks the folded delta row carries
// both sides and the N it compared at.
func TestMicrobenchDeltaRowBindsLargestCell(t *testing.T) {
	baseline := microbenchRow{Kind: "baseline", RSSPerAgentBytes: 1 << 20}
	inproc := microbenchRow{Kind: "inprocess", N: 1000, RSSPerAgentBytes: 1 << 10}
	d := microbenchDeltaRow(baseline, inproc)
	if d.Kind != "delta" || d.DeltaAtN != 1000 {
		t.Fatalf("delta row = kind %q at N=%d, want delta at N=1000", d.Kind, d.DeltaAtN)
	}
	if d.BaselineRSSPerAgentBytes != 1<<20 || d.InprocessRSSPerAgentBytes != 1<<10 {
		t.Errorf("delta row sides = %d vs %d, want both measured sides carried", d.BaselineRSSPerAgentBytes, d.InprocessRSSPerAgentBytes)
	}
	if d.BaselineOverInprocessRSS != 1024 {
		t.Errorf("delta factor = %v, want 1024", d.BaselineOverInprocessRSS)
	}
}
