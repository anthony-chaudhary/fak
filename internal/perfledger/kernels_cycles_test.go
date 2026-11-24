package perfledger

import (
	"encoding/json"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestNewNativeKernelsCarriesCyclesWithClockProvenance(t *testing.T) {
	w := kernelWindowFixture()
	w.SMClockMHz, w.ClockSource = 2000, "declared"
	got := NewNativeKernels(w)
	if got.SMClockMHz != 2000 || got.ClockSource != "declared" || got.GPUMCyclesPerToken != 200 {
		t.Fatalf("clock=%v source=%q mcycles/token=%v, want 2000 declared 200", got.SMClockMHz, got.ClockSource, got.GPUMCyclesPerToken)
	}
	want := map[string][2]float64{
		"q2k_matvec": {5e7, 100},
		"attention":  {6e7, 60},
		"rmsnorm":    {5e6, 20},
		"d2d_copy":   {4e6, 12},
		KernelOther:  {8e6, 8},
	}
	var sum float64
	for _, r := range got.SelectedKernels {
		w, ok := want[r.Kernel]
		if !ok || r.CyclesPerCall != w[0] || r.MCyclesPerToken != w[1] {
			t.Fatalf("%s cycles/call=%v mcycles/token=%v, want %v", r.Kernel, r.CyclesPerCall, r.MCyclesPerToken, w)
		}
		sum += r.MCyclesPerToken
	}
	if sum != got.GPUMCyclesPerToken {
		t.Fatalf("row mcycles/token sum %v != table %v", sum, got.GPUMCyclesPerToken)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNewNativeKernelsOmitsCyclesWithoutKnownClock(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mhz    float64
		source string
	}{{"no clock", 0, ""}, {"clock without source", 2000, ""}, {"source without clock", 0, "observed"}} {
		w := kernelWindowFixture()
		w.SMClockMHz, w.ClockSource = tc.mhz, tc.source
		raw, err := json.Marshal(NewNativeKernels(w))
		if err != nil {
			t.Fatal(err)
		}
		for _, key := range []string{"cycles", "sm_clock_mhz", "clock_source"} {
			if strings.Contains(string(raw), key) {
				t.Fatalf("%s: wire carries %q; an unknown clock must omit cycles, not report zero: %s", tc.name, key, raw)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCyclesFromNS(t *testing.T) {
	if got := CyclesFromNS(17800, 2617); got != 46582.6 {
		t.Fatalf("17.8us at 2617 MHz = %v cycles, want 46582.6", got)
	}
}
