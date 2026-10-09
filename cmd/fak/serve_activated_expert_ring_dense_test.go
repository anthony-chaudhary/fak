package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// fak#13668: the ring route's dense base must reach device memory without a host-resident staging
// copy, and its host peak must be admitted rather than skipped as an unestimated route.
// fak-test:runtime fast est=10ms lane=default
func TestServeHaloGPUActivatedExpertsRingStreamsDenseBaseWithAdmittedHostPeak(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	be := serveCapBackend{total: 1 << 20, free: 1 << 20, known: true}
	ring, ok, err := serveActivatedExpertRingPlacement(ws, be, true, serveRingFullPlanTooBig(), 0, serveFitBudget{Base: 1 << 20})
	if err != nil || !ok {
		t.Fatalf("ring placement not selected: ok=%v err=%v", ok, err)
	}
	opts := serveActivatedExpertRingLoadOptions(ring)
	eff := ggufload.ApplyQ4KLoadOptions(opts)
	if !eff.StreamedDenseQ4K || !eff.StreamedDenseBounded {
		t.Fatalf("ring load options %+v stage the dense base in host RAM, want a bounded streamed dense base", eff)
	}
	peak, err := ws.EstimateStreamedExpertHostLoadPeak(opts...)
	if err != nil {
		t.Fatalf("ring route host peak is unestimated, so admission would skip it: %v", err)
	}
	if peak.PeakAnonBytes <= 0 {
		t.Fatalf("ring route host peak = %+v, want a positive admitted charge", peak)
	}
}
