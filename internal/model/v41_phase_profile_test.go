package model

import (
	"reflect"
	"testing"
	"time"
)

// fak-test:runtime fast est=2s lane=default
func TestV41PhaseProfileExclusiveHostWall(t *testing.T) {
	m := v41ReducedCompressIndexModel(t)
	ids := []int{1, 2, 3, 4}

	unprofiled := &Session{M: m}
	want := unprofiled.Prefill(ids)
	if unprofiled.PhaseProfiler != nil {
		t.Fatal("nil phase profiler was allocated during V4.1 prefill")
	}

	profiler := NewPhaseProfiler()
	profiled := &Session{M: m, PhaseProfiler: profiler}
	started := time.Now()
	got := profiled.Prefill(ids)
	totalNanos := time.Since(started).Nanoseconds()
	if !reflect.DeepEqual(got, want) {
		t.Fatal("attaching the phase profiler changed V4.1 logits")
	}

	wantPhases := map[string]bool{
		v41PhaseHostLayerSetupQKV:          true,
		v41PhaseHostLayerCompressorIndexer: true,
		v41PhaseHostLayerAttention:         true,
		v41PhaseHostLayerMoE:               true,
		v41PhaseHostLayerCommit:            true,
	}
	profile := profiler.Snapshot("prefill", len(ids), 0, totalNanos)
	if profile == nil {
		t.Fatal("attached phase profiler returned no snapshot")
	}
	var attributedNanos int64
	seen := make(map[string]bool, len(wantPhases))
	for _, phase := range profile.Phases {
		if !wantPhases[phase.Phase] {
			t.Fatalf("unexpected V4.1 phase %q", phase.Phase)
		}
		if phase.Calls != m.Cfg.NumLayers {
			t.Fatalf("phase %q calls = %d, want one per layer (%d)", phase.Phase, phase.Calls, m.Cfg.NumLayers)
		}
		seen[phase.Phase] = true
		attributedNanos += phase.Nanos
	}
	if !reflect.DeepEqual(seen, wantPhases) {
		t.Fatalf("profiled phases = %v, want %v", seen, wantPhases)
	}
	if attributedNanos > totalNanos {
		t.Fatalf("exclusive layer phases = %dns exceed whole prefill = %dns", attributedNanos, totalNanos)
	}
}
