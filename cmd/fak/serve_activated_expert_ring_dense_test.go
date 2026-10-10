package main

import (
	"errors"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// fak#13668: ring options must describe a streamed dense base and support a header-only estimate.
// Qualification and admission remain separate; this estimate does not authorize a load.
// fak-test:runtime fast est=10ms lane=default
func TestServeHaloGPUActivatedExpertsRingStreamsDenseBaseWithEstimatedHostPeak(t *testing.T) {
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
		t.Fatalf("ring route header-only host peak estimate unavailable: %v", err)
	}
	if peak.PeakAnonBytes <= 0 {
		t.Fatalf("ring route host peak = %+v, want a positive estimated charge", peak)
	}
}

// fak#13774: fixed-term sizing must not ask actual-route admission to accept a zero bound.
// A reachable planner estimate does not qualify the streamed load for admission.
// fak-test:justify why=regression when=changed:cmd/fak/serve_activated_expert_ring.go
// fak-test:runtime fast est=20ms lane=default
func TestServeActivatedExpertRingDenseSizingUsesFixedPeakWithoutAdmitting(t *testing.T) {
	const gib = int64(1 << 30)
	path := writeServeStreamedSynthGGUF(t, "ring-fixed-peak.gguf")
	opts := append(serveActivatedExpertRingLoadOptions(serveActivatedExpertRing{RingBytes: 1 << 20}), ggufload.WithStreamedDenseQ4KWorkingSet(127))
	before := ggufload.ApplyQ4KLoadOptions(opts)
	fixed, err := estimateServeActivatedExpertRingFixedHostPeak(path, opts)
	if err != nil || fixed.PeakAnonBytes != 3*1024 {
		// The fixture has one 1024-byte F32 tensor: 1024 steady + 2048 worker window.
		t.Fatalf("fixed estimate=%+v error=%v, want 3072 bytes", fixed, err)
	}
	env := hostPeakTestEnv(nil)
	margin := serveHostPeakMarginBytes(env)
	fallback := serveFitBudget{Base: 32 * gib}
	fallbackBound := serveStreamedDenseQ4KWorkingSetBound(fallback)
	const free = 8 * gib
	wantSized := int64(float64(free-margin-fixed.PeakAnonBytes) * (1 - serveCPUOffloadStreamedResidentMargin))
	if wantSized <= 0 || wantSized == fallbackBound {
		t.Fatal("fixture must distinguish fixed-peak sizing from the historical fallback")
	}
	for _, tc := range []struct {
		name  string
		path  string
		free  int64
		known bool
		want  int64
	}{
		{"fixed terms reserved", path, free, true, wantSized},
		{"no fixed headroom", path, fixed.PeakAnonBytes + margin, true, 1},
		{"unreadable header", filepath.Join(t.TempDir(), "missing.gguf"), free, true, fallbackBound},
		{"unknown host", path, free, false, fallbackBound},
		{"empty host budget", path, 0, true, fallbackBound},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			memory := func() (int64, int64, bool) {
				calls++
				return 32 * gib, tc.free, tc.known
			}
			option := serveActivatedExpertRingDenseOptionForHost(tc.path, opts, fallback, env, memory)
			actual := append(append([]ggufload.Q4KLoadOption(nil), opts...), option)
			effects := ggufload.ApplyQ4KLoadOptions(actual)
			if calls != 1 || !effects.StreamedDenseQ4K || !effects.StreamedDenseBounded || effects.StreamedDenseBytes != tc.want {
				t.Fatalf("calls=%d effects=%+v, want bound %d", calls, effects, tc.want)
			}
			if ggufload.ApplyQ4KLoadOptions(opts) != before {
				t.Fatal("fixed-term sizing mutated the caller's load options")
			}
			if tc.name == "fixed terms reserved" {
				peak, estimateErr := estimateServeNativeHostLoadPeak(path, false, actual)
				if estimateErr != nil {
					t.Fatal(estimateErr)
				}
				_, admissionErr := serveNativeHostLoadPeakDecision(peak, nil, free, true, env, actual...)
				var unavailable *serveHostLoadPeakUnavailableError
				if !errors.As(admissionErr, &unavailable) || unavailable.Reason != "streamed-staging-peak-unqualified" {
					t.Fatalf("admission error=%v, want unchanged unqualified-staging refusal", admissionErr)
				}
			}
		})
	}
}
