package compute

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

// These synthetic software fixtures exercise selection policy; they are not hardware evidence.
func TestVulkanAutotuneSelectsMeasuredGfx1151Candidate(t *testing.T) {
	now := time.Date(2026, 9, 12, 12, 0, 0, 0, time.UTC)
	key := VulkanAutotuneKey{DeviceUUID: "gpu-1", DriverVersion: "driver-1", QuantFormat: "q4_k", SPIRVDigest: "spirv-1", M: 1, N: 4096, K: 4096}
	env := VulkanAutotuneEnvelope{SourceRevision: "source-1", BinaryDigest: "binary-1", ModelDigest: "model-1", WorkloadDigest: "workload-1", PowerEnvelope: "power-1"}
	control := VulkanRowtileVariant{ID: "control", Key: key, ResourceCost: 100}
	candidate := VulkanRowtileVariant{ID: "candidate", Key: key, ResourceCost: 80}
	request := VulkanAutotuneRequest{GFX: "gfx1151", Control: control, Candidates: []VulkanRowtileVariant{candidate}, Envelope: env, Now: now, MaxAge: time.Hour, Pairs: 8, MinSpeedup: 1.05}
	profile := func(v VulkanRowtileVariant, d, completed time.Duration) VulkanAutotuneObservation {
		return VulkanAutotuneObservation{Variant: v, Envelope: env, Engine: "fak-native", Duration: d, CompletedAt: now.Add(-30 * time.Minute).Add(completed), OraclePassed: true}
	}
	pairs := func(v VulkanRowtileVariant, ratios ...float64) []VulkanAutotunePair {
		out := make([]VulkanAutotunePair, len(ratios))
		cursor := time.Duration(0)
		for i, ratio := range ratios {
			cd, vd := 25*time.Millisecond, time.Duration(float64(25*time.Millisecond)/ratio)
			if i%2 == 0 {
				cursor += cd
				out[i].Control = profile(control, cd, cursor)
				cursor += vd
				out[i].Candidate = profile(v, vd, cursor)
			} else {
				cursor += vd
				out[i].Candidate = profile(v, vd, cursor)
				cursor += cd
				out[i].Control = profile(control, cd, cursor)
			}
		}
		return out
	}
	steady := []float64{1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25}

	t.Run("promotes only matching measured evidence", func(t *testing.T) {
		got := SelectVulkanRowtileCandidate(request, map[string][]VulkanAutotunePair{candidate.ID: pairs(candidate, steady...)})
		if !got.Promoted || got.Variant.ID != candidate.ID || got.LowerBound <= request.MinSpeedup {
			t.Fatalf("selection = %+v", got)
		}
	})

	t.Run("held-out shape cannot reuse a profile", func(t *testing.T) {
		held := request
		held.Control.Key.M = 2
		heldCandidate := candidate
		heldCandidate.Key.M = 2
		held.Candidates = []VulkanRowtileVariant{heldCandidate}
		got := SelectVulkanRowtileCandidate(held, map[string][]VulkanAutotunePair{candidate.ID: pairs(candidate, steady...)})
		if got.Promoted || !reflect.DeepEqual(got.Variant, held.Control) {
			t.Fatalf("wrong-shape evidence selected: %+v", got)
		}
	})

	t.Run("stale and noisy evidence retain the exact control", func(t *testing.T) {
		stale := pairs(candidate, steady...)
		for i := range stale {
			stale[i].Control.CompletedAt = now.Add(-2 * time.Hour)
			stale[i].Candidate.CompletedAt = now.Add(-2*time.Hour + time.Second)
		}
		noisy := pairs(candidate, 2, 1.01, 2, 1.01, 2, 1.01, 2, 1.01)
		for name, evidence := range map[string][]VulkanAutotunePair{"stale": stale, "noisy": noisy} {
			t.Run(name, func(t *testing.T) {
				got := SelectVulkanRowtileCandidate(request, map[string][]VulkanAutotunePair{candidate.ID: evidence})
				if got.Promoted || !reflect.DeepEqual(got.Variant, control) {
					t.Fatalf("refusal = %+v", got)
				}
			})
		}
	})

	t.Run("breaks bound ties by resource then ID", func(t *testing.T) {
		a := VulkanRowtileVariant{ID: "a", Key: key, ResourceCost: 70}
		b := VulkanRowtileVariant{ID: "b", Key: key, ResourceCost: 70}
		c := VulkanRowtileVariant{ID: "c", Key: key, ResourceCost: 90}
		r := request
		r.Candidates = []VulkanRowtileVariant{c, b, a}
		got := SelectVulkanRowtileCandidate(r, map[string][]VulkanAutotunePair{a.ID: pairs(a, steady...), b.ID: pairs(b, steady...), c.ID: pairs(c, steady...)})
		if !got.Promoted || got.Variant.ID != "a" {
			t.Fatalf("tie selection = %+v", got)
		}
	})

	t.Run("invalidates every exact identity field", func(t *testing.T) {
		mutations := map[string]func(*VulkanAutotunePair){
			"device":   func(p *VulkanAutotunePair) { p.Candidate.Variant.Key.DeviceUUID = "other" },
			"driver":   func(p *VulkanAutotunePair) { p.Candidate.Variant.Key.DriverVersion = "other" },
			"quant":    func(p *VulkanAutotunePair) { p.Candidate.Variant.Key.QuantFormat = "other" },
			"m":        func(p *VulkanAutotunePair) { p.Candidate.Variant.Key.M++ },
			"n":        func(p *VulkanAutotunePair) { p.Candidate.Variant.Key.N++ },
			"k":        func(p *VulkanAutotunePair) { p.Candidate.Variant.Key.K++ },
			"spirv":    func(p *VulkanAutotunePair) { p.Candidate.Variant.Key.SPIRVDigest = "other" },
			"source":   func(p *VulkanAutotunePair) { p.Candidate.Envelope.SourceRevision = "other" },
			"binary":   func(p *VulkanAutotunePair) { p.Candidate.Envelope.BinaryDigest = "other" },
			"model":    func(p *VulkanAutotunePair) { p.Candidate.Envelope.ModelDigest = "other" },
			"workload": func(p *VulkanAutotunePair) { p.Candidate.Envelope.WorkloadDigest = "other" },
			"power":    func(p *VulkanAutotunePair) { p.Candidate.Envelope.PowerEnvelope = "other" },
			"engine":   func(p *VulkanAutotunePair) { p.Candidate.Engine = "other" },
			"oracle":   func(p *VulkanAutotunePair) { p.Candidate.OraclePassed = false },
			"fallback": func(p *VulkanAutotunePair) { p.Candidate.FallbackCount = 1 },
		}
		for name, mutate := range mutations {
			t.Run(name, func(t *testing.T) {
				evidence := pairs(candidate, steady...)
				for i := range evidence {
					mutate(&evidence[i])
				}
				got := SelectVulkanRowtileCandidate(request, map[string][]VulkanAutotunePair{candidate.ID: evidence})
				if got.Promoted || !reflect.DeepEqual(got.Variant, control) {
					t.Fatalf("mismatched identity selected: %+v", got)
				}
			})
		}
	})

	t.Run("measurement is sorted bounded and atomic", func(t *testing.T) {
		r := request
		r.Pairs = 5
		r.Candidates = []VulkanRowtileVariant{{ID: "z", Key: key, ResourceCost: 2}, {ID: "a", Key: key, ResourceCost: 1}}
		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(time.Hour))
		defer cancel()
		wantErr := errors.New("probe failed")
		var called []string
		selection, profiles, err := MeasureVulkanRowtileCandidates(ctx, r, func(_ context.Context, v VulkanRowtileVariant) (VulkanAutotuneObservation, error) {
			called = append(called, v.ID)
			if len(called) == 2 {
				return VulkanAutotuneObservation{}, wantErr
			}
			return profile(v, 20*time.Millisecond, time.Second), nil
		})
		if !errors.Is(err, wantErr) || len(profiles) != 0 || selection.Promoted || !reflect.DeepEqual(selection.Variant, control) || !reflect.DeepEqual(called, []string{"control", "a"}) {
			t.Fatalf("error sweep: selection=%+v profiles=%d called=%v err=%v", selection, len(profiles), called, err)
		}
		called = nil
		selection, profiles, err = MeasureVulkanRowtileCandidates(context.Background(), r, func(context.Context, VulkanRowtileVariant) (VulkanAutotuneObservation, error) {
			called = append(called, "unexpected")
			return VulkanAutotuneObservation{}, nil
		})
		if err == nil || len(called) != 0 || len(profiles) != 0 || selection.Promoted || !reflect.DeepEqual(selection.Variant, control) {
			t.Fatalf("deadline refusal: selection=%+v profiles=%d called=%v err=%v", selection, len(profiles), called, err)
		}
		canceled, cancelNow := context.WithDeadline(context.Background(), time.Now().Add(time.Hour))
		cancelNow()
		selection, profiles, err = MeasureVulkanRowtileCandidates(canceled, r, func(context.Context, VulkanRowtileVariant) (VulkanAutotuneObservation, error) {
			called = append(called, "unexpected")
			return VulkanAutotuneObservation{}, nil
		})
		if !errors.Is(err, context.Canceled) || len(called) != 0 || len(profiles) != 0 || selection.Promoted || !reflect.DeepEqual(selection.Variant, control) {
			t.Fatalf("cancellation refusal: selection=%+v profiles=%d called=%v err=%v", selection, len(profiles), called, err)
		}
	})
}
