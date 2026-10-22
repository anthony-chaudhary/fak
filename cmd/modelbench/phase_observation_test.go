package main

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:justify why=integration when=changed:cmd/modelbench/phase_observation.go,cmd/modelbench/main.go
// fak-test:runtime medium est=8s lane=default
func TestPhaseProfileV41AttributionJSONBoundary(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	path := engProjGGUF(t)
	budget := ggufload.MaxEagerF32Bytes
	ggufload.MaxEagerF32Bytes = 4 << 20
	defer func() { ggufload.MaxEagerF32Bytes = budget }()
	m, err := ggufload.LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()

	f := testCompleteBenchFlags()
	*f.q4k, *f.phaseProfile = true, true
	*f.prefillReps, *f.decodeReps, *f.decodePrompt, *f.decodeSteps = 1, 1, 2, 2
	newSession := func() *model.Session { return newBenchSession(m, f, nil) }
	report, phases := map[string]any{}, map[string]any{}
	runPrefill(f, nil, newSession, m.Cfg.VocabSize, []int{2}, report, phases)
	runDecode(f, nil, newSession, m.Cfg.VocabSize, report, phases)

	raw, err := json.Marshal(phases)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) > 64<<10 {
		t.Fatalf("phase receipt grew beyond bounded 64 KiB JSON: %d bytes", len(raw))
	}
	var boundary struct {
		Prefill []phaseProfileReceipt `json:"prefill"`
		Decode  phaseProfileReceipt   `json:"decode"`
	}
	if err := json.Unmarshal(raw, &boundary); err != nil {
		t.Fatal(err)
	}
	if len(boundary.Prefill) != 1 {
		t.Fatalf("prefill receipts=%d want=1", len(boundary.Prefill))
	}
	for _, tc := range []struct {
		name string
		want int
		got  *phaseProfileReceipt
	}{
		{name: "prefill", want: 2, got: &boundary.Prefill[0]},
		{name: "decode", want: 2, got: &boundary.Decode},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := tc.got.V41ModelAttribution
			if tc.got.PhaseProfile == nil || a == nil {
				t.Fatalf("missing legacy profile or V4.1 attribution: %+v", tc.got)
			}
			if a.Schema != smokeV41ModelAttributionSchema || a.Scope != "model_lifetime_snapshots" || a.Phase != tc.name || !a.Partial || a.ExclusiveRequestAttribution {
				t.Fatalf("attribution contract=%+v", a)
			}
			if delta := a.After.Tokens - a.Before.Tokens; delta != tc.want {
				t.Fatalf("measured token delta=%d want=%d; before=%d after=%d", delta, tc.want, a.Before.Tokens, a.After.Tokens)
			}
			if a.Before.Tokens == 0 {
				t.Fatal("before snapshot omitted earlier model-lifetime work")
			}
			if a.WallNanos <= 0 || a.WallNanos != tc.got.TotalNanos || a.AggregateTimerSemantics != "may_nest" || a.HostWallNanos != nil || a.HostWallObservationStatus != "unknown" || a.CopyWallNanos != nil || a.CopyWallObservationStatus != "unknown" || a.HardwareExecutionEvidence != "not_established_by_model_counters" {
				t.Fatalf("wall/evidence semantics=%+v", a)
			}
			phaseRequireColdStagePartition(t, tc.name, a.ColdStages, m.Cfg.NumLayers)
		})
	}
	if final := m.V41ExpertFaultAttribution().Prefill; final.Tokens <= boundary.Prefill[0].V41ModelAttribution.After.Tokens {
		t.Fatal("later decode-prompt Prefill did not prove the serialized prefill snapshot was frozen")
	}
	if !bytes.Contains(raw, []byte(`"total_nanos"`)) || !bytes.Contains(raw, []byte(`"v41_model_attribution"`)) {
		t.Fatalf("additive wrapper lost legacy or attribution JSON: %s", raw)
	}

	nonV41 := model.NewSynthetic(model.Config{ModelType: "llama", HiddenSize: 8, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, IntermediateSize: 16, VocabSize: 16})
	nonV41Session := nonV41.NewSession()
	o := beginPhaseObservation(nonV41Session, "prefill")
	legacy := &model.PhaseProfile{Mode: "prefill"}
	plain, err := json.Marshal(phaseProfileReceipt{PhaseProfile: legacy, V41ModelAttribution: o.end(nonV41Session, 1)})
	nonV41Session.Close()
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(plain, []byte(`"v41_model_attribution"`)) {
		t.Fatalf("non-V4.1 profile published attribution: %s", plain)
	}
	legacyJSON, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(plain, legacyJSON) {
		t.Fatalf("non-V4.1 wrapper changed legacy JSON: got=%s want=%s", plain, legacyJSON)
	}
}

// phaseRequireColdStagePartition checks the receipt's closed contract: stage
// rows plus unassigned sum to total in every published family, unobserved
// families are omitted, and the cold prefill enters each per-layer stage once
// per layer.
func phaseRequireColdStagePartition(t *testing.T, phase string, c *phaseV41ColdStages, layers int) {
	t.Helper()
	if c == nil || c.Schema != phaseV41ColdStagesSchema || c.Semantics != "exclusive_partition" || len(c.Stages) != int(model.V41ColdStageCount) {
		t.Fatalf("%s cold-stage receipt=%+v", phase, c)
	}
	if c.DeviceWorkObserved != (c.SubmitWaitCoverage != "") {
		t.Fatalf("%s submit_wait_coverage=%q with device_work_observed=%t", phase, c.SubmitWaitCoverage, c.DeviceWorkObserved)
	}
	type sums struct {
		entries, d2hCount, d2hBytes, h2dCount, h2dBytes, d2d, d2dBytes, submits, dispatches, deviceSubmits, workDispatches, wait uint64
		wall                                                                                                                     int64
	}
	fold := func(s *sums, row phaseV41ColdStage) {
		s.entries += row.Entries
		s.wall += row.HostWallNanos
		if (row.Counters != nil) != c.CountersObserved || (row.DeviceSubmits != nil) != c.DeviceSubmitsObserved || (row.DeviceWork != nil) != c.DeviceWorkObserved {
			t.Fatalf("%s %s family presence disagrees with the receipt flags: %+v", phase, row.Stage, row)
		}
		if k := row.Counters; k != nil {
			if (k.DispatchSubmits != nil) != c.DispatchCountsLive || (k.ComputeDispatches != nil) != c.DispatchCountsLive {
				t.Fatalf("%s %s dispatch counts presence disagrees with dispatch_counts_live=%t", phase, row.Stage, c.DispatchCountsLive)
			}
			s.d2hCount += k.D2HCount
			s.d2hBytes += k.D2HBytes
			s.h2dCount += k.H2DCount
			s.h2dBytes += k.H2DBytes
			s.d2d += k.D2DCopies
			s.d2dBytes += k.D2DBytes
			if k.DispatchSubmits != nil {
				s.submits += *k.DispatchSubmits
				s.dispatches += *k.ComputeDispatches
			}
		}
		if row.DeviceSubmits != nil {
			s.deviceSubmits += *row.DeviceSubmits
		}
		if w := row.DeviceWork; w != nil {
			s.workDispatches += w.Dispatches
			s.wait += w.SubmitWaitNanos
		}
	}
	var sum, total sums
	fold(&sum, c.Unassigned)
	for i, row := range c.Stages {
		if row.Stage != model.V41ColdStage(i).String() {
			t.Fatalf("%s stage %d named %q", phase, i, row.Stage)
		}
		fold(&sum, row)
	}
	fold(&total, c.Total)
	if sum != total {
		t.Fatalf("%s stages+unassigned do not close on total: sum=%+v total=%+v", phase, sum, total)
	}
	if phase == "prefill" && c.Stages[model.V41ColdStageQKV].Entries != uint64(layers) {
		t.Fatalf("prefill qkv entries=%d want %d layers", c.Stages[model.V41ColdStageQKV].Entries, layers)
	}
}

// TestPhaseV41ColdStagesDeviceSubmits pins the always-on submit family: it is
// published per row whenever observed, closes on total, and stays present when
// the profiler's dispatch counts are not live.
// fak-test:runtime fast est=10ms lane=default
func TestPhaseV41ColdStagesDeviceSubmits(t *testing.T) {
	var w model.V41ColdStageAttribution
	w.CountersObserved, w.DeviceSubmitsObserved = true, true
	w.Stages[model.V41ColdStageRouted] = model.V41ColdStageCounters{Entries: 3, HostWallNanos: 50, DeviceSubmits: 7}
	w.Stages[model.V41ColdStageAttention] = model.V41ColdStageCounters{Entries: 3, HostWallNanos: 20, DeviceSubmits: 2}
	w.Unassigned.DeviceSubmits = 1
	c := newPhaseV41ColdStages(w)
	phaseRequireColdStagePartition(t, "decode", c, 0)
	routed := c.Stages[model.V41ColdStageRouted]
	if !c.DeviceSubmitsObserved || c.DispatchCountsLive || routed.DeviceSubmits == nil || *routed.DeviceSubmits != 7 ||
		routed.Counters == nil || routed.Counters.DispatchSubmits != nil || c.Total.DeviceSubmits == nil || *c.Total.DeviceSubmits != 10 {
		t.Fatalf("profile-off receipt lost the always-on submits: %+v", c)
	}
	w.DeviceSubmitsObserved = false
	c = newPhaseV41ColdStages(w)
	phaseRequireColdStagePartition(t, "decode", c, 0)
	if c.Total.DeviceSubmits != nil || c.Stages[model.V41ColdStageRouted].DeviceSubmits != nil {
		t.Fatalf("unobserved submits were published: %+v", c)
	}
}
