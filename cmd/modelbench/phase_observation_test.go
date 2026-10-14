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
