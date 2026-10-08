package model

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41IncrementalExpertFixture(t *testing.T, role, streamed bool) *Model {
	t.Helper()
	var m *Model
	if streamed {
		m, _ = v41TierOnlyBudgetModel(t, 0)
	} else {
		m = v41MixedQuantExpertModel(t)
	}
	c := &m.Cfg
	c.Window = []int{-1}
	c.DeepSeekV41.CompressRatios = []int{0}
	c.DeepSeekV41.KVSourceLayerIDs = nil
	c.DeepSeekV41.IndexSourceLayerIDs = nil
	c.DeepSeekV41.EngramLayerIDs = nil
	if role {
		c.HeadDim, c.NumHeads, c.NumKVHeads, c.OGroups = v41KVLoraRank, 1, 1, 1
		c.DeepSeekV41.Attention = DeepSeekV41AttentionGeometry{}
		c.DeepSeekV41.CompressRatios = []int{2}
		c.DeepSeekV41.IndexSourceLayerIDs = []int{0}
		c.IndexNHeads, c.IndexHeadDim, c.IndexTopK = 1, v41KVLoraRank, 2
		H, width := c.HiddenSize, v41CompressorWidth(*c)
		extra := []synthTensor{
			{layerName(0, "mhc.mixes.weight"), []int{4 * H, v41MHCMixWidth}},
			{layerName(0, "attn.wq_a_norm.weight"), []int{c.QLoraRank}},
			{layerName(0, "attn.wq_b.weight"), []int{c.HeadDim, c.QLoraRank}},
			{layerName(0, "attn.wkv.weight"), []int{v41KVLoraRank, H}},
			{layerName(0, "attn.kv_norm.weight"), []int{v41KVLoraRank}},
			{layerName(0, "attn.wo_a.weight"), []int{c.OLoraRank, c.HeadDim}},
			{layerName(0, "attn.wo_b.weight"), []int{H, c.OLoraRank}},
			{layerName(0, "attn.sink"), []int{1}},
			{layerName(0, "attn.compressor.wkv.weight"), []int{width, H}},
			{layerName(0, "attn.compressor.wgate.weight"), []int{width, H}},
			{layerName(0, "attn.compressor.norm.weight"), []int{width}},
			{layerName(0, "indexer.wq_b.weight"), []int{c.IndexNHeads * c.IndexHeadDim, c.QLoraRank}},
			{layerName(0, "indexer.wk.weight"), []int{c.IndexHeadDim, width}},
			{layerName(0, "indexer.k_norm.weight"), []int{c.IndexHeadDim}},
			{layerName(0, "indexer.weights_proj.weight"), []int{c.IndexNHeads, H}},
		}
		man, raw := synthBuildRaw(extra, func(name string, next func() float32) float32 {
			if hasSuffix(name, "norm.weight") {
				return 1
			}
			return synthMatmulFill(name, next)
		})
		for k, v := range man {
			m.manifest[k] = v
		}
		for k, v := range raw {
			m.raw[k] = v
		}
	}
	return m
}

func v41IncrementalExpertSession(t *testing.T, m *Model) (*Session, *v41HalSeamBackend) {
	t.Helper()
	b := &v41HalSeamBackend{Backend: compute.Default()}
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	s.v41State().denseProjection = nil
	s.v41State().groupedOutput = nil
	t.Cleanup(s.Close)
	if len(s.Prefill([]int{1, 2, 3})) == 0 || !s.v41IncrementalEligible() {
		t.Fatal("fresh prefill did not seed an eligible session")
	}
	return s, b
}

func v41IncrementalExpertPhase(t *testing.T, m *Model, phase string) map[string]float64 {
	t.Helper()
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var phases map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &phases); err != nil {
		t.Fatal(err)
	}
	out := map[string]float64{}
	for _, key := range []string{"incremental_device_gate_up_calls", "incremental_device_down_calls", "incremental_device_dispatch_nanos", "dequant_bytes", "contraction_nanos"} {
		field, ok := phases[phase][key]
		if !ok {
			t.Errorf("default %s attribution missing %s", phase, key)
			continue
		}
		var n float64
		if err := json.Unmarshal(field, &n); err != nil {
			t.Fatal(err)
		}
		out[key] = n
	}
	return out
}

// fak-test:runtime slow est=30s lane=default
func TestV41IncrementalDeviceExpertSessionRoutes(t *testing.T) {
	t.Parallel()
	for _, role := range []bool{false, true} {
		for _, suffix := range []bool{false, true} {
			name := "plain"
			if role {
				name = "compressed-index-source"
			}
			if suffix {
				name += "/suffix"
			} else {
				name += "/step"
			}
			t.Run(name, func(t *testing.T) {
				m := v41IncrementalExpertFixture(t, role, false)
				s, b := v41IncrementalExpertSession(t, m)
				snapshot := captureV41ForwardSnapshot(s.v41Forward)
				scratchState := &v41ForwardState{history: snapshot.history, layers: snapshot.layers, attn: snapshot.attn, expertGateUp: s.v41Forward.expertGateUp, expertDown: s.v41Forward.expertDown}
				before := b.matmuls
				phase := "decode"
				ids := []int{4}
				var got []float32
				if suffix {
					phase = "prefill"
					ids = []int{4, 5}
					got = s.Prefill(ids)
				} else {
					got = s.Step(4)
				}
				rows := len(ids) * m.Cfg.NumExpertsPerTok * m.Cfg.NumLayers
				if n := b.matmuls - before; n != 3*rows {
					t.Errorf("actual session route executed %d expert MatMuls, want %d", n, 3*rows)
				}
				oracle := v41IncrementalExpertFixture(t, role, false).Forward(append([]int{1, 2, 3}, ids...))
				assertV41LogitsClose(t, got, lastLogits(oracle), "incremental device expert session vs cold host oracle")
				p := v41IncrementalExpertPhase(t, m, phase)
				if p["incremental_device_gate_up_calls"] != float64(rows) || p["incremental_device_down_calls"] != float64(rows) {
					t.Errorf("handled callback counts = %v, want %d per operation", p, rows)
				}
				if p["incremental_device_dispatch_nanos"] <= 0 {
					t.Error("incremental callback dispatch elapsed absent")
				}
				continuedBefore := b.matmuls
				continued := s.Step(6)
				if b.matmuls-continuedBefore != 3*m.Cfg.NumExpertsPerTok*m.Cfg.NumLayers {
					t.Error("subsequent Step recomputed history or bypassed expert callbacks")
				}
				history := append(append([]int{1, 2, 3}, ids...), 6)
				cold := v41IncrementalExpertFixture(t, role, false).Forward(history)
				assertV41LogitsClose(t, continued, lastLogits(cold), "subsequent incremental expert Step vs cold host oracle")
				// A separate incremental composition exposes the actual materialization scratch.
				scratch := &v41ProjScratch{}
				if _, _, err := m.forwardV41Step(4, scratchState, scratch); err != nil {
					t.Fatal(err)
				}
				if len(scratch.exp1)+len(scratch.exp3)+len(scratch.exp2) != 0 || len(scratch.layerExperts) != 0 {
					t.Error("handled experts materialized whole host F32 projections")
				}
			})
		}
	}
}

// fak-test:runtime medium est=10s lane=default
func TestV41IncrementalDeviceExpertStreamedAttribution(t *testing.T) {
	t.Parallel()
	m := v41IncrementalExpertFixture(t, false, true)
	s, b := v41IncrementalExpertSession(t, m)
	before := b.matmuls
	got := s.Step(4)
	if len(got) == 0 || b.matmuls-before != 3*m.Cfg.NumExpertsPerTok {
		t.Error("streamed Step bypassed installed expert callbacks")
	}
	p := v41IncrementalExpertPhase(t, m, "decode")
	if p["dequant_bytes"] != 0 {
		t.Errorf("handled streamed expert whole F32 materialization = %g bytes", p["dequant_bytes"])
	}
	if p["contraction_nanos"] != 0 {
		t.Errorf("fully handled experts charged host contraction time %g", p["contraction_nanos"])
	}
	if p["incremental_device_dispatch_nanos"] <= 0 {
		t.Error("streamed staging and callback dispatch were not timed")
	}
}

// fak-test:runtime slow est=21s lane=default
func TestV41IncrementalDeviceExpertClosedOutcomes(t *testing.T) {
	t.Parallel()
	for _, role := range []bool{false, true} {
		for _, op := range []string{"gate_up", "down"} {
			t.Run(op+"/role="+itoa(boolToIntV41Expert(role)), func(t *testing.T) {
				m := v41IncrementalExpertFixture(t, role, false)
				s, _ := v41IncrementalExpertSession(t, m)
				st := s.v41Forward
				before := captureV41ForwardSnapshot(st)
				sentinel := errors.New("selected expert execution")
				gate, down := st.expertGateUp, st.expertDown
				if op == "gate_up" {
					st.expertGateUp = func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
						return nil, v41GateUpError, sentinel
					}
				} else {
					st.expertDown = func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
						return nil, v41DownError, sentinel
					}
				}
				err := panicAsError(func() { s.Step(4) })
				if !errors.Is(err, sentinel) {
					t.Errorf("selected failure did not preserve sentinel: %v", err)
				}
				if !reflect.DeepEqual(before, captureV41ForwardSnapshot(st)) {
					t.Error("failed Step changed retained session state")
				}
				p := v41IncrementalExpertPhase(t, m, "decode")
				key := "incremental_device_gate_up_calls"
				if op == "down" {
					key = "incremental_device_down_calls"
				}
				if p[key] != 1 {
					t.Errorf("selected failure callback count %s=%g, want 1", key, p[key])
				}
				st.expertGateUp, st.expertDown = gate, down
				// Decline is a closed fallback, with no handled dispatch receipt.
				st.expertGateUp = func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
					return nil, v41GateUpDeclined, nil
				}
				st.expertDown = func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
					return nil, v41DownDeclined, nil
				}
				if op == "down" {
					st.expertGateUp = gate
				}
				old := v41IncrementalExpertPhase(t, m, "decode")
				got := s.Step(4)
				oracle := v41IncrementalExpertFixture(t, role, false).Forward([]int{1, 2, 3, 4})
				assertV41LogitsClose(t, got, lastLogits(oracle), "declined expert host fallback")
				now := v41IncrementalExpertPhase(t, m, "decode")
				wantGate := old["incremental_device_gate_up_calls"]
				if op == "down" {
					wantGate += float64(m.Cfg.NumExpertsPerTok * m.Cfg.NumLayers)
				}
				if now["incremental_device_gate_up_calls"] != wantGate || now["incremental_device_down_calls"] != old["incremental_device_down_calls"] {
					t.Error("declined callbacks fabricated handled counts")
				}
			})
		}
	}
}
func boolToIntV41Expert(b bool) int {
	if b {
		return 1
	}
	return 0
}

// fak-test:runtime medium est=10s lane=default
func TestV41IncrementalDeviceExpertSessionOwnership(t *testing.T) {
	t.Parallel()
	m := v41IncrementalExpertFixture(t, false, false)
	a, ba := v41IncrementalExpertSession(t, m)
	b, bb := v41IncrementalExpertSession(t, m)
	na, nb := ba.matmuls, bb.matmuls
	snapshot := captureV41ForwardSnapshot(a.v41Forward)
	got := b.Step(4)
	if ba.matmuls != na || bb.matmuls-nb != 3*m.Cfg.NumExpertsPerTok {
		t.Error("fresh session dispatched through stale peer backend")
	}
	if !reflect.DeepEqual(snapshot, captureV41ForwardSnapshot(a.v41Forward)) {
		t.Error("peer step mutated retained session state")
	}
	host := m.NewSession()
	t.Cleanup(host.Close)
	host.Prefill([]int{1, 2, 3})
	want := host.Step(4)
	assertV41LogitsClose(t, got, want, "session-owned callbacks vs nil-backend session")
	if host.v41Forward.expertGateUp != nil || host.v41Forward.expertDown != nil {
		t.Error("nil-backend session inherited device callbacks")
	}
}

// fak-test:runtime medium est=3s lane=default
func TestV41IncrementalDeviceExpertDoesNotQualifyDeviceOnly(t *testing.T) {
	t.Parallel()
	m := v41IncrementalExpertFixture(t, false, false)
	s, b := v41IncrementalExpertSession(t, m)
	before := b.matmuls
	s.executionPolicy = ExecutionPolicyDeviceOnly
	err := panicAsError(func() { s.Step(4) })
	var operation *BackendForwardOperationError
	if !errors.As(err, &operation) {
		t.Fatalf("architecture guard lost typed refusal: %v", err)
	}
	if b.matmuls != before {
		t.Error("device-only refusal executed expert operations")
	}
}

// fak-test:runtime medium est=12s lane=default
func TestV41IncrementalDeviceExpertSelectedStageCause(t *testing.T) {
	t.Parallel()
	for _, role := range []bool{false, true} {
		for _, op := range []string{"gate_up", "down"} {
			t.Run(op+"/role="+itoa(boolToIntV41Expert(role)), func(t *testing.T) {
				m := v41IncrementalExpertFixture(t, role, false)
				s, _ := v41IncrementalExpertSession(t, m)
				before := captureV41ForwardSnapshot(s.v41Forward)
				calls := 0
				if op == "gate_up" {
					s.v41Forward.expertGateUp = func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
						calls++
						return nil, v41GateUpError, ErrV41ForwardStage
					}
				} else {
					s.v41Forward.expertDown = func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
						calls++
						return nil, v41DownError, ErrV41ForwardStage
					}
				}
				err := panicAsError(func() { s.Step(4) })
				var selected interface {
					SelectedExpertOperation() bool
					Unwrap() error
				}
				var stage *V41ForwardError
				if !errors.As(err, &selected) || !selected.SelectedExpertOperation() || !errors.Is(err, ErrV41ForwardStage) {
					t.Errorf("selected stage cause lost operation marker or cause: %v", err)
				}
				if !errors.As(err, &stage) || stage.Stage != v41StageMoE {
					t.Errorf("selected failure lost existing MoE stage: %v", err)
				}
				if calls != 1 {
					t.Errorf("selected callback executed %d times, want exactly 1 with no full-history retry", calls)
				}
				if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
					t.Error("selected stage failure changed retained state")
				}
			})
		}
	}
}
