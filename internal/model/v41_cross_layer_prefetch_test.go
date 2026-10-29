package model

import (
	"fmt"
	"reflect"
	"testing"
)

// v41CrossLayerFixture streams layer 0's experts 0..2 from a mapped checkpoint whose
// read-ahead advice is recorded in *hints, and opens a routed-layer session on it.
func v41CrossLayerFixture(t *testing.T) (*Model, *Session, *ExpertCheckpointTier, int64, *[]string) {
	t.Helper()
	m := v41IncrementalExpertFixture(t, true, false)
	tier := NewExpertCheckpointTier(0)
	hints := &[]string{}
	var tripleBytes int64
	for p, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
		names := func(e int) string {
			stem := "ffn.experts." + itoa(e)
			return [3]string{layerName(0, stem+".w1.weight"), layerName(0, stem+".w3.weight"), layerName(0, stem+".w2.weight")}[p]
		}
		w := m.kqw[names(0)]
		stride := int64(len(w.raw))
		var blob []byte
		for e := 0; e < 3; e++ {
			blob = append(blob, m.kqw[names(e)].raw...)
			delete(m.kqw, names(e))
		}
		reader := &mappedExpertReader{data: blob}
		r := &checkpointReadAheadWitness{mappedExpertReader: reader, ahead: func(off, size int64) error {
			*hints = append(*hints, fmt.Sprintf("%s:%d", proj, off/stride))
			return nil
		}}
		quant := ExpertCheckpointQ2K
		if p == 2 {
			quant = ExpertCheckpointQ3K
		}
		if err := tier.AddShard(r, int64(len(blob)), []FusedExpertTensor{{Name: proj, Layer: 0, Proj: proj, Arch: "deepseek41", Quant: quant, Experts: 3, Rows: w.out, Cols: w.in}}); err != nil {
			t.Fatal(err)
		}
		tripleBytes += stride
	}
	m.SetExpertCheckpoint(tier)
	s, err := m.NewBackendSessionChecked(newV41RoutedLayerTestBackend())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	s.ExpertRingBytes = 2 * tripleBytes
	return m, s, tier, tripleBytes, hints
}

// The knob reaches the bound V4.1 routed layer: on, the layer reconciles the pending
// prediction for itself and predicts layer+1 from its own input; a token-routed next
// layer is recorded without a prediction. Off (unset, or an on-demand session), the
// ledger never moves. Every arm returns the same routed sum.
// fak-test:runtime fast est=300ms lane=default
func TestV41CrossLayerPrefetchKnobReachesRoutedLayer(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	var oracle []float32
	for _, arm := range []string{"off", "env", "mode", "on_demand_env", "hash_next"} {
		t.Run(arm, func(t *testing.T) {
			env := ""
			if arm == "env" || arm == "on_demand_env" || arm == "hash_next" {
				env = "1"
			}
			t.Setenv(v41CrossLayerPrefetchEnv, env)
			m, s, _, _, _ := v41CrossLayerFixture(t)
			switch arm {
			case "mode":
				s.ExpertPrefetch = ExpertPrefetchCrossLayer
			case "on_demand_env":
				s.ExpertPrefetch = ExpertPrefetchOnDemand
			}
			// Give the model a second layer whose router aliases layer 0's, so the
			// prediction for layer 1 is checkable against the host route.
			m.Cfg.NumLayers = 2
			for _, leaf := range []string{"ffn.gate.weight", v41LeafSuffix[v41LeafGateBias]} {
				if meta, ok := m.manifest[layerName(0, leaf)]; ok {
					m.manifest[layerName(1, leaf)] = meta
				}
			}
			if arm == "hash_next" {
				m.manifest[v4HashTensorName(1)] = tensorMeta{}
			}
			t.Cleanup(func() { ResetCrossLayerPrefetchStats(s) })
			picks := []routePick{{expert: 0, weight: .25}, {expert: 2, weight: .5}}
			recordCrossLayerPrediction(s, 0, []routePick{{expert: 0}, {expert: 1}})
			_, x := v41RoutedLayerFaultPicks(m)
			layer := s.v41RoutedLayerFunc()
			if layer == nil {
				t.Fatal("layer did not bind")
			}
			result, outcome, err := layer(0, picks, x)
			if outcome != v41RoutedExpertHandled || err != nil || s.BackendSessionClosed() {
				t.Fatalf("outcome=%v err=%v closed=%v, want handled", outcome, err, s.BackendSessionClosed())
			}
			if oracle == nil {
				oracle = append([]float32(nil), result...)
			} else if !reflect.DeepEqual(result, oracle) {
				t.Fatal("cross-layer prefetch changed the routed sum")
			}
			st := CrossLayerPrefetchStatsFor(s)
			crossLayerGateState.mu.Lock()
			pending := crossLayerGateState.pending[s]
			var next map[int]struct{}
			if pending != nil {
				next = pending[1]
			}
			crossLayerGateState.mu.Unlock()
			if arm == "off" || arm == "on_demand_env" {
				if st != (CrossLayerPrefetchStats{}) || next != nil {
					t.Fatalf("knob off moved the ledger: stats=%+v next=%v", st, next)
				}
				return
			}
			if st.Predicted != 2 || st.Actual != 2 || st.Hits != 1 || st.Precision() != .5 || st.Recall() != .5 {
				t.Fatalf("reconcile stats=%+v, want 2 predicted, 2 actual, 1 hit", st)
			}
			if arm == "hash_next" {
				if st.HashRouted != 1 || next != nil {
					t.Fatalf("hash-routed next layer: stats=%+v next=%v, want recorded without a prediction", st, next)
				}
				return
			}
			want := m.v41PredictLayerPicks(1, x)
			if len(want) != m.Cfg.NumExpertsPerTok || len(next) != len(want) {
				t.Fatalf("layer 1 prediction=%v pending=%v", want, next)
			}
			for _, pk := range want {
				if _, ok := next[pk.expert]; !ok {
					t.Fatalf("pending prediction %v misses predicted expert %d", next, pk.expert)
				}
			}
			if s.expertRing != nil && (len(s.expertRing.holds) != 0 || s.ringDepth != 0) {
				t.Fatal("cross-layer prefetch left demand holds behind")
			}
		})
	}
}

// The predictor is the real router applied to the given input: same gate, same bias,
// same full-geometry top-k.
// fak-test:runtime fast est=100ms lane=default
func TestV41CrossLayerPredictorMatchesHostRoute(t *testing.T) {
	m := v41IncrementalExpertFixture(t, true, false)
	_, x := v41RoutedLayerFaultPicks(m)
	got := m.v41PredictLayerPicks(0, x)
	cfg, err := v41RouterConfigFullGeometry(m.Cfg)
	if err != nil {
		t.Fatal(err)
	}
	var bias []float32
	if m.has(m.v41LeafName(0, v41LeafGateBias)) {
		bias = m.v41LeafTensor(0, v41LeafGateBias)
	}
	E, H := m.Cfg.NumExperts, m.Cfg.HiddenSize
	want, err := v41RouteForGeometry(matRows(m.tensor(layerName(0, "ffn.gate.weight")), x, E, H), bias, cfg, true)
	if err != nil || len(want) == 0 || !reflect.DeepEqual(got, want) {
		t.Fatalf("predicted=%v want host route=%v err=%v", got, want, err)
	}
	if m.v41PredictLayerPicks(0, x[:1]) != nil || m.v41PredictLayerPicks(7, x) != nil {
		t.Fatal("predictor did not refuse a malformed input or an absent router")
	}
}

// Predicted hints follow the read-ahead rules: the longest pick-order prefix the ring
// could pin, advice only (no read, no ring staging, no hold), resident projections
// skipped, nothing for an on-demand session; the prediction is recorded either way.
// fak-test:runtime fast est=300ms lane=default
func TestV41CrossLayerPredictedHintsFollowReadAheadRules(t *testing.T) {
	for _, mode := range []string{"fits", "prefix", "resident", "on_demand", "no_ring"} {
		t.Run(mode, func(t *testing.T) {
			_, s, tier, tripleBytes, hints := v41CrossLayerFixture(t)
			t.Cleanup(func() { ResetCrossLayerPrefetchStats(s) })
			predicted := []routePick{{expert: 0, weight: 2}, {expert: 2, weight: 1}}
			want := []string{"gate_proj:0", "up_proj:0", "down_proj:0", "gate_proj:2", "up_proj:2", "down_proj:2"}
			switch mode {
			case "prefix":
				s.ExpertRingBytes = tripleBytes
				want = want[:3]
			case "resident":
				for _, name := range []string{layerName(0, "ffn.experts.0.w1.weight"), layerName(0, "ffn.experts.0.w3.weight"), layerName(0, "ffn.experts.0.w2.weight")} {
					_, release, ok := s.expertInputDeviceWeight(name)
					if !ok {
						t.Fatal("resident setup declined")
					}
					release()
				}
				*hints = nil
				want = want[3:]
			case "on_demand":
				s.ExpertPrefetch = ExpertPrefetchOnDemand
				want = nil
			case "no_ring":
				s.ExpertRingBytes = 0
				want = nil
			}
			before := tier.Stats()
			s.v41HintPredictedPicks(0, predicted)
			if !reflect.DeepEqual(*hints, want) && !(len(*hints) == 0 && len(want) == 0) {
				t.Fatalf("hints=%v want %v", *hints, want)
			}
			st := tier.Stats()
			if st.Reads != before.Reads || st.Prefetched != before.Prefetched || st.ResidentBytes != before.ResidentBytes || len(tier.loading) != 0 || len(tier.ready) != 0 {
				t.Fatalf("a hint became a read or a stage: before=%+v after=%+v", before, st)
			}
			if r := s.expertRing; r != nil {
				if len(r.holds) != 0 || len(r.heldPins) != 0 || s.ringDepth != 0 {
					t.Fatal("hints left ring holds behind")
				}
				if mode != "resident" && r.isResident(layerName(0, "ffn.experts.2.w1.weight")) {
					t.Fatal("a hint staged into the ring")
				}
			}
			ledger := CrossLayerPrefetchStatsFor(s)
			if wantBytes := st.Readahead.Bytes - before.Readahead.Bytes; ledger.HintedBytes != wantBytes || (len(want) > 0) != (wantBytes > 0) {
				t.Fatalf("hinted bytes=%d, tier advised %d", ledger.HintedBytes, wantBytes)
			}
			s.recordNextLayerActual(0, predicted[:1])
			if got := CrossLayerPrefetchStatsFor(s); got.Predicted != 2 || got.Actual != 1 || got.Hits != 1 {
				t.Fatalf("prediction not recorded for reconcile: %+v", got)
			}
		})
	}
}
