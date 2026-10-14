package model

import (
	"errors"
	"reflect"
	"testing"
)

// v41RatioOneForwardFixture is the published layer-20/21+ topology at two
// layers: layer 0 is the ratio-one KV and index-key source, layer 1 a ratio-one
// reader of it (an index-query source too when readerQueries). Neither layer
// owns a gate, and the reader owns no compressor or index-key tensors.
func v41RatioOneForwardFixture(t *testing.T, readerQueries bool) *Model {
	t.Helper()
	m := v41CompressorTestFixtureIndex(t, 128)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	d41 := m.Cfg.DeepSeekV41
	d41.CompressRatios = []int{1, 1}
	d41.KVSourceLayerIDs, d41.IndexSourceLayerIDs = []int{0}, []int{0}
	if readerQueries {
		d41.IndexSourceLayerIDs = []int{0, 1}
	}
	m.Cfg.Window = []int{3, 3}
	for layer := 0; layer < 2; layer++ {
		delete(m.manifest, layerName(layer, "attn.compressor.wgate.weight"))
	}
	for _, leaf := range []string{"attn.compressor.wkv.weight", "attn.compressor.norm.weight", "indexer.wk.weight", "indexer.k_norm.weight"} {
		delete(m.manifest, layerName(1, leaf))
	}
	if !readerQueries {
		delete(m.manifest, layerName(1, "indexer.wq_b.weight"))
		delete(m.manifest, layerName(1, "indexer.weights_proj.weight"))
	}
	return m
}

// v41RatioOneCall is one captured combined contraction: the query, sink and
// canonical (window-unoffset) compressed IDs the forward selected, plus its
// output and the position's own projected window row.
type v41RatioOneCall struct {
	q, sink, windowRow, out []float32
	ids                     []int32
}

// Ratio-one prefill and two incremental decode steps match an independent
// composition oracle (stepRatioOne + v41CombinedAttention, source:
// inference/model.py:700-780 at dba1be0a40aa45a94ad051997016db3960a90277) at
// every position of both the source and the reader, and the decode logits match
// a full-history recompute. Host F32 only: not FP8/FP4 or device parity.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=20s lane=default
func TestV41RatioOneForwardParity(t *testing.T) {
	for _, readerQueries := range []bool{false, true} {
		t.Run("reader-queries="+itoa(boolToIntV41Expert(readerQueries)), func(t *testing.T) {
			m := v41RatioOneForwardFixture(t, readerQueries)
			cfg := m.Cfg
			if err := m.v41ForwardAdmitted(); err != nil {
				t.Fatalf("sourced ratio-one schedule refused: %v", err)
			}
			hd, nH := cfg.HeadDim, cfg.NumHeads
			calls := [2]map[int]v41RatioOneCall{{}, {}}
			var carriers [][]float32
			gateProjections := 0
			bind := func(st *v41ForwardState, start int) {
				pos := [2]int{start, start}
				st.setDenseProjection(func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
					switch {
					case leaf == "attn.compressor.wgate.weight":
						gateProjections++
					case layer == 0 && leaf == "attn.compressor.wkv.weight":
						if rows != 1 {
							t.Fatalf("ratio-one producer projected %d carriers in one call", rows)
						}
						carriers = append(carriers, append([]float32(nil), panel...))
					case layer == 1 && leaf == "attn.compressor.wkv.weight":
						t.Fatal("ratio-one reader projected a private compressor row")
					}
					return nil, v41ProjectionDeclined, nil
				})
				st.sharedAttention = func(layer int, r v41SharedAttentionRequest) ([]float32, error) {
					r.kv, r.rowKV = v41TestAttentionFlatKV(r), nil
					at := pos[layer]
					pos[layer]++
					width := min(at+1, cfg.Window[layer])
					if len(r.idx) < width {
						t.Fatalf("layer %d position %d dropped its window", layer, at)
					}
					ids := append([]int32(nil), r.idx[width:]...)
					for i, id := range ids {
						if id >= 0 {
							ids[i] = id - int32(width)
						}
					}
					out, err := V41SparseAttentionSink(r.q, r.kv, r.sink, r.idx, r.plain)
					if err != nil {
						return nil, err
					}
					calls[layer][at] = v41RatioOneCall{
						q: append([]float32(nil), r.q...), sink: append([]float32(nil), r.sink...),
						windowRow: append([]float32(nil), r.kv[(width-1)*hd:width*hd]...),
						out:       append([]float32(nil), out...), ids: ids,
					}
					return out, nil
				}
			}

			prefix := []int{1, 2, 3, 4}
			decode := []int{5, 6}
			st := &v41ForwardState{}
			bind(st, 0)
			if _, err := m.forwardV41(prefix, st); err != nil {
				t.Fatalf("ratio-one prefill: %v", err)
			}
			history := append([]int(nil), prefix...)
			for i, token := range decode {
				bind(st, len(prefix)+i)
				logits, stats, err := m.forwardV41Step(token, st, nil)
				if err != nil || !stats.Committed {
					t.Fatalf("ratio-one decode step %d: %v", i, err)
				}
				history = append(history, token)
				assertV41LogitsParity(t, m, history, logits, 1e-4)
			}
			if gateProjections != 0 {
				t.Fatalf("ratio one projected %d gate rows, want none", gateProjections)
			}
			total := len(history)
			if len(carriers) != total {
				t.Fatalf("source projected %d carriers, want one per token (%d)", len(carriers), total)
			}

			newState := func() *V41AttentionState {
				s, err := NewV41AttentionState(hd, 1)
				if err != nil {
					t.Fatal(err)
				}
				s.windowSize, s.window, s.indexHeadDim = 3, s.window[:3], cfg.IndexHeadDim
				return s
			}
			source, reader, registry := newState(), newState(), newState()
			normWeight := m.tensor(layerName(0, "attn.compressor.norm.weight"))
			eps := float32(cfg.RMSNormEps)
			for at := 0; at < total; at++ {
				for layer, state := range []*V41AttentionState{source, reader} {
					call, ok := calls[layer][at]
					if !ok {
						t.Fatalf("layer %d position %d ran no combined contraction", layer, at)
					}
					request := v41RatioOneStep{
						Layer: layer, KVSourceLayer: 0, IndexKeySourceLayer: -1, Pos: at, Window: cfg.Window[layer], Heads: nH,
						WindowKV: call.windowRow,
						Consume: func(window, compressed, _ [][]float32) ([]float32, error) {
							return v41CombinedAttention(layer, at, 1, nH, hd, call.q, call.sink, window, compressed, call.ids, cfg.attnScale(), nil)
						},
					}
					if readerQueries || layer == 0 {
						request.IndexKeySourceLayer = 0
					}
					if layer == 0 {
						request.Input, request.NormWeight, request.Epsilon = carriers[at], normWeight, eps
						request.ProjectKV = func(in []float32) ([]float32, error) {
							return m.v41ProjMatRows(0, "attn.compressor.wkv.weight", in, hd, cfg.HiddenSize)
						}
						request.ProjectIndex = func(latent []float32) ([]float32, error) {
							keys, err := m.v41IndexKeys(0, [][]float32{latent}, nil)
							if err != nil {
								return nil, err
							}
							return keys[0], nil
						}
						request.Finalize = func(start int, latent, key []float32) error {
							return m.v41CompressedPublicationRoPEWithDevice(0, start, latent, key, nil)
						}
					}
					want, err := state.stepRatioOne(registry, request)
					if err != nil {
						t.Fatalf("oracle layer %d position %d: %v", layer, at, err)
					}
					if !reflect.DeepEqual(call.out, want) {
						t.Fatalf("layer %d position %d contraction differs from the ratio-one composition oracle", layer, at)
					}
				}
			}
			for _, kind := range []string{"kv", "index"} {
				var got, want [][]float32
				var gotOK, wantOK bool
				if kind == "kv" {
					got, gotOK = st.attn.KVSourceRows(0)
					want, wantOK = registry.KVSourceRows(0)
				} else {
					got, gotOK = st.attn.IndexKeys(0)
					want, wantOK = registry.IndexKeys(0)
				}
				if !gotOK || !wantOK || len(got) != total || !reflect.DeepEqual(got, want) {
					t.Fatalf("published %s source rows differ from the oracle's", kind)
				}
			}
			if _, ok := st.attn.KVSourceRows(1); ok {
				t.Fatal("ratio-one reader published private KV rows")
			}
		})
	}
}

// The production Session routes ratio-one decode through the incremental seam,
// not the full-history fallback, and each step's logits match a recompute.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=15s lane=default
func TestV41RatioOneSessionIncremental(t *testing.T) {
	m := v41RatioOneForwardFixture(t, true)
	s := m.NewSession()
	defer s.Close()
	prefix := []int{1, 2, 3}
	s.Prefill(prefix)
	if !s.v41IncrementalEligible() {
		t.Fatal("prefilled ratio-one session is not incrementally eligible")
	}
	fired := 0
	prev := v41IncrementalStepProbe
	v41IncrementalStepProbe = func() { fired++ }
	defer func() { v41IncrementalStepProbe = prev }()
	history := append([]int(nil), prefix...)
	for _, token := range []int{4, 5} {
		logits := s.Step(token)
		history = append(history, token)
		assertV41LogitsParity(t, m, history, logits, 1e-4)
	}
	if fired != 2 {
		t.Fatalf("incremental probe fired %d times, want 2", fired)
	}
}

// Ratio-one admission is a closed contract over its sources: the published
// topology admits, and each malformed source refuses with ErrV41ForwardStage at
// the offending layer before any arithmetic.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=2s lane=default
func TestV41RatioOneAdmission(t *testing.T) {
	refuses := func(t *testing.T, m *Model, layer int) {
		t.Helper()
		err := m.v41ForwardAdmitted()
		var stage *V41ForwardError
		if !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &stage) || stage.Layer != layer {
			t.Fatalf("admission = %v, want ErrV41ForwardStage at layer %d", err, layer)
		}
		if act, err := m.forwardV41([]int{1, 2}, nil); act != nil || !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("refused schedule ran: act=%v err=%v", act, err)
		}
	}
	for _, c := range []struct {
		name  string
		layer int
		edit  func(*Model)
	}{
		{"source-missing-compressor-wkv", 0, func(m *Model) { delete(m.manifest, layerName(0, "attn.compressor.wkv.weight")) }},
		{"source-missing-compressor-norm", 0, func(m *Model) { delete(m.manifest, layerName(0, "attn.compressor.norm.weight")) }},
		{"source-missing-index-key", 0, func(m *Model) { delete(m.manifest, layerName(0, "indexer.wk.weight")) }},
		{"self-producer-without-compressor", 1, func(m *Model) {
			m.Cfg.DeepSeekV41.CompressRatios = []int{0, 1}
			m.Cfg.DeepSeekV41.KVSourceLayerIDs, m.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil, nil
		}},
		{"window-only-kv-source", 1, func(m *Model) {
			m.Cfg.DeepSeekV41.CompressRatios = []int{1, 0}
			m.Cfg.DeepSeekV41.KVSourceLayerIDs = []int{1}
		}},
		{"index-reader-of-keyless-source", 1, func(m *Model) { m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{1} }},
		{"negative-ratio", 1, func(m *Model) { m.Cfg.DeepSeekV41.CompressRatios = []int{1, -1} }},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := v41RatioOneForwardFixture(t, true)
			c.edit(m)
			refuses(t, m, c.layer)
		})
	}
	for _, readerQueries := range []bool{false, true} {
		if err := v41RatioOneForwardFixture(t, readerQueries).v41ForwardAdmitted(); err != nil {
			t.Fatalf("sourced ratio-one schedule (reader queries=%t) refused: %v", readerQueries, err)
		}
	}
}

// The published Flash schedule (ratio 0/2/1 mix, KV sources 2/8/14/20) resolves
// every ratio-one layer onto source 20 at its own row span, so only layer 20
// projects a compressed stream for layers 20-39, the reader/source ratio rule
// (src/models/deepseek41.cpp:236-243@5210c7c) admits it and refuses a reader
// moved to another ratio, and candidate source 20 is admitted within its pool.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=50ms lane=default
func TestV41RatioOnePublishedSchedule(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	roles := v41AttentionRoles(cfg)
	ratioOne := 0
	for layer := 0; layer < cfg.NumLayers; layer++ {
		plan, err := v41AttentionPlanFor(cfg, layer, roles)
		if err != nil {
			t.Fatalf("layer %d plan: %v", layer, err)
		}
		if plan.Ratio != 1 {
			continue
		}
		ratioOne++
		if plan.KVSourceLayer != 20 || plan.kvGroupSize(cfg) != 1 || v41OwnsCompressedRows(plan) != (layer == 20) {
			t.Fatalf("ratio-one layer %d plan=%+v, want one-row spans from source 20 owned only by 20", layer, plan)
		}
	}
	if ratioOne != 20 {
		t.Fatalf("published schedule has %d ratio-one layers, want 20", ratioOne)
	}
	if err := v41ReaderSourceRatioAdmitted(cfg); err != nil {
		t.Fatalf("published schedule refused: %v", err)
	}
	// Candidate source 20 (2048 blocks of 8) is the identity through 16384
	// ratio-one rows and refuses typed past them (deepseek41.cpp:32-34@5210c7c).
	if err := v41CandidateSourceForwardAdmitted(cfg); err != nil {
		t.Fatalf("published candidate source refused: %v", err)
	}
	if err := v41CandidatePoolCovers(cfg, 16384); err != nil {
		t.Fatalf("16384 positions refused inside the candidate pool: %v", err)
	}
	var pool *V41ForwardError
	if err := v41CandidatePoolCovers(cfg, 16385); !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &pool) || pool.Layer != 20 {
		t.Fatalf("16385 positions: %v, want ErrV41ForwardStage at candidate source 20", err)
	}
	cfg.DeepSeekV41.CompressRatios = append([]int(nil), cfg.DeepSeekV41.CompressRatios...)
	cfg.DeepSeekV41.CompressRatios[21] = 2
	err := v41ReaderSourceRatioAdmitted(cfg)
	var stage *V41ForwardError
	if !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &stage) || stage.Layer != 21 {
		t.Fatalf("ratio-2 reader of a ratio-one source: %v, want ErrV41ForwardStage at layer 21", err)
	}
}
