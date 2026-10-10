package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A projected-input wiring witness for inference/model.py:700-780 at
// dba1be0a40aa45a94ad051997016db3960a90277. Distinct per-layer window rows and
// zero queries make the one-sink denominator independently calculable. This
// does not qualify projection/norm numerics, FP8 window or FP4 cache arithmetic,
// full-model parity, or physical device execution. Runtime is not yet measured.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=10s lane=default
func TestV41CompressedWindowComposition(t *testing.T) {
	m := v41CompressorTestFixtureIndex(t, 128)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	m.Cfg.Window = []int{2, 3}
	m.Cfg.DeepSeekV41.CompressRatios = []int{2, 2}
	m.Cfg.IndexTopK = 3
	m.Cfg.RopeTheta, m.Cfg.DeepSeekV41.CompressRopeTheta = 256, 4096
	m.Cfg.RopeThetaPerLayer = nil
	m.Cfg.RopeScaling, m.Cfg.LongRope = "", nil
	m.Cfg.RopeFactor, m.Cfg.RopeOrigContext = 0, 0
	for _, leaf := range []string{"attn.compressor.wkv.weight", "attn.compressor.wgate.weight", "attn.compressor.norm.weight", "indexer.wk.weight", "indexer.k_norm.weight", "indexer.wq_b.weight", "indexer.weights_proj.weight"} {
		delete(m.manifest, layerName(1, leaf))
	}
	windowRow := func(layer, pos int) []float32 {
		row := make([]float32, m.Cfg.HeadDim)
		for d := range row {
			row[d] = float32(2+3*layer+pos) + float32(d%3)/8
		}
		return row
	}
	near := func(got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("output widths %d/%d", len(got), len(want))
		}
		for i := range want {
			if delta := math.Abs(float64(got[i]) - float64(want[i])); math.IsNaN(delta) || delta > 2e-5+2e-6*math.Abs(float64(want[i])) {
				t.Fatalf("output[%d]=%g want %g", i, got[i], want[i])
			}
		}
	}
	boom := errors.New("selected combined reader failure")
	failPos, failures, contractions, projections := -1, 0, 0, 0
	recordCold := true
	var coldOutputs [2][][]float32
	bind := func(st *v41ForwardState, start int) {
		projectPos, attendPos := [2]int{start, start}, [2]int{start, start}
		var expected [2][]float32
		st.denseProjection = func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
			if layer == 1 && (strings.HasPrefix(leaf, "attn.compressor.") || strings.HasPrefix(leaf, "indexer.")) {
				t.Fatalf("reader projected private source tensor %s", leaf)
			}
			return nil, v41ProjectionDeclined, nil
		}
		st.tailRoPE = func(layer int, q, kv, cos, sin []float32, heads, headDim, rotaryDim int) ([]float32, []float32, error) {
			pos := projectPos[layer]
			projectPos[layer]++
			return make([]float32, len(q)), windowRow(layer, pos), nil
		}
		st.sharedAttention = func(layer int, r v41SharedAttentionRequest) ([]float32, error) {
			pos := attendPos[layer]
			attendPos[layer]++
			contractions++
			width := min(pos+1, m.Cfg.Window[layer])
			source, _ := st.attn.KVSourceRows(0)
			if r.mode != compute.V41SharedAttentionPlain || r.plain.N != width+len(source) ||
				r.plain.TopK != width+3 || r.plain.Inverse != nil || len(r.idx) != width+3 {
				t.Fatalf("layer %d position %d missing combined sparse payload: %+v, %v", layer, pos, r.plain, r.idx)
			}
			for i := 0; i < width; i++ {
				if r.idx[i] != int32(i) || !reflect.DeepEqual(r.kv[i*m.Cfg.HeadDim:(i+1)*m.Cfg.HeadDim], windowRow(layer, pos-width+1+i)) {
					t.Fatalf("layer %d position %d lost its own window row %d", layer, pos, i)
				}
			}
			for group, row := range source {
				if !reflect.DeepEqual(r.kv[(width+group)*m.Cfg.HeadDim:(width+group+1)*m.Cfg.HeadDim], row) {
					t.Fatal("combined stream changed borrowed source bytes")
				}
			}
			for _, id := range r.idx[width:] {
				if id != -1 && (int(id) < width || (int(id)-width+1)*2 > pos+1) {
					t.Fatal("compressed selection lost source-width causality or padding")
				}
			}
			if layer == 1 && pos >= 1 && !reflect.DeepEqual(r.idx[width:], []int32{int32(width), int32(width), -1}) {
				t.Fatal("reader deduplicated or incorrectly offset the published selection")
			}
			if layer == 1 && pos == failPos {
				failures++
				return nil, boom
			}
			out, err := V41SparseAttentionSink(r.q, r.kv, r.sink, r.idx, r.plain)
			if err != nil {
				return nil, err
			}
			// q=0 gives one equal score per valid slot, plus exactly one sink.
			want := make([]float32, len(out))
			denominator := math.Exp(float64(r.sink[0]))
			for _, id := range r.idx {
				if id >= 0 {
					denominator++
				}
			}
			for d := range want {
				var numerator float64
				for _, id := range r.idx {
					if id >= 0 {
						numerator += float64(r.kv[int(id)*m.Cfg.HeadDim+d])
					}
				}
				want[d] = float32(numerator / denominator)
			}
			near(out, want)
			if recordCold {
				coldOutputs[layer] = append(coldOutputs[layer], append([]float32(nil), out...))
			} else {
				near(out, coldOutputs[layer][pos])
			}
			expected[layer] = v41InverseOutputOracle(want, m.Cfg.NumHeads, m.Cfg.HeadDim, m.Cfg.QKRopeHeadDim, pos, 4096)
			if layer == 0 && pos >= 1 {
				// Supply a legal repeated canonical selection for the reader. Cold
				// stores one row per position; decode stores only the newest row.
				selection := min(pos, len(st.attn.topk)-1)
				st.attn.topk[selection] = []int32{0, 0, -1}
			}
			return out, nil
		}
		st.groupedOutput = func(layer int, out []float32, heads, headDim, groups, rank, dim int) ([]float32, v41DenseProjectionOutcome, error) {
			near(out, expected[layer]) // inverse output RoPE exactly once, on both layers
			projections++
			return make([]float32, dim), v41ProjectionHandled, nil
		}
	}
	checkWindow := func(st *v41ForwardState, positions int) {
		t.Helper()
		for layer := 0; layer < 2; layer++ {
			s := st.layerState(layer)
			width := min(positions, m.Cfg.Window[layer])
			if s.nextWindowPos != positions || s.nextCompressRow != positions || s.retainedWindowRows != width {
				t.Fatalf("layer %d window/compressor cursor or retention differs", layer)
			}
			for i, row := range s.retainedWindowKV() {
				if !reflect.DeepEqual(row, windowRow(layer, positions-width+i)) {
					t.Fatal("committed window lost its own projected history")
				}
			}
			if layer == 1 && (len(s.partialInputs) != 0 || len(s.kvPublications) != 0 || len(s.indexPublications) != 0) {
				t.Fatal("reader retained private compressor history")
			}
		}
	}

	ids := []int{1, 2, 3, 4, 5, 6}
	cold := &v41ForwardState{}
	bind(cold, 0)
	if _, err := m.forwardV41(ids, cold); err != nil {
		t.Fatal(err)
	}
	checkWindow(cold, len(ids))
	if contractions != 12 || projections != 12 {
		t.Fatal("cold path skipped or duplicated a window-plus-compressed contraction")
	}
	recordCold = false
	st := &v41ForwardState{}
	if _, err := st.attentionState(m.Cfg.HeadDim, 8, m.Cfg.IndexHeadDim); err != nil {
		t.Fatal(err)
	}
	for layer := 0; layer < 2; layer++ {
		s, err := NewV41AttentionState(m.Cfg.HeadDim, 8)
		if err != nil {
			t.Fatal(err)
		}
		// Exercise actual slot overwrite without a long-prefix fixture.
		s.windowSize, s.window, s.indexHeadDim = 3, s.window[:3], m.Cfg.IndexHeadDim
		st.setLayerState(layer, 2, s)
	}
	for pos, token := range ids {
		if pos == 0 || pos == 1 || pos == 3 {
			before := captureV41ForwardSnapshot(st)
			if pos == 3 {
				st = before.clone().restore()
			}
			bind(st, pos)
			failPos = pos
			calls := contractions
			got, stats, err := m.forwardV41Step(token, st, nil)
			var selected *V41SharedAttentionOperationError
			if got != nil || stats.Committed || stats.LayersRolledBack != 2 || !errors.Is(err, boom) ||
				!errors.As(err, &selected) || selected.Layer != 1 || contractions != calls+2 ||
				!reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
				t.Fatalf("position %d selected failure replayed or changed window/source state: %v", pos, err)
			}
			failPos = -1
		}
		bind(st, pos)
		if _, stats, err := m.forwardV41Step(token, st, nil); err != nil || !stats.Committed {
			t.Fatalf("position %d continuation: %v", pos, err)
		}
		checkWindow(st, pos+1)
	}
	if failures != 3 || contractions != 30 || projections != 27 {
		t.Fatalf("unexpected selected/finished/failure counts: %d/%d/%d", contractions, projections, failures)
	}
}
