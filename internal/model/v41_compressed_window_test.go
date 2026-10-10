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
	duplicateReader := false
	ids := []int{1, 2, 3, 4, 5, 6}
	sourceRows := make(map[int][][]float32)
	sourceIDs := make(map[int][]int32)
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
			// Cold forward owns a private transactional registry. Its payload
			// contains all completed prefix groups, even for an early query;
			// causal masking belongs to idx, not to the stored row count.
			groups := (pos + 1) / 2
			if recordCold {
				groups = len(ids) / 2
			}
			if r.mode != compute.V41SharedAttentionPlain || r.plain.N != width+groups ||
				r.plain.TopK != width+3 || r.plain.Inverse != nil || len(r.idx) != width+3 || len(r.kv) != (width+groups)*m.Cfg.HeadDim {
				t.Fatalf("layer %d position %d missing combined sparse payload: %+v, %v", layer, pos, r.plain, r.idx)
			}
			for i := 0; i < width; i++ {
				if r.idx[i] != int32(i) || !reflect.DeepEqual(r.kv[i*m.Cfg.HeadDim:(i+1)*m.Cfg.HeadDim], windowRow(layer, pos-width+1+i)) {
					t.Fatalf("layer %d position %d lost its own window row %d", layer, pos, i)
				}
			}
			if layer == 0 {
				rows := make([][]float32, groups)
				for group := range rows {
					rows[group] = append([]float32(nil), r.kv[(width+group)*m.Cfg.HeadDim:(width+group+1)*m.Cfg.HeadDim]...)
				}
				sourceRows[pos] = rows
				canonical := append([]int32(nil), r.idx[width:]...)
				for i, id := range canonical {
					if id >= 0 {
						canonical[i] -= int32(width)
					}
				}
				sourceIDs[pos] = canonical
			} else {
				if len(sourceRows[pos]) != groups {
					t.Fatal("reader has no matching source payload")
				}
				for group, row := range sourceRows[pos] {
					if !reflect.DeepEqual(r.kv[(width+group)*m.Cfg.HeadDim:(width+group+1)*m.Cfg.HeadDim], row) {
						t.Fatal("combined stream changed borrowed source bytes")
					}
				}
				wantIDs := append([]int32(nil), sourceIDs[pos]...)
				for i, id := range wantIDs {
					if id >= 0 {
						wantIDs[i] += int32(width)
					}
				}
				if !reflect.DeepEqual(r.idx[width:], wantIDs) {
					t.Fatal("reader changed or incorrectly offset the published selection")
				}
			}
			for _, id := range r.idx[width:] {
				if id != -1 && (int(id) < width || (int(id)-width+1)*2 > pos+1) {
					t.Fatal("compressed selection lost source-width causality or padding")
				}
			}
			if duplicateReader && layer == 1 && !reflect.DeepEqual(r.idx[width:], []int32{int32(width), int32(width), -1}) {
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
			} else if !duplicateReader {
				near(out, coldOutputs[layer][pos])
			}
			// Full attention publishes BF16 before inverse rotation and copies
			// the F32 complex result back to BF16 before grouped projection.
			expected[layer] = v41LatentNormOracleBF16(v41InverseOutputOracle(v41LatentNormOracleBF16(want), m.Cfg.NumHeads, m.Cfg.HeadDim, m.Cfg.QKRopeHeadDim, pos, 4096))
			if duplicateReader && layer == 0 {
				// Only the incremental call uses this active registry directly.
				// Its source has published exactly one newest selection row.
				if st.attn == nil || len(st.attn.topk) != 1 {
					t.Fatal("duplicate witness requires the active incremental registry")
				}
				st.attn.topk[0] = []int32{0, 0, -1}
				sourceIDs[pos] = append([]int32(nil), st.attn.topk[0]...)
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
		published, ok := st.attn.KVSourceRows(0)
		if positions >= 2 && (!ok || !reflect.DeepEqual(published, sourceRows[positions-1])) {
			t.Fatal("committed registry differs from the source contraction payload")
		}
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
	// Keep duplicate-ID reuse coverage on a real active continuation, after
	// the ordinary source selections have established cold/step parity above.
	duplicateReader = true
	bind(st, len(ids))
	if _, stats, err := m.forwardV41Step(7, st, nil); err != nil || !stats.Committed {
		t.Fatalf("duplicate selection continuation: %v", err)
	}
	checkWindow(st, len(ids)+1)
	if failures != 3 || contractions != 32 || projections != 29 {
		t.Fatal("duplicate selection skipped or repeated a contraction/projection")
	}
}
