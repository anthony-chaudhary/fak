package model

import (
	"fmt"
	"reflect"
	"testing"
)

func topKNormalizationFixture(t testing.TB, count, width int) (*Model, V41AttentionPlan, *v41ForwardState) {
	t.Helper()
	state, err := NewV41AttentionState(1, 8)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([][]int32, count)
	for i := range rows {
		rows[i] = make([]int32, i%(width+1))
		for j := range rows[i] {
			rows[i][j] = int32((i + j) % 7)
		}
	}
	if err := state.PublishTopK(4, rows); err != nil {
		t.Fatal(err)
	}
	return &Model{}, V41AttentionPlan{Layer: 1, Ratio: 4, TopKWidth: width, IndexSourceLayer: 0}, &v41ForwardState{attn: state}
}

// fak-test:runtime fast est=100ms
func TestV41TopKNormalizationAllocations(t *testing.T) {
	for _, count := range []int{1, 8, 64} {
		for _, width := range []int{3, 32} {
			t.Run(fmt.Sprintf("rows%d_width%d", count, width), func(t *testing.T) {
				m, p, st := topKNormalizationFixture(t, count, width)
				want, wantErr := m.v41AttentionIndexListNormalizeBaseline(p, st, nil, 1, 8, 1)
				got, gotErr := m.v41AttentionIndexList(p, st, nil, 1, 8, 1)
				if wantErr != nil || gotErr != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("got %v / %v want %v / %v", got, gotErr, want, wantErr)
				}
				// Output writes and append growth must not change retained publication.
				before, _, _ := st.attn.TopK()
				got[0] = 99
				got = append(got, 98)
				after, _, _ := st.attn.TopK()
				if !reflect.DeepEqual(before, after) {
					t.Fatal("result aliases publication")
				}
				got, gotErr = m.v41AttentionIndexList(p, st, nil, 1, 8, 1)
				if gotErr != nil || !reflect.DeepEqual(got, want) {
					t.Fatal("previous result mutation corrupted later read")
				}
				var sink []int32
				var err error
				baseline := testing.AllocsPerRun(20, func() { sink, err = m.v41AttentionIndexListNormalizeBaseline(p, st, nil, 1, 8, 1) })
				if err != nil || len(sink) != count*width {
					t.Fatal("invalid baseline workload")
				}
				candidate := testing.AllocsPerRun(20, func() { sink, err = m.v41AttentionIndexList(p, st, nil, 1, 8, 1) })
				if err != nil || len(sink) != count*width {
					t.Fatal("invalid candidate workload")
				}
				t.Logf("baseline=%.0f candidate=%.0f removed=%.0f", baseline, candidate, baseline-candidate)
				if candidate != 1 || baseline-candidate != float64(count) {
					t.Fatalf("want one owned flat allocation and %d eliminated row allocations", count)
				}
			})
		}
	}
}

var topKNormalizationBenchmarkSink []int32

func BenchmarkV41TopKNormalization(b *testing.B) {
	for _, count := range []int{1, 64, 512} {
		for _, width := range []int{8, 64} {
			m, p, st := topKNormalizationFixture(b, count, width)
			for _, baseline := range []bool{true, false} {
				name := "candidate"
				if baseline {
					name = "baseline"
				}
				b.Run(fmt.Sprintf("rows%d_width%d/%s", count, width, name), func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						var out []int32
						var err error
						if baseline {
							out, err = m.v41AttentionIndexListNormalizeBaseline(p, st, nil, 1, 8, 1)
						} else {
							out, err = m.v41AttentionIndexList(p, st, nil, 1, 8, 1)
						}
						if err != nil {
							b.Fatal(err)
						}
						topKNormalizationBenchmarkSink = out
					}
				})
			}
		}
	}
}

// Captured pre-change 7b08d6 reader/source implementation, including normalized
// row allocations and validation order. This is a CPU seam oracle, not a model
// throughput or native hardware witness.
func (m *Model) v41AttentionIndexListNormalizeBaseline(plan V41AttentionPlan, st *v41ForwardState, localIdx []int32, headDim, groups, seq int) ([]int32, error) {
	if plan.Ratio == 0 || plan.TopKWidth <= 0 || seq <= 0 {
		return nil, nil
	}
	isSource := indexSourceAt(m.Cfg.DeepSeekV41, plan.Layer)
	if !isSource && plan.IndexSourceLayer < 0 {
		return nil, nil
	}

	// one validates a resolved selection row against the compressed stream and
	// normalizes it to the published stride, padding a short top-k with -1 (the
	// reference's empty-slot marker) when fewer groups are visible than top-k.
	one := func(row []int32) ([]int32, error) {
		if len(row) > plan.TopKWidth {
			return nil, v41StageErr(v41StageIndexer, plan.Layer,
				fmt.Errorf("%w: layer %d index row width %d exceeds stride %d", ErrV41ForwardStage, plan.Layer, len(row), plan.TopKWidth))
		}
		normalized := make([]int32, plan.TopKWidth)
		for i := range normalized {
			normalized[i] = -1
		}
		for i, id := range row {
			if id < -1 || int(id) >= groups {
				return nil, v41StageErr(v41StageIndexer, plan.Layer,
					fmt.Errorf("%w: layer %d index row %d out of range for %d groups", ErrV41ForwardStage, plan.Layer, id, groups))
			}
			normalized[i] = id
		}
		return normalized, nil
	}

	if isSource {
		if localIdx == nil && !(plan.Ratio > 0 && groups == 0) {
			return nil, nil
		}
		row, err := one(localIdx)
		if err != nil {
			return nil, err
		}
		// Cold prefill resolves each source query separately. The normalized
		// row already owns its storage; a single position needs no second copy.
		if seq == 1 {
			return row, nil
		}
		flat := make([]int32, 0, seq*plan.TopKWidth)
		for t := 0; t < seq; t++ {
			flat = append(flat, row...)
		}
		return flat, nil
	}

	// Reader: reuse the source's published, per-position selection.
	if st == nil {
		return nil, v41StageErr(v41StageIndexer, plan.Layer,
			fmt.Errorf("%w: layer %d reads index source %d but no session state carries the publication", ErrV41ForwardStage, plan.Layer, plan.IndexSourceLayer))
	}
	attn, err := st.attentionState(headDim, 8)
	if err != nil {
		return nil, v41StageErr(v41StageIndexer, plan.Layer, err)
	}
	published, _, ok := attn.topKReadOnly()
	if !ok || len(published) == 0 {
		return nil, v41StageErr(v41StageIndexer, plan.Layer,
			fmt.Errorf("%w: layer %d reads index source %d but no top-k selection was published", ErrV41ForwardStage, plan.Layer, plan.IndexSourceLayer))
	}
	flat := make([]int32, 0, len(published)*plan.TopKWidth)
	for _, row := range published {
		normalized, err := one(row)
		if err != nil {
			return nil, err
		}
		flat = append(flat, normalized...)
	}
	return flat, nil
}
