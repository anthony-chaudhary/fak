package model

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=100ms
func TestV41PrefillTopKReaderCopy(t *testing.T) {
	m := &Model{}
	plan := V41AttentionPlan{Layer: 1, Ratio: 4, TopKWidth: 3, IndexSourceLayer: 0}
	// Retain the old snapshot-then-normalize reader as an independent oracle.
	legacy := func(st *v41ForwardState) ([]int32, error) {
		return m.v41AttentionIndexListSnapshotOracle(plan, st, nil, 1, 4, 1)
	}
	for _, tc := range []struct {
		name  string
		rows  [][]int32
		unset bool
	}{
		{name: "unset", unset: true},
		{name: "nil"},
		{name: "empty", rows: [][]int32{}},
		{name: "padding-duplicates-published-length", rows: [][]int32{nil, {}, {2, 2, -1}, {0}}},
		{name: "earlier-bad-id-before-later-width", rows: [][]int32{{4}, {0, 1, 2, 3}}},
		{name: "width-before-own-bad-id", rows: [][]int32{{-2, 0, 1, 2}}},
		{name: "negative-id", rows: [][]int32{{-2}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state, err := NewV41AttentionState(1, 8)
			if err != nil {
				t.Fatal(err)
			}
			if !tc.unset {
				if err := state.PublishTopK(4, tc.rows); err != nil {
					t.Fatal(err)
				}
			}
			st := &v41ForwardState{attn: state}
			before, ratio, set := state.TopK()
			want, wantErr := legacy(st)
			got, gotErr := m.v41AttentionIndexList(plan, st, nil, 1, 4, 1)
			if !reflect.DeepEqual(got, want) || fmt.Sprint(gotErr) != fmt.Sprint(wantErr) || errors.Is(gotErr, ErrV41ForwardStage) != errors.Is(wantErr, ErrV41ForwardStage) {
				t.Fatalf("got %v / %v; want %v / %v", got, gotErr, want, wantErr)
			}
			if len(got) > 0 {
				got[0] = 99
			}
			after, afterRatio, afterSet := state.TopK()
			if !reflect.DeepEqual(before, after) || ratio != afterRatio || set != afterSet {
				t.Fatal("reader result aliases publication")
			}
		})
	}
	state, err := NewV41AttentionState(1, 8)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([][]int32, 64)
	for i := range rows {
		rows[i] = []int32{0, 1, 2}
	}
	if err := state.PublishTopK(4, rows); err != nil {
		t.Fatal(err)
	}
	st := &v41ForwardState{attn: state}
	rows[0][0] = 99
	snapshot, ratio, set := state.TopK()
	if snapshot[0][0] != 0 || ratio != 4 || !set {
		t.Fatal("publication aliases input")
	}
	snapshot[0][0] = 98
	snapshot[1] = []int32{97}
	owned, err := m.v41AttentionIndexList(plan, st, nil, 1, 4, 1)
	if err != nil || owned[0] != 0 || owned[3] != 0 {
		t.Fatalf("exported snapshot aliases publication: %v %v", owned, err)
	}
	before := append([]int32(nil), owned...)
	if err := state.PublishTopK(4, [][]int32{{3}}); err != nil {
		t.Fatal(err)
	}
	state.Reset()
	if !reflect.DeepEqual(owned, before) {
		t.Fatal("publication replacement/reset changed result")
	}
	for i := range rows {
		rows[i] = []int32{0, 1, 2}
	}
	if err := state.PublishTopK(4, rows); err != nil {
		t.Fatal(err)
	}
	var sink []int32
	var runErr error
	oldAllocs := testing.AllocsPerRun(20, func() { sink, runErr = legacy(st) })
	if runErr != nil || len(sink) != len(rows)*plan.TopKWidth {
		t.Fatal("invalid oracle allocation workload")
	}
	newAllocs := testing.AllocsPerRun(20, func() { sink, runErr = m.v41AttentionIndexList(plan, st, nil, 1, 4, 1) })
	if runErr != nil || len(sink) != len(rows)*plan.TopKWidth {
		t.Fatal("invalid reader allocation workload")
	}
	t.Logf("snapshot oracle %.0f allocations; reader %.0f; removed %.0f", oldAllocs, newAllocs, oldAllocs-newAllocs)
	if oldAllocs-newAllocs != float64(2*len(rows)+1) {
		t.Fatalf("want removal of outer snapshot plus copied and normalized storage for %d rows", len(rows))
	}
}

// Exact pre-optimization reader/source seam: retain normalization escape behavior
// as well as outputs when measuring the removed snapshot allocations.
func (m *Model) v41AttentionIndexListSnapshotOracle(plan V41AttentionPlan, st *v41ForwardState, localIdx []int32, headDim, groups, seq int) ([]int32, error) {
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
	published, _, ok := attn.TopK()
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
