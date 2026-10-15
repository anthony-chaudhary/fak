package model

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// The oracle retains the former snapshot-plus-normalize reader path.
func v41StepTopKCopyOracle(m *Model, state *V41AttentionState, l int, plan V41AttentionPlan, groups int) ([]int32, error) {
	if plan.TopKWidth <= 0 || groups <= 0 {
		return nil, nil
	}
	published, _, ok := state.TopK()
	if !ok || len(published) == 0 {
		return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: layer %d reads index source %d but no top-k selection was published", ErrV41ForwardStage, l, plan.IndexSourceLayer))
	}
	row := published[len(published)-1]
	if len(row) > plan.TopKWidth {
		return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: layer %d index row width %d exceeds stride %d", ErrV41ForwardStage, l, len(row), plan.TopKWidth))
	}
	out := make([]int32, plan.TopKWidth)
	for i := range out {
		out[i] = -1
	}
	for i, id := range row {
		if id < -1 || int(id) >= groups {
			return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: layer %d index row %d out of range for %d groups", ErrV41ForwardStage, l, id, groups))
		}
		out[i] = id
	}
	return out, nil
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41StepTopKCopyParity(t *testing.T) {
	m := &Model{}
	for _, tc := range []struct {
		name          string
		state         *V41AttentionState
		width, groups int
	}{
		{"nil", nil, 4, 8},
		{"missing", &V41AttentionState{}, 4, 8},
		{"empty-publication", &V41AttentionState{topkSet: true}, 4, 8},
		{"empty-row", &V41AttentionState{topkSet: true, topk: [][]int32{{}}}, 4, 8},
		{"latest-and-padding", &V41AttentionState{topkSet: true, topk: [][]int32{{99}, {3, 1, 3}}}, 4, 8},
		{"sentinel", &V41AttentionState{topkSet: true, topk: [][]int32{{-1, 0, 7, -1}}}, 4, 8},
		{"too-wide", &V41AttentionState{topkSet: true, topk: [][]int32{{0, 1, 2, 3, 4}}}, 4, 8},
		{"bad-negative", &V41AttentionState{topkSet: true, topk: [][]int32{{1, -2}}}, 4, 8},
		{"bad-range", &V41AttentionState{topkSet: true, topk: [][]int32{{1, 8}}}, 4, 8},
		{"zero-width", nil, 0, 8}, {"zero-groups", nil, 4, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := V41AttentionPlan{TopKWidth: tc.width, IndexSourceLayer: 2}
			want, we := v41StepTopKCopyOracle(m, tc.state, 3, plan, tc.groups)
			got, ge := m.v41RoleStepIndex(tc.state, 3, 7, plan, nil, nil, tc.groups)
			if !reflect.DeepEqual(got, want) || fmt.Sprint(ge) != fmt.Sprint(we) || errors.Is(ge, ErrV41ForwardStage) != errors.Is(we, ErrV41ForwardStage) {
				t.Fatalf("got %v/%v want %v/%v", got, ge, want, we)
			}
			if ge != nil {
				var stage *V41ForwardError
				if !errors.As(ge, &stage) || stage.Stage != v41StageIndexer || stage.Layer != 3 {
					t.Fatalf("wrong error stage: %v", ge)
				}
			}
			if ge != nil && reflect.TypeOf(ge) != reflect.TypeOf(we) {
				t.Fatalf("error type %T want %T", ge, we)
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41StepTopKCopyIsolation(t *testing.T) {
	m := &Model{}
	state := &V41AttentionState{ratioCap: 2}
	plan := V41AttentionPlan{TopKWidth: 4, IndexSourceLayer: 2}
	input := []int32{3, 1, 3}
	if err := state.PublishTopK(1, [][]int32{input}); err != nil {
		t.Fatal(err)
	}
	input[0] = 99
	out, err := m.v41RoleStepIndex(state, 3, 7, plan, nil, nil, 8)
	if err != nil {
		t.Fatal(err)
	}
	out[0] = 77 // A retaining/mutating attention callback owns only this result.
	public, _, _ := state.TopK()
	public[0][1] = 66
	next, err := m.v41RoleStepIndex(state, 3, 7, plan, nil, nil, 8)
	if err != nil || !reflect.DeepEqual(next, []int32{3, 1, 3, -1}) {
		t.Fatalf("publication aliased: %v %v", next, err)
	}
	if err := state.PublishTopK(1, [][]int32{{2}}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(next, []int32{3, 1, 3, -1}) {
		t.Fatal("later publication mutated returned row")
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41StepTopKCopyAllocations(t *testing.T) {
	m := &Model{}
	row := make([]int32, 512)
	for i := range row {
		row[i] = int32(i)
	}
	state := &V41AttentionState{topkSet: true, topk: [][]int32{row}}
	plan := V41AttentionPlan{TopKWidth: 512, IndexSourceLayer: 2}
	var result []int32
	before := testing.AllocsPerRun(100, func() { result, _ = v41StepTopKCopyOracle(m, state, 3, plan, 1024) })
	after := testing.AllocsPerRun(100, func() { result, _ = m.v41RoleStepIndex(state, 3, 1023, plan, nil, nil, 1024) })
	if len(result) != 512 || before-after != 1 || after != 1 {
		t.Fatalf("allocations before=%g after=%g", before, after)
	}
	t.Logf("reader allocations %g -> %g, published 30-reader trace saves %g allocations and %d payload bytes/token", before, after, 30*(before-after), 30*512*4)
}
