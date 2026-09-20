package model

import (
	"errors"
	"reflect"
	"testing"
)

func v41DecodeRows(count, width int) [][]float32 {
	rows := make([][]float32, count)
	for i := range rows {
		rows[i] = make([]float32, width)
		for j := range rows[i] {
			rows[i][j] = float32(i*width + j + 1)
		}
	}
	return rows
}

func v41DecodeStateModel(t *testing.T, layers int) *Model {
	t.Helper()
	m := v41ReducedModelLayers(t, layers)
	// The reduced temporal-state fixture does not carry Engram packed rows.
	m.Cfg.DeepSeekV41.EngramLayerIDs = nil
	return m
}

func TestV41DecodeStateSeedTemporalRetainsOnlyBoundedTails(t *testing.T) {
	state, err := NewV41AttentionState(2, 8)
	if err != nil {
		t.Fatal(err)
	}
	rows := v41DecodeRows(141, 2)
	if err := state.seedTemporal(rows, 4, 7); err != nil {
		t.Fatalf("seedTemporal: %v", err)
	}
	wantWindow := rows[len(rows)-7:]
	if got := state.retainedWindowKV(); !reflect.DeepEqual(got, wantWindow) {
		t.Fatalf("retained window = %v, want trailing configured window %v", got, wantWindow)
	}
	wantPartial := rows[len(rows)-1:]
	if got := state.partialGroupKV(); !reflect.DeepEqual(got, wantPartial) {
		t.Fatalf("partial compressor group = %v, want %v", got, wantPartial)
	}
	if got, want := state.retainedCopies, 8; got != want {
		t.Fatalf("retained copies = %d, want configured window 7 + partial group 1 = %d", got, want)
	}
	if state.retainedCopies >= len(rows) {
		t.Fatalf("retained copies = %d, copied the full %d-row prefix", state.retainedCopies, len(rows))
	}

	// Returned rows are inspection copies, not aliases into retained state.
	got := state.retainedWindowKV()
	got[0][0] = -1
	if state.retainedWindowKV()[0][0] != wantWindow[0][0] {
		t.Fatal("retainedWindowKV result aliases session state")
	}

	// Reseeding replaces the prior prefix and obeys the next configured window.
	if err := state.seedTemporal(rows[:140], 4, 3); err != nil {
		t.Fatalf("reseedTemporal: %v", err)
	}
	if got := state.retainedWindowKV(); !reflect.DeepEqual(got, rows[137:140]) {
		t.Fatalf("reseeded window = %v, want %v", got, rows[137:140])
	}
	if got := state.retainedCopies; got != 3 {
		t.Fatalf("reseeded retained copies = %d, want 3 without a partial group", got)
	}

	shortRows := v41DecodeRows(9, 2)
	shortAllocs := testing.AllocsPerRun(20, func() {
		if err := state.seedTemporal(shortRows, 4, 7); err != nil {
			panic(err)
		}
	})
	longAllocs := testing.AllocsPerRun(20, func() {
		if err := state.seedTemporal(rows, 4, 7); err != nil {
			panic(err)
		}
	})
	t.Logf("bounded seed allocations: short=%v long=%v", shortAllocs, longAllocs)
	if longAllocs > shortAllocs+1 {
		t.Fatalf("seed allocations grow with full prefix: short=%v long=%v", shortAllocs, longAllocs)
	}
	if got, want := state.retainedCopies, 8; got != want {
		t.Fatalf("long-prefix copies = %d, want same window+partial bound %d", got, want)
	}
}

func TestV41DecodeStateForwardKeepsIndependentLayerWindows(t *testing.T) {
	m := v41DecodeStateModel(t, 2)
	m.Cfg.Window = []int{1, 2}
	state := &v41ForwardState{}

	if _, err := m.forwardV41([]int{1, 3}, state); err != nil {
		t.Fatalf("forwardV41: %v", err)
	}
	if got, want := state.history, []int{1, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("history = %v, want %v", got, want)
	}
	layer0, layer1 := state.layerState(0), state.layerState(1)
	if layer0 == nil || layer1 == nil {
		t.Fatalf("layer states = (%p, %p), want two populated states", layer0, layer1)
	}
	if layer0 == layer1 {
		t.Fatal("decoder layers alias one attention state")
	}
	if got := len(layer0.retainedWindowKV()); got != 1 {
		t.Fatalf("layer 0 retained window rows = %d, want configured 1", got)
	}
	if got := len(layer1.retainedWindowKV()); got != 2 {
		t.Fatalf("layer 1 retained window rows = %d, want configured 2", got)
	}
	if got := state.retainedCopyCount(); got != 3 {
		t.Fatalf("retained layer copies = %d, want 1+2", got)
	}

	if _, err := m.forwardV41([]int{5}, state); err != nil {
		t.Fatalf("decode continuation: %v", err)
	}
	if got, want := state.history, []int{1, 3, 5}; !reflect.DeepEqual(got, want) {
		t.Fatalf("continued history = %v, want %v", got, want)
	}
	if got := len(state.layerState(0).retainedWindowKV()); got != 1 {
		t.Fatalf("continued layer 0 retained rows = %d, want 1", got)
	}
	if got := len(state.layerState(1).retainedWindowKV()); got != 2 {
		t.Fatalf("continued layer 1 retained rows = %d, want 2", got)
	}
	if got := state.retainedCopyCount(); got != 3 {
		t.Fatalf("continued retained layer copies = %d, want bounded 3", got)
	}
}

func TestV41DecodeStateContinuationMatchesStatelessForward(t *testing.T) {
	m := v41DecodeStateModel(t, 2)
	state := &v41ForwardState{}
	if _, err := m.forwardV41([]int{1, 3}, state); err != nil {
		t.Fatalf("stateful prefill: %v", err)
	}
	stateful, err := m.forwardV41([]int{5}, state)
	if err != nil {
		t.Fatalf("stateful continuation: %v", err)
	}
	stateless, err := m.forwardV41([]int{1, 3, 5}, nil)
	if err != nil {
		t.Fatalf("stateless full forward: %v", err)
	}
	if len(stateful.Logits) != len(stateless.Logits) {
		t.Fatalf("logit rows = %d, want %d", len(stateful.Logits), len(stateless.Logits))
	}
	const tolerance = 1e-6
	for row := range stateful.Logits {
		if len(stateful.Logits[row]) != len(stateless.Logits[row]) {
			t.Fatalf("logit row %d width = %d, want %d", row, len(stateful.Logits[row]), len(stateless.Logits[row]))
		}
		for col, got := range stateful.Logits[row] {
			want := stateless.Logits[row][col]
			delta := got - want
			if delta < 0 {
				delta = -delta
			}
			if delta > tolerance {
				t.Fatalf("logits[%d][%d] delta = %g, want <= %g (stateful=%g stateless=%g)", row, col, delta, tolerance, got, want)
			}
		}
	}
}

func TestV41DecodeStateSessionsDoNotAliasMutableLayerState(t *testing.T) {
	m := v41DecodeStateModel(t, 2)
	first, second := &v41ForwardState{}, &v41ForwardState{}
	if _, err := m.forwardV41([]int{1, 3}, first); err != nil {
		t.Fatalf("first session: %v", err)
	}
	if _, err := m.forwardV41([]int{1, 3}, second); err != nil {
		t.Fatalf("second session: %v", err)
	}
	for layer := 0; layer < 2; layer++ {
		if first.layerState(layer) == second.layerState(layer) {
			t.Fatalf("session layer %d aliases the same state pointer", layer)
		}
		if &first.layerState(layer).window[0][0] == &second.layerState(layer).window[0][0] {
			t.Fatalf("session layer %d aliases mutable window storage", layer)
		}
	}
	wantSecond := second.layerState(0).window[0][0]
	first.layerState(0).window[0][0] += 99
	if got := second.layerState(0).window[0][0]; got != wantSecond {
		t.Fatalf("first-session mutation changed second session: got %g want %g", got, wantSecond)
	}
}

func TestV41DecodeStateRegistryKeysSourceAndAbsoluteRange(t *testing.T) {
	registry, err := NewV41AttentionState(2, 8)
	if err != nil {
		t.Fatal(err)
	}
	updates := []V41AttentionStateUpdate{
		{Ref: V41AttentionStateRef{LayerID: 2, Ratio: 4, IsKVSource: true, IsIndexSource: true}, Latent: []float32{1, 2}, IndexKey: []float32{11, 12}},
		{Ref: V41AttentionStateRef{LayerID: 2, Ratio: 4, IsKVSource: true, IsIndexSource: true}, Latent: []float32{3, 4}, IndexKey: []float32{13, 14}},
		{Ref: V41AttentionStateRef{LayerID: 8, Ratio: 4, IsKVSource: true}, Latent: []float32{5, 6}},
	}
	if err := registry.publishUpdates(updates); err != nil {
		t.Fatalf("publishUpdates: %v", err)
	}
	wantKeys := []v41AttentionPublicationKey{
		{sourceLayer: 2, start: 0, end: 1},
		{sourceLayer: 2, start: 1, end: 2},
		{sourceLayer: 8, start: 0, end: 1},
	}
	for _, key := range wantKeys {
		if _, ok := registry.kvPublications[key]; !ok {
			t.Fatalf("missing KV publication key %+v", key)
		}
	}
	if got, ok := registry.KVSourceRows(2); !ok || !reflect.DeepEqual(got, [][]float32{{1, 2}, {3, 4}}) {
		t.Fatalf("source 2 rows = %v, %t", got, ok)
	}
	if got, ok := registry.KVSourceRows(8); !ok || !reflect.DeepEqual(got, [][]float32{{5, 6}}) {
		t.Fatalf("source 8 rows = %v, %t", got, ok)
	}
	if got, ok := registry.IndexKeys(2); !ok || !reflect.DeepEqual(got, [][]float32{{11, 12}, {13, 14}}) {
		t.Fatalf("source 2 index rows = %v, %t", got, ok)
	}
	if _, ok := registry.IndexKeys(8); ok {
		t.Fatal("KV-only source 8 unexpectedly resolved index publications")
	}
}

func TestV41DecodeStateLateLayerFailureRollsBackAllCommittedState(t *testing.T) {
	m := v41DecodeStateModel(t, 2)
	m.Cfg.Window = []int{1, 2}
	state := &v41ForwardState{}
	if _, err := m.forwardV41([]int{1, 3}, state); err != nil {
		t.Fatalf("initial forwardV41: %v", err)
	}

	wantHistory := append([]int(nil), state.history...)
	wantLayer0, wantLayer1 := state.layerState(0), state.layerState(1)
	wantWindow0 := wantLayer0.retainedWindowKV()
	wantWindow1 := wantLayer1.retainedWindowKV()
	wantCopies := state.retainedCopyCount()

	// This tensor is consumed late in layer 1, after its staged temporal state
	// has been built, so the failure proves commit is all-or-nothing.
	delete(m.manifest, layerName(1, "ffn.shared_experts.w2.weight"))
	_, err := m.forwardV41([]int{5}, state)
	if err == nil {
		t.Fatal("late layer tensor loss returned nil error")
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("late error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
	var typed *V41ForwardError
	if !errors.As(err, &typed) || typed.Layer != 1 {
		t.Fatalf("late error = %#v, want *V41ForwardError for layer 1", err)
	}

	if !reflect.DeepEqual(state.history, wantHistory) {
		t.Fatalf("history after failed decode = %v, want committed %v", state.history, wantHistory)
	}
	if state.layerState(0) != wantLayer0 || state.layerState(1) != wantLayer1 {
		t.Fatal("failed decode replaced a previously committed layer pointer")
	}
	if !reflect.DeepEqual(state.layerState(0).retainedWindowKV(), wantWindow0) ||
		!reflect.DeepEqual(state.layerState(1).retainedWindowKV(), wantWindow1) {
		t.Fatal("failed decode mutated previously committed layer content")
	}
	if got := state.retainedCopyCount(); got != wantCopies {
		t.Fatalf("retained copies after failed decode = %d, want committed %d", got, wantCopies)
	}
}

func TestV41DecodeStateResetClearsHistoryLayersAndRegistry(t *testing.T) {
	m := v41DecodeStateModel(t, 2)
	decode := &V41DecodeState{fwd: &v41ForwardState{}}
	if _, err := m.forwardV41([]int{1, 3}, decode.forwardState()); err != nil {
		t.Fatalf("forwardV41: %v", err)
	}
	decode.fwd.attn, _ = NewV41AttentionState(m.Cfg.HeadDim, 8)
	decode.Reset()
	if decode.fwd != nil || len(decode.History()) != 0 {
		t.Fatalf("reset state = fwd:%p history:%v, want released continuation", decode.fwd, decode.History())
	}
}
