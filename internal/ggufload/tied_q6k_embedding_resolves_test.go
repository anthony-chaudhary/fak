package ggufload

import "testing"

// ResolvesTiedQ6KEmbedding must report the loader's own tied-table decision, so a placement
// planner can refuse shapes the tied table cannot run in.
func TestResolvesTiedQ6KEmbeddingMatchesLoader(t *testing.T) {
	path, _, _ := tiedQ6KFixture(t)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		label string
		opts  []Q4KLoadOption
	}{
		{"default", nil},
		{"opt-out", []Q4KLoadOption{WithTiedQ6KEmbeddingResident(false)}},
		{"no dense Q6_K residency", []Q4KLoadOption{WithDenseKQuantResident(false)}},
	} {
		m, err := LoadModelQ4KProfileOptions(path, nil, tc.opts...)
		if err != nil {
			t.Fatalf("%s load: %v", tc.label, err)
		}
		want := m.TiedQ6KEmbeddingShared()
		m.CloseWeights()
		if got := ws.ResolvesTiedQ6KEmbedding(cfg, tc.opts...); got != want {
			t.Fatalf("%s: ResolvesTiedQ6KEmbedding = %v, loader shared = %v", tc.label, got, want)
		}
	}
	if !ws.ResolvesTiedQ6KEmbedding(cfg) {
		t.Fatal("fixture default must resolve the tied Q6_K table; the comparison above proved nothing")
	}
	var nilWS *WeightSource
	if nilWS.ResolvesTiedQ6KEmbedding(cfg) {
		t.Fatal("nil weight source must not resolve a tied table")
	}
}
