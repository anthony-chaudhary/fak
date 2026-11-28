package model

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestInkling2LoopRKSubstepCommitsCacheOnce(t *testing.T) {
	hp, w := inkling2TinyModel()
	ref, err := NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	rkHost, err := NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	rkHost.RK = &Inkling2LoopRKConfig{Window: 1, K: 3, Beta: 0.5}

	for _, tok := range []int{1, 5} {
		_ = ref.Step(tok)
		_ = rkHost.Step(tok)
	}
	if rkHost.KVRows() != ref.KVRows() || rkHost.Pos() != ref.Pos() {
		t.Fatalf("RK substep mutated cache count: got rows=%d pos=%d, want rows=%d pos=%d",
			rkHost.KVRows(), rkHost.Pos(), ref.KVRows(), ref.Pos())
	}
}
