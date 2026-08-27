package model

import "testing"

func TestDecideRecomputeWinAndReversal(t *testing.T) {
	win, err := DecideRecompute(RecomputeCandidate{Name: "rotary-phase", StoredBytes: 4096, ReadBytes: 4096, RecomputeFLOPs: 512, MemoryJoules: .004, ComputeJoules: .001, MemoryNanoseconds: 900, ComputeNanoseconds: 200, Deterministic: true}, 8)
	if err != nil {
		t.Fatal(err)
	}
	if !win.Recompute || win.RemovedBytes != 8192 || win.BytesRemovedPerAccepted != 1024 || win.NetNanoseconds != 700 {
		t.Fatalf("win=%+v", win)
	}
	reversal, _ := DecideRecompute(RecomputeCandidate{Name: "wide-exp", StoredBytes: 1024, ReadBytes: 1024, RecomputeFLOPs: 1e6, MemoryJoules: .001, ComputeJoules: .01, MemoryNanoseconds: 100, ComputeNanoseconds: 2000, Deterministic: true}, 1)
	if reversal.Recompute || reversal.RemovedBytes != 0 {
		t.Fatalf("reversal=%+v", reversal)
	}
	nondeterministic, _ := DecideRecompute(RecomputeCandidate{Name: "rng", StoredBytes: 1024, ReadBytes: 1024, MemoryJoules: 1, ComputeJoules: .1, MemoryNanoseconds: 1000, ComputeNanoseconds: 1}, 1)
	if nondeterministic.Recompute {
		t.Fatal("nondeterministic intermediate admitted")
	}
}

func TestDecideRecomputeRejectsInvalidEnvelope(t *testing.T) {
	if _, err := DecideRecompute(RecomputeCandidate{Name: "bad", StoredBytes: -1}, 1); err == nil {
		t.Fatal("negative bytes accepted")
	}
}
