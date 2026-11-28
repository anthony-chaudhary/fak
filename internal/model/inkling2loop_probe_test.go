package model

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestInkling2LoopPassShiftProbe(t *testing.T) {
	hp, w := inkling2TinyModel()
	over, err := hp.WithPassesUnbounded(4)
	if err != nil {
		t.Fatal(err)
	}
	h, err := NewInkling2LoopHost(over, w)
	if err != nil {
		t.Fatal(err)
	}
	h.PassProbe = &Inkling2LoopPassProbeConfig{Enabled: true, CollapseThreshold: 0.2}
	_ = h.Step(1)
	if h.LastPassProbe == nil || len(h.LastPassProbe.PassCosines) != 3 {
		t.Fatalf("expected 3 consecutive pass cosines for 4 passes, got %+v", h.LastPassProbe)
	}

	// Sign-flip / collapse detection on synthetic exit vectors.
	rep := evaluatePassProbe([][]float32{{1, 0, 0}, {-1, 0, 0}}, 0.2)
	if !rep.SignFlip || !rep.Collapse || rep.MinCosine >= 0 {
		t.Fatalf("expected sign-flip and collapse, got %+v", rep)
	}
	healthy := evaluatePassProbe([][]float32{{1, 0.1, 0}, {1, 0.12, 0}}, 0.2)
	if healthy.SignFlip || healthy.Collapse {
		t.Fatalf("expected healthy pass cosines, got %+v", healthy)
	}
}
