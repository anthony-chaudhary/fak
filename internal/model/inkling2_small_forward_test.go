package model

import (
	"math"
	"testing"
)

// inkling2SmallTinyModel is the tiny loop model with its core removed, the
// shape of the loopless Inkling-2-Small graph.
func inkling2SmallTinyModel(at int) (Inkling2LoopHParams, *Inkling2LoopWeights) {
	hp, w := inkling2TinyModel()
	hp.CoreStart, hp.CoreEnd, hp.Passes = at, at, 1
	w.Controllers = nil
	return hp, w
}

func inkling2SmallRun(t *testing.T, hp Inkling2LoopHParams, w *Inkling2LoopWeights) [][]float32 {
	t.Helper()
	h, err := NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	if h.Slots() != hp.Layers || h.Visits() != hp.Layers {
		t.Fatalf("slots %d visits %d, want %d physical layers", h.Slots(), h.Visits(), hp.Layers)
	}
	var out [][]float32
	toks := []int{1, 4, 2, 7, 3}
	for _, tok := range toks {
		out = append(out, append([]float32(nil), h.Step(tok)...))
	}
	for s := range h.keys {
		if len(h.keys[s]) != len(toks) {
			t.Fatalf("slot %d holds %d keys, want one per decoded token (%d)", s, len(h.keys[s]), len(toks))
		}
	}
	return out
}

func TestInkling2SmallEmptyCoreAcceptedAndFinite(t *testing.T) {
	for _, at := range []int{0, 3, 6} {
		hp, w := inkling2SmallTinyModel(at)
		for i, logits := range inkling2SmallRun(t, hp, w) {
			if len(logits) != hp.VocabUnpadded {
				t.Fatalf("core at %d step %d: %d logits, want %d", at, i, len(logits), hp.VocabUnpadded)
			}
			for j, v := range logits {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)) > 30 {
					t.Fatalf("core at %d step %d logit %d = %v", at, i, j, v)
				}
			}
		}
	}
}

func TestInkling2SmallEmptyCoreRejectsExtraPasses(t *testing.T) {
	hp, w := inkling2SmallTinyModel(2)
	hp.Passes = 2
	if _, err := NewInkling2LoopHost(hp, w); err == nil {
		t.Fatal("empty loop core with 2 passes accepted")
	}
	hp, w = inkling2SmallTinyModel(7)
	if _, err := NewInkling2LoopHost(hp, w); err == nil {
		t.Fatal("empty loop core past the last layer accepted")
	}
}

// The first pass never applies controllers, so one pass over a dummy core must
// equal the loopless forward bit for bit.
func TestInkling2SmallEmptyCoreMatchesOnePassDummyCore(t *testing.T) {
	hp, w := inkling2SmallTinyModel(2)
	small := inkling2SmallRun(t, hp, w)

	dhp, dw := inkling2SmallTinyModel(2)
	dhp.CoreEnd = 4
	dw.Controllers = make([]Inkling2LoopControllerWeights, dhp.CoreEnd-dhp.CoreStart)
	dummy := inkling2SmallRun(t, dhp, dw)

	for i := range small {
		for j := range small[i] {
			if math.Float32bits(small[i][j]) != math.Float32bits(dummy[i][j]) {
				t.Fatalf("step %d logit %d: empty core %v, dummy core %v", i, j, small[i][j], dummy[i][j])
			}
		}
	}
}
