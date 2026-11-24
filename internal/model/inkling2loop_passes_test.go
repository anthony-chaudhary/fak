package model

import (
	"errors"
	"testing"
)

func TestInkling2LoopWithPassesBoundsTheRequestByTheCheckpoint(t *testing.T) {
	hp, _ := inkling2TinyModel()
	for req, want := range map[int]int{0: 3, 1: 1, 2: 2, 3: 3} {
		got, err := hp.WithPasses(req)
		if err != nil || got.Passes != want {
			t.Fatalf("WithPasses(%d) = %d, %v; want %d", req, got.Passes, err, want)
		}
	}
	if hp.Passes != 3 {
		t.Fatalf("WithPasses mutated the loaded hparams: passes %d", hp.Passes)
	}
	for _, req := range []int{-1, 4, 48} {
		_, err := hp.WithPasses(req)
		var pe *Inkling2LoopPassesError
		if !errors.As(err, &pe) || pe.Requested != req || pe.Max != 3 {
			t.Fatalf("WithPasses(%d) err = %v, want typed refusal with max 3", req, err)
		}
	}
}

func TestInkling2LoopWithPassesSizesCacheAndMatchesABuildTimeCount(t *testing.T) {
	hp, w := inkling2TinyModel()
	for passes := 1; passes <= hp.Passes; passes++ {
		run, err := hp.WithPasses(passes)
		if err != nil {
			t.Fatal(err)
		}
		h, err := NewInkling2LoopHost(run, w)
		if err != nil {
			t.Fatal(err)
		}
		core := hp.CoreEnd - hp.CoreStart
		if want := hp.Layers + (passes-1)*core; h.Slots() != want || h.Visits() != want {
			t.Fatalf("%d passes: slots %d visits %d, want %d", passes, h.Slots(), h.Visits(), want)
		}
		built := hp
		built.Passes = passes
		ref, err := NewInkling2LoopHost(built, w)
		if err != nil {
			t.Fatal(err)
		}
		for _, tok := range []int{1, 5, 2} {
			got, want := h.Step(tok), ref.Step(tok)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%d passes token %d logit %d: runtime %v, build-time %v", passes, tok, i, got[i], want[i])
				}
			}
		}
	}
}
