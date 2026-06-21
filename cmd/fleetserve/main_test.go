package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestBuildResultPromptsGeometryAndDeterminism(t *testing.T) {
	const (
		turns        = 4
		agents       = 3
		resultTokens = 5
		vocab        = 97
		rep          = 2
	)

	got := buildResultPrompts(turns, agents, resultTokens, vocab, rep)
	if len(got) != turns-1 {
		t.Fatalf("turn result prompt groups = %d, want %d", len(got), turns-1)
	}
	for turn := range got {
		if len(got[turn]) != agents {
			t.Fatalf("turn %d agents = %d, want %d", turn, len(got[turn]), agents)
		}
		for agent := range got[turn] {
			if len(got[turn][agent]) != resultTokens {
				t.Fatalf("turn %d agent %d tokens = %d, want %d", turn, agent, len(got[turn][agent]), resultTokens)
			}
			for _, id := range got[turn][agent] {
				if id < 0 || id >= vocab {
					t.Fatalf("turn %d agent %d id %d outside vocab %d", turn, agent, id, vocab)
				}
			}
		}
	}

	again := buildResultPrompts(turns, agents, resultTokens, vocab, rep)
	if got[1][2][3] != again[1][2][3] {
		t.Fatalf("result prompt generation is not deterministic: %d != %d", got[1][2][3], again[1][2][3])
	}
	if got[0][0][0] == got[1][0][0] || got[0][0][0] == got[0][1][0] {
		t.Fatalf("turn/agent result prompts did not diverge as expected")
	}
}

func TestBuildResultPromptsDisabledForSingleTurnOrNoResults(t *testing.T) {
	if got := buildResultPrompts(1, 2, 5, 97, 0); got != nil {
		t.Fatalf("single-turn result prompts = %#v, want nil", got)
	}
	if got := buildResultPrompts(3, 2, 0, 97, 0); got != nil {
		t.Fatalf("zero-token result prompts = %#v, want nil", got)
	}
}

func TestRunTurnsAdvancesCacheForDecodeAndPrivateResults(t *testing.T) {
	cfg := model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 101, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := model.NewSynthetic(cfg)
	bs := m.NewBatchSession(3)
	prompts := [][]int{
		{1, 2},
		{3, 4, 5},
		{6},
	}
	bs.PrefillEach(prompts)

	ids0 := []int{7, 8, 9}
	idsCopy := append([]int(nil), ids0...)
	resultPrompts := buildResultPrompts(3, len(ids0), 2, cfg.VocabSize, 0)
	runTurns(bs, ids0, resultPrompts, 4, cfg.VocabSize)
	for i := range ids0 {
		if ids0[i] != idsCopy[i] {
			t.Fatalf("runTurns mutated ids0[%d] = %d, want %d", i, ids0[i], idsCopy[i])
		}
	}

	wantExtra := 3*4 + 2*2
	for agent, s := range bs.Seqs {
		wantLen := len(prompts[agent]) + wantExtra
		if got := s.Cache.Len(); got != wantLen {
			t.Fatalf("agent %d cache len = %d, want %d", agent, got, wantLen)
		}
	}
}
