package ggufload

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInkling2LoopGenerateRefusesLoopsPastTheCheckpoint(t *testing.T) {
	m := &Inkling2LoopModel{HP: model.Inkling2LoopHParams{Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3}}
	for _, loops := range []int{-1, 4} {
		_, _, err := Inkling2LoopGenerate(m, nil, Inkling2LoopRequest{Prompt: []int{1}, MaxTokens: 1, Loops: loops})
		var pe *model.Inkling2LoopPassesError
		if !errors.As(err, &pe) || pe.Max != 3 {
			t.Fatalf("loops %d: err %v, want typed refusal with max 3", loops, err)
		}
	}
	if m.HP.Passes != 3 {
		t.Fatalf("refused request mutated the loaded model: passes %d", m.HP.Passes)
	}
}
