package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// loopCkpt is a checkpoint with n KV slots, the layout one row holds at a loop depth.
func loopCkpt(pos, n int) *ink2RowCkpt {
	return &ink2RowCkpt{pos: pos, slots: make([]ink2SlotCkpt, n)}
}

// A store bound to a loop config keeps only checkpoints taken at that depth: a
// checkpoint at another pass count has a different slot layout and would restore
// into the wrong cache, so it is refused and never served.
func TestInkling2PrefixStoreRefusesOtherLoopDepth(t *testing.T) {
	hp := model.Inkling2LoopHParams{Layers: 4, CoreStart: 1, CoreEnd: 3, Passes: 3}
	st := &inkling2PrefixStore{capBytes: 1 << 20, minTokens: 2, loopSlots: inkling2LoopSlots(hp)}
	toks := seqTokens(4, 10)
	shallow := loopCkpt(len(toks), inkling2LoopSlots(model.Inkling2LoopHParams{Layers: 4, CoreStart: 1, CoreEnd: 3, Passes: 2}))
	st.put(toks, shallow)
	if len(st.entries) != 0 {
		t.Fatalf("a checkpoint at 2 passes was kept by a 3-pass store")
	}
	st.put(toks, loopCkpt(len(toks), st.loopSlots))
	if len(st.entries) != 1 {
		t.Fatalf("a checkpoint at the bound depth was refused: %d entries", len(st.entries))
	}
	if e := st.lookup(append(append([]int(nil), toks...), 1)); e == nil {
		t.Fatal("the bound-depth entry did not serve the next request")
	}
}

// Slot count grows with each extra core pass and is the layer count at one pass.
func TestInkling2LoopSlotsPerPassCount(t *testing.T) {
	hp := model.Inkling2LoopHParams{Layers: 4, CoreStart: 1, CoreEnd: 3, Passes: 1}
	for passes, want := range map[int]int{1: 4, 2: 6, 3: 8} {
		hp.Passes = passes
		if got := inkling2LoopSlots(hp); got != want {
			t.Fatalf("%d passes: %d slots, want %d", passes, got, want)
		}
	}
}
