package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInkling2ResidentCtxAutoSet(t *testing.T) {
	cases := []struct {
		name string
		env  map[string]string
		want bool
	}{
		{"unset derives", map[string]string{}, true},
		{"blank derives", map[string]string{"FAK_INKLING2_CTX": "  "}, true},
		{"explicit is fixed", map[string]string{"FAK_INKLING2_CTX": "16384"}, false},
	}
	for _, c := range cases {
		lookup := func(k string) (string, bool) { v, ok := c.env[k]; return v, ok }
		if got := inkling2ResidentCtxAutoSet(lookup); got != c.want {
			t.Errorf("%s: auto = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestInkling2FitResidentCtx(t *testing.T) {
	const gib = int64(1) << 30
	const perPos = 1 << 20 // 1 MiB per position keeps the arithmetic exact
	cases := []struct {
		name          string
		free, reserve int64
		perPos        int64
		want          int
	}{
		{"floors to 256 steps", 2*gib + 300*perPos, 0, perPos, 2304},
		{"reserve is subtracted", 2*gib + 300*perPos, 300 * perPos, perPos, 2048},
		{"exact multiple kept", 1256 * perPos, 0, perPos, 1024},
		{"below floor refuses", 511 * perPos, 0, perPos, 0},
		{"at floor fits", 512 * perPos, 0, perPos, 512},
		{"ceiling clamps", 64 * gib, 0, perPos, Inkling2ResidentCtxMax(inkling2ResidentBackend)},
		{"reserve swallows free", gib, 2 * gib, perPos, 0},
		{"zero per position refuses", gib, 0, 0, 0},
	}
	for _, c := range cases {
		if got := inkling2FitResidentCtx(c.free, c.reserve, c.perPos); got != c.want {
			t.Errorf("%s: ctx = %d, want %d", c.name, got, c.want)
		}
	}
}

func TestInkling2FitResidentCtxNeverExceedsFree(t *testing.T) {
	const perPos = 3 * 1000 * 1000
	for free := int64(1); free < 4<<30; free += 37 << 20 {
		reserve := inkling2ResidentReserve(free)
		ctx := inkling2FitResidentCtx(free, reserve, perPos)
		if ctx == 0 {
			continue
		}
		if need := int64(ctx) * perPos; need > free-reserve {
			t.Fatalf("free %d reserve %d: ctx %d needs %d bytes", free, reserve, ctx, need)
		}
	}
}

func TestInkling2ResidentPerPosBytes(t *testing.T) {
	// Layers 4 with local layers 0,1,2 and global layer 3; no revisited core.
	hp := model.Inkling2LoopHParams{
		Layers: 4, Passes: 1, CoreStart: 0, CoreEnd: 0,
		Heads: 8, KVHeads: 2, HeadDim: 64,
		LocalLayer: []bool{true, true, true, false},
	}
	kvDim := int64(hp.KVHeads * hp.HeadDim)
	want := int64(1)*2*4*kvDim + 2*4*int64(hp.Heads)
	if got := inkling2ResidentPerPosBytes(hp); got != want {
		t.Fatalf("per-position bytes = %d, want %d (one global slot)", got, want)
	}
}

func TestInkling2ResidentPerPosBytesCountsRevisitedCore(t *testing.T) {
	// Two passes revisit a two-layer core: slots 4..5 map to global layers 2,3 again.
	hp := model.Inkling2LoopHParams{
		Layers: 4, Passes: 2, CoreStart: 2, CoreEnd: 4,
		Heads: 4, KVHeads: 1, HeadDim: 32,
		LocalLayer: []bool{true, true, false, false},
	}
	kvDim := int64(hp.KVHeads * hp.HeadDim)
	globalSlots := int64(4) // layers 2,3 on the first pass, then 2,3 again on the revisit
	want := globalSlots*2*4*kvDim + 2*4*int64(hp.Heads)
	if got := inkling2ResidentPerPosBytes(hp); got != want {
		t.Fatalf("per-position bytes = %d, want %d", got, want)
	}
}

// The derived context must leave room for the widest prefill arena too: the activation
// the 20480 OOM ran out of is allocated at the widest chunk, so the fit reserves it.
func TestInkling2FitLeavesWidestPrefillArena(t *testing.T) {
	hp := model.Inkling2LoopHParams{
		Layers: 4, Passes: 1, CoreStart: 0, CoreEnd: 0,
		Hidden: 512, Heads: 8, KVHeads: 2, HeadDim: 64,
		SlidingWindow: 512, ConvKernel: 4,
		LocalLayer: []bool{true, true, true, false},
	}
	s := &inkling2Resident{hp: hp, chunkWanted: 2048, layers: make([]ink2Layer, hp.Layers)}
	widest := s.prefillWidestBytes()
	if widest <= 0 {
		t.Fatalf("widest prefill arena = %d bytes, want > 0", widest)
	}
	if base := (&inkling2Resident{hp: hp, chunkWanted: inkling2PrefillChunkBase, layers: s.layers}).prefillWidestBytes(); base >= widest {
		t.Fatalf("widest arena %d not above base-chunk arena %d", widest, base)
	}
	perPos := inkling2ResidentPerPosBytes(hp)
	for _, free := range []int64{4 << 30, 8 << 30, 16 << 30} {
		reserve := inkling2ResidentReserve(free) + widest
		ctx := inkling2FitResidentCtx(free, reserve, perPos)
		if ctx == 0 {
			continue
		}
		if need := int64(ctx)*perPos + reserve; need > free {
			t.Fatalf("free %d: ctx %d plus widest arena needs %d bytes", free, ctx, need)
		}
	}
}
