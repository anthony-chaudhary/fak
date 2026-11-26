package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func cohortPriceHP() model.Inkling2LoopHParams {
	return model.Inkling2LoopHParams{
		Layers: 4, Passes: 2, CoreStart: 1, CoreEnd: 3,
		Hidden: 256, Heads: 4, KVHeads: 2, HeadDim: 64, RelLength: 8,
		SlidingWindow: 128, ConvKernel: 4, Experts: 8, ExpertsUsed: 2, ExpertWidth: 128,
		LocalLayer: []bool{true, false, true, false},
	}
}

func TestInkling2MetalCohortBytesBoundsRowState(t *testing.T) {
	hp := cohortPriceHP()
	ctx := min(2048, Inkling2LoopResidentContext)
	one, err := inkling2MetalCohortBytes(hp, 1000, 1, ctx)
	if err != nil {
		t.Fatal(err)
	}
	four, err := inkling2MetalCohortBytes(hp, 1000, 4, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if four <= one {
		t.Fatalf("4 rows %d not above 1 row %d", four, one)
	}
	// every row holds its own caches over every slot at the cohort's caps
	caps := inkling2BatchCaps(hp, inkling2ResidentCaps(hp, 0, true, inkling2PrefillChunkBase), ctx, inkling2PrefillChunkWantedOn(inkling2MetalNamed{}, func(string) (string, bool) { return "", false }))
	if state := inkling2BatchStateBytes(hp, caps, 4, false); four < state+inkling2BatchHeadroom {
		t.Fatalf("4-row price %d below its KV state %d plus headroom", four, state)
	}
	short, err := inkling2MetalCohortBytes(hp, 1000, 4, 512)
	if err != nil || short >= four {
		t.Fatalf("shorter context price %d (err %v) not below %d", short, err, four)
	}
	// the engine never runs a cohort past the resident context, so neither does the price
	at, _ := inkling2MetalCohortBytes(hp, 1000, 2, Inkling2LoopResidentContext)
	past, err := inkling2MetalCohortBytes(hp, 1000, 2, Inkling2LoopResidentContext+4096)
	if err != nil || past != at {
		t.Fatalf("price past the resident context %d (err %v), at it %d", past, err, at)
	}
	if _, err := inkling2MetalCohortBytes(hp, 1000, 0, ctx); err == nil {
		t.Fatal("zero rows priced")
	}
}
