package compute

import "testing"

// TestPromptPanelHeadroomDemandWidensChunk pins the free-VRAM widening of the prompt-panel
// chunk: spare device budget buys rows up to the cap, the priced demand round-trips into
// exactly that chunk, and no spare budget leaves the transient-only chunk and an empty plan.
func TestPromptPanelHeadroomDemandWidensChunk(t *testing.T) {
	qwen25_3b := TransformerScratchConfig{
		HiddenSize: 2048, IntermediateSize: 11008, VocabSize: 151936, NumLayers: 36,
		NumHeads: 16, NumKVHeads: 2, HeadDim: 128, IncludeLogits: true,
	}
	row := PromptPanelRowBytes(qwen25_3b)
	budget := EstimateHALTransientMemoryPlan(qwen25_3b).Total()
	base := PromptPanelChunkTokens(qwen25_3b, 512)

	for _, avail := range []int64{0, -1} {
		if plan := PromptPanelHeadroomDemand(qwen25_3b, 512, 4096, avail); plan != nil {
			t.Fatalf("avail %d: demand = %v, want nil", avail, plan)
		}
		if got := PromptPanelChunkTokensWithHeadroom(qwen25_3b, 512, avail); got != base {
			t.Fatalf("headroom %d chunk = %d, want transient-only %d", avail, got, base)
		}
	}

	plan := PromptPanelHeadroomDemand(qwen25_3b, 512, 4096, 1<<30)
	if len(plan) != 1 || plan[0].Class != MemoryScratchpad || plan[0].ScopeOrDefault() != MemoryScopeDevice {
		t.Fatalf("demand = %v, want one device scratchpad demand", plan)
	}
	if want := 512*row - budget; plan[0].Bytes != want {
		t.Fatalf("demand bytes = %d, want 512 rows less the transient budget = %d", plan[0].Bytes, want)
	}
	chunk := PromptPanelChunkTokensWithHeadroom(qwen25_3b, 512, plan.Total())
	if chunk != 512 {
		t.Fatalf("headroom chunk = %d, want 512 (>= 256, base %d)", chunk, base)
	}

	// A tight budget buys only the rows it covers, and the chunk never outgrows the priced bytes.
	tight := PromptPanelHeadroomDemand(qwen25_3b, 512, 4096, 100*row)
	if tight == nil {
		t.Fatal("100-row spare budget priced no demand")
	}
	got := PromptPanelChunkTokensWithHeadroom(qwen25_3b, 512, tight.Total())
	if int64(got)*row > budget+tight.Total() || got <= base || got > base+100 {
		t.Fatalf("tight chunk = %d (base %d, demand %d), want inside the priced budget", got, base, tight.Total())
	}

	// A prompt shorter than the cap prices only its own rows.
	if short := PromptPanelHeadroomDemand(qwen25_3b, 512, 200, 1<<30); short.Total() != 200*row-budget {
		t.Fatalf("200-token demand = %d, want %d", short.Total(), 200*row-budget)
	}
	if short := PromptPanelHeadroomDemand(qwen25_3b, 512, base, 1<<30); short != nil {
		t.Fatalf("prompt within the base chunk priced %v, want nil", short)
	}
}
