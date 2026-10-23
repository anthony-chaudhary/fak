package compute

import "testing"

// TestPromptPanelChunkTokensFitsTransientPlan pins the prompt-panel chunk to the transient
// demand the context fit already prices: chunk x row bytes never exceeds it, the cap holds,
// and geometry the estimators cannot size falls back to one-row chunks.
func TestPromptPanelChunkTokensFitsTransientPlan(t *testing.T) {
	qwen25_3b := TransformerScratchConfig{
		HiddenSize: 2048, IntermediateSize: 11008, VocabSize: 151936, NumLayers: 36,
		NumHeads: 16, NumKVHeads: 2, HeadDim: 128, IncludeLogits: true,
	}
	row := PromptPanelRowBytes(qwen25_3b)
	// 5*2048 + 3*2048 + 3*256 + 3*11008 f32 values plus 11008 int8 codes and 344 f32 scales.
	if want := int64((5*2048+3*2048+3*256+3*11008)*4 + 11008 + 344*4); row != want {
		t.Fatalf("PromptPanelRowBytes = %d, want %d", row, want)
	}
	budget := EstimateHALTransientMemoryPlan(qwen25_3b).Total()
	chunk := PromptPanelChunkTokens(qwen25_3b, 512)
	if chunk < 2 || int64(chunk)*row > budget || int64(chunk+1)*row <= budget {
		t.Fatalf("chunk = %d with row %d and budget %d, want the largest chunk inside the budget", chunk, row, budget)
	}
	if got := PromptPanelChunkTokens(qwen25_3b, 8); got != 8 {
		t.Fatalf("capped chunk = %d, want 8", got)
	}
	for _, cfg := range []TransformerScratchConfig{{}, {HiddenSize: 32, NumLayers: 2}} {
		if got := PromptPanelChunkTokens(cfg, 512); got != 1 {
			t.Fatalf("incomplete geometry %+v chunk = %d, want 1", cfg, got)
		}
	}
	if got := PromptPanelChunkTokens(qwen25_3b, 0); got != 1 {
		t.Fatalf("non-positive cap chunk = %d, want 1", got)
	}
}
