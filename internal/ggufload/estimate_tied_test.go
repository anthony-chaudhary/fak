package ggufload

import "testing"

// This is a physical GGUF storage fixture, not a full-model inference witness.
// A default tied load retains both embedding gather bytes and output-head Q8.
func TestEstimateQ4KDefaultTiedEmbeddingMatchesLoadedWeights(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	const dim, vocab = 256, 4
	path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ4_K, TensorQ4_K, false, true)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.TieWordEmbeddings || !cfg.IsQwen35Hybrid() || cfg.IsMoE() {
		t.Fatalf("fixture must select default tied dense Qwen hybrid: %+v", cfg)
	}
	// No packed-embedding option: its untied-only contract stays unchanged.
	m, err := LoadModelQ4KProfileOptions(path, nil)
	if err != nil {
		t.Fatalf("actual default tied loader: %v", err)
	}
	defer m.CloseWeights()
	const targetQ4 int64 = 3 * dim * dim / 256 * 144
	const gatherF32 int64 = vocab * dim * 4
	const headQ8 int64 = vocab * dim / 32 * 36
	const want = targetQ4 + gatherF32 + headQ8
	r := m.ResidentReport()
	if !m.HasF32("model.embed_tokens.weight") || !m.HasQ8("model.embed_tokens.weight") || m.Q2KEmbedding != nil {
		t.Fatal("default tied loader must retain F32 gather and Q8 head at the embedding key")
	}
	if r.Q4KBytes != targetQ4 || r.F32Bytes != gatherF32 || r.Q8Bytes != headQ8 || r.TotalResidentBytes != want {
		t.Fatalf("actual storage=%+v, independent Q4=%d F32=%d Q8=%d total=%d", r, targetQ4, gatherF32, headQ8, want)
	}
	raw, err := ws.EstimateLoadBytes()
	if err != nil {
		t.Fatal(err)
	}
	if raw != targetQ4+vocab*dim/256*144 {
		t.Fatalf("fixture raw=%d, want111168", raw)
	}
	t.Logf("actual default tied load succeeded: raw=%d stored=%d Q4=%d F32gather=%d Q8head=%d", raw, r.TotalResidentBytes, r.Q4KBytes, r.F32Bytes, r.Q8Bytes)
	plan, err := ws.EstimateQ4KLoadMemoryPlan()
	if err != nil {
		t.Fatalf("estimator rejected supported default tied load: %v", err)
	}
	if plan.Total() != want {
		t.Fatalf("estimate=%d, actual stored=%d", plan.Total(), want)
	}
}
