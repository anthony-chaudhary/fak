package model

import "testing"

// TestQ2KEmbeddingQwen35MetalSessionGuard pins issue #12955: the UD-Q2_K_XL
// artifact's token_embd.weight is genuinely Q2_K, and the packed-embedding
// admission guard must admit it on the native Metal row-gather route (not just
// the Q4_K format) instead of panicking every turn.
func TestQ2KEmbeddingQwen35MetalSessionGuard(t *testing.T) {
	const vocab, hidden = 2, qkK
	embed, err := NewQ2KEmbedding(makeTestQ2KPayload(vocab, hidden), vocab, hidden)
	if err != nil {
		t.Fatalf("NewQ2KEmbedding: %v", err)
	}
	if embed.Format() != "Q2_K" {
		t.Fatalf("Format() = %q, want Q2_K", embed.Format())
	}
	m := &Model{
		Cfg: Config{
			ModelType:  "qwen3_5_text",
			VocabSize:  vocab,
			HiddenSize: hidden,
			NumLayers:  1,
			LayerTypes: []string{"linear_attention"},
		},
		Q2KEmbedding: embed,
	}
	s := &Session{M: m, Cache: NewKVCache(m.Cfg), Q4K: true, Metal: true, MetalQ4K: true}

	// The native planner's exact backend-nil resident-Q4_K Metal tuple must pass
	// the public-entry guard before Qwen hybrid route selection.
	s.validateDenseGPULayers()

	// Pin the row-gather continuation: a whole-table fallback would panic in embedRows.
	rows := make([]float32, 2*hidden)
	m.embedRowsInto(rows, []int{1, 0}, hidden, m.Cfg)
	if len(rows) != 2*hidden {
		t.Fatalf("embedRowsInto wrote %d rows, want %d", len(rows), 2*hidden)
	}

	rejected := []struct {
		name   string
		mutate func(*Session)
	}{
		{name: "wrong architecture", mutate: func(s *Session) { s.M.Cfg.LayerTypes = []string{"full_attention"} }},
		{name: "MoE", mutate: func(s *Session) { s.M.Cfg.NumExperts = 8 }},
		{name: "missing MetalQ4K", mutate: func(s *Session) { s.MetalQ4K = false }},
		{name: "conflicting F16", mutate: func(s *Session) { s.F16 = true }},
	}
	for _, tc := range rejected {
		t.Run(tc.name, func(t *testing.T) {
			candidateModel := &Model{Cfg: m.Cfg, Q2KEmbedding: embed}
			candidate := &Session{
				M: candidateModel, Cache: NewKVCache(candidateModel.Cfg),
				Q4K: true, Metal: true, MetalQ4K: true,
			}
			tc.mutate(candidate)
			defer func() {
				if recover() == nil {
					t.Fatal("validateDenseGPULayers did not reject unsupported Q2_K embedding session")
				}
			}()
			candidate.validateDenseGPULayers()
		})
	}
}
