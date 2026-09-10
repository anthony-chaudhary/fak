package model

import (
	"encoding/binary"
	"testing"
)

func TestQ4KEmbeddingQwen35MetalSessionGuard(t *testing.T) {
	const vocab, hidden = 2, qkK
	raw := make([]byte, vocab*q4kBlockBytes)
	for row := 0; row < vocab; row++ {
		block := raw[row*q4kBlockBytes : (row+1)*q4kBlockBytes]
		binary.LittleEndian.PutUint16(block, 0x4000) // d = 2
		scales := block[4 : 4+12]
		scales[0], scales[1], scales[2], scales[3] = 1, 1, 1, 1
		scales[8], scales[9], scales[10], scales[11] = 1, 1, 1, 1
		for i := 4 + 12; i < len(block); i++ {
			block[i] = 0x33
		}
	}
	embed, err := NewQ4KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatalf("NewQ4KEmbedding: %v", err)
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
	for i, got := range rows {
		if got != 6 {
			t.Fatalf("gathered embedding[%d] = %v, want 6", i, got)
		}
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
					t.Fatal("validateDenseGPULayers did not reject unsupported Q4_K embedding session")
				}
			}()
			candidate.validateDenseGPULayers()
		})
	}
}
