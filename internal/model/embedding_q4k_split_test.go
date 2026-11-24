package model

import (
	"bytes"
	"fmt"
	"testing"
)

// buildQ4KEmbeddingSplitModel returns a synthetic Qwen3.5-hybrid model whose token table
// equals the dequantized rows of a packed Q4_K table. packed=true swaps the f32 table for
// the packed one (the pinned Qwen3.8 Q4_K_M serve shape) and keeps an untied head on the
// same bytes, so both variants compute the same function.
func buildQ4KEmbeddingSplitModel(t *testing.T, packed bool) *Model {
	t.Helper()
	cfg := qwen35GraphTestCfg()
	cfg.HiddenSize = qkK // one Q4_K superblock per row
	m := NewSynthetic(cfg)
	embed, err := NewQ4KEmbedding(makeTestQ4KRaw(cfg.VocabSize, cfg.HiddenSize), cfg.VocabSize, cfg.HiddenSize)
	if err != nil {
		t.Fatalf("NewQ4KEmbedding: %v", err)
	}
	table, err := embed.DequantizeTable()
	if err != nil {
		t.Fatalf("DequantizeTable: %v", err)
	}
	copy(m.tensor("model.embed_tokens.weight"), table)
	m.manifest["lm_head.weight"] = m.manifest["model.embed_tokens.weight"]
	m.Cfg.TieWordEmbeddings = false
	if packed {
		m.Q2KEmbedding = embed
		delete(m.manifest, "model.embed_tokens.weight")
	}
	return m
}

// A device-backed Q4K session with a packed Q4_K token table and a dense GPU layer split must
// decode exactly like the same model with the f32 table: the row is gathered on the host and
// uploaded as layer 0's input, and the split reads the residual back before the host tail.
//
// fak-test:runtime fast est=2s
func TestQ4KEmbeddingDeviceLayerSplitMatchesF32Embedding(t *testing.T) {
	ref0, packed0 := buildQ4KEmbeddingSplitModel(t, false), buildQ4KEmbeddingSplitModel(t, true)
	if !bytes.Equal(ref0.raw, packed0.raw) {
		t.Fatal("synthetic models diverge; the parity comparison would be meaningless")
	}
	for _, n := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("gpu-layers=%d", n), func(t *testing.T) {
			ref, packed := buildQ4KEmbeddingSplitModel(t, false), buildQ4KEmbeddingSplitModel(t, true)
			sRef := newQwenGraphSession(t, ref, newTestQwenGraphBackend(ref))
			sPk := newQwenGraphSession(t, packed, newTestQwenGraphBackend(packed))
			for _, s := range []*Session{sRef, sPk} {
				s.Q4K = true
				s.DenseGPULayers = n
			}
			prompt := []int{3, 7, 11}
			assertSameF32(t, "prefill", sPk.Prefill(prompt), sRef.Prefill(prompt))
			for _, id := range []int{5, 12, 19} {
				assertSameF32(t, fmt.Sprintf("step %d", id), sPk.Step(id), sRef.Step(id))
			}
		})
	}
}

// The device-split admission is Q4_K-only and device-only: every neighbouring shape keeps the
// historical refusal.
//
// fak-test:runtime fast est=1s
func TestQ4KEmbeddingDeviceLayerSplitGuardBoundaries(t *testing.T) {
	m := buildQ4KEmbeddingSplitModel(t, true)
	admitted := newQwenGraphSession(t, m, newTestQwenGraphBackend(m))
	admitted.Q4K = true
	admitted.DenseGPULayers = 2
	if n, split := admitted.validateDenseGPULayers(); n != 2 || !split {
		t.Fatalf("validateDenseGPULayers = (%d,%v), want (2,true)", n, split)
	}
	rejected := []struct {
		name   string
		mutate func(*Session)
	}{
		{name: "not a Q4K session", mutate: func(s *Session) { s.Q4K = false }},
		{name: "F16", mutate: func(s *Session) { s.F16 = true }},
		{name: "Metal flag", mutate: func(s *Session) { s.MetalQ4K = true }},
		{name: "MoE", mutate: func(s *Session) { s.M.Cfg.NumExperts = 8 }},
		{name: "Q2_K table", mutate: func(s *Session) {
			q2k, err := NewQ2KEmbedding(makeTestQ2KPayload(s.M.Cfg.VocabSize, s.M.Cfg.HiddenSize), s.M.Cfg.VocabSize, s.M.Cfg.HiddenSize)
			if err != nil {
				t.Fatal(err)
			}
			s.M.Q2KEmbedding = q2k
		}},
	}
	for _, tc := range rejected {
		t.Run(tc.name, func(t *testing.T) {
			cm := buildQ4KEmbeddingSplitModel(t, true)
			s := newQwenGraphSession(t, cm, newTestQwenGraphBackend(cm))
			s.Q4K = true
			s.DenseGPULayers = 2
			tc.mutate(s)
			defer func() {
				if recover() == nil {
					t.Fatal("validateDenseGPULayers admitted an unsupported packed-embedding split")
				}
			}()
			s.validateDenseGPULayers()
		})
	}
}
