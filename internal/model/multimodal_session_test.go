package model

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1s
func TestMultimodalSessionBuildsExpandedKVAndDecodes(t *testing.T) {
	m := multimodalTestModel()
	image := validVisionEmbedding(m, 2)
	req := MultimodalRequest{
		Policy: MultimodalPolicy{Mode: MultimodalModeQuarantine},
		Parts: []MultimodalPart{
			{TokenIDs: []int{1}},
			{Image: image},
			{TokenIDs: []int{2}},
		},
	}
	s := m.NewSession()
	logits, verdict, err := s.PrefillMultimodal(req, 3)
	if err != nil {
		t.Fatalf("PrefillMultimodal: %v", err)
	}
	if verdict.Decision != MultimodalAllow || verdict.EmbeddingTokens != 2 {
		t.Fatalf("verdict = %+v, want admitted two-row image", verdict)
	}
	if got := s.Cache.Len(); got != 4 {
		t.Fatalf("cache length = %d, want expanded sequence length 4", got)
	}
	if len(logits) != m.Cfg.VocabSize {
		t.Fatalf("logits = %d, want vocab %d", len(logits), m.Cfg.VocabSize)
	}
	if next := s.Step(4); len(next) != m.Cfg.VocabSize || s.Cache.Len() != 5 {
		t.Fatalf("Step logits/cache = %d/%d, want %d/5", len(next), s.Cache.Len(), m.Cfg.VocabSize)
	}
}

// fak-test:runtime fast est=1s
func TestMultimodalSessionRefusesUnsupportedBeforeCacheWork(t *testing.T) {
	m := multimodalTestModel()
	s := m.NewSession()
	s.Quant = true
	_, verdict, err := s.PrefillMultimodal(MultimodalRequest{
		Policy: MultimodalPolicy{Mode: MultimodalModeQuarantine},
		Parts:  []MultimodalPart{{Image: validVisionEmbedding(m, 1)}},
	}, 3)
	if !errors.Is(err, ErrMultimodalDenied) {
		t.Fatalf("error = %v, want ErrMultimodalDenied", err)
	}
	if verdict.Decision != MultimodalDeny || s.Cache.Len() != 0 {
		t.Fatalf("verdict/cache = %+v/%d, want deny before cache work", verdict, s.Cache.Len())
	}
}
