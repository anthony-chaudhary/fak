package model

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type headlessSequencePrefillBackend struct {
	*cappedSequencePrefillBackend
	headlessPath string
}

func (b *headlessSequencePrefillBackend) Qwen35SequenceHeadlessPrefixPath() string {
	return b.headlessPath
}

// fak-test:runtime fast est=20ms lane=default
func TestQwen35HeadlessPrefixAdmission(t *testing.T) {
	newSession := func(t *testing.T, path string, capBytes int64) (*Session, *sequencePrefillBackend) {
		t.Helper()
		m := NewSynthetic(qwen35HybridTestCfg())
		// An untied name makes staging the output distinguishable from the input.
		m.manifest["lm_head.weight"] = m.manifest["model.embed_tokens.weight"]
		base := newSequencePrefillBackend(m)
		capped := &cappedSequencePrefillBackend{sequencePrefillBackend: base, maxBufferBytes: capBytes}
		var be compute.Backend = capped
		if path != "" {
			be = &headlessSequencePrefillBackend{cappedSequencePrefillBackend: capped, headlessPath: path}
		}
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, base
	}
	const headless = "qwen35-hybrid-sequence-headless-prefix-v1"
	t.Run("prefix_omits_unused_oversized_head", func(t *testing.T) {
		s, be := newSession(t, headless, 1024)
		_, used, err := s.tryQwen35SequencePrefill([]int{3, 7}, false)
		if err != nil || !used || be.calls != 1 {
			t.Fatalf("prefix used=%v calls=%d error=%v", used, be.calls, err)
		}
		r := be.requests[0]
		if !r.TokenEmbeddingRows || r.Output.Buf() != nil || r.NeedLogits || r.NeedAllLogits || r.OutputNorm.Buf() == nil || s.halKV.Len() != 2 {
			t.Fatalf("headless request lost embedding, norm, KV or output contract: rows=%v output=%v norm=%v kv=%d", r.TokenEmbeddingRows, r.Output.Buf() != nil, r.OutputNorm.Buf() != nil, s.halKV.Len())
		}
		for key := range s.halW {
			if strings.Contains(key, "lm_head.weight") {
				t.Fatalf("prefix staged unused head under %q", key)
			}
		}
	})
	t.Run("legacy_backend_keeps_cap_refusal", func(t *testing.T) {
		s, be := newSession(t, "", 1024)
		if _, used, err := s.tryQwen35SequencePrefill([]int{3, 7}, false); err != nil || used || be.calls != 0 {
			t.Fatalf("legacy used=%v calls=%d error=%v", used, be.calls, err)
		}
	})
	t.Run("requested_logits_keeps_head", func(t *testing.T) {
		s, be := newSession(t, headless, 0)
		if _, used, err := s.tryQwen35SequencePrefill([]int{3, 7}, true); err != nil || !used || be.calls != 1 {
			t.Fatalf("logits used=%v calls=%d error=%v", used, be.calls, err)
		}
		if be.requests[0].Output.Buf() == nil || !be.requests[0].NeedLogits {
			t.Fatal("logits request omitted its head")
		}
	})
	t.Run("all_logits_builder_keeps_head", func(t *testing.T) {
		s, _ := newSession(t, headless, 0)
		// The verifier sets NeedAllLogits after this existing builder returns.
		r := s.qwen35SequencePrefillRequest([]int{3, 7}, false)
		r.NeedAllLogits = true
		if r.Output.Buf() == nil || r.NeedLogits {
			t.Fatal("verification builder lost its full output contract")
		}
	})
	t.Run("wrong_identity_refuses_before_dispatch", func(t *testing.T) {
		s, be := newSession(t, "wrong", 1024)
		_, used, err := s.tryQwen35SequencePrefill([]int{3, 7}, false)
		var unsupported *UnsupportedSequencePrefillError
		if !used || !errors.As(err, &unsupported) || be.calls != 0 {
			t.Fatalf("wrong identity used=%v calls=%d error=%v", used, be.calls, err)
		}
	})
}
