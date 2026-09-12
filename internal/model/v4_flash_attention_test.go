package model

import (
	"context"
	"errors"
	"testing"
)

// v4_flash_attention_test.go - the bounded ratio-0 tracer witness for
// DeepSeek-V4-Flash-0731 per-layer attention (parent #12636).
//
// The published contract gives every layer a 128-token sliding window and ONE
// per-layer compression ratio. Ratio 0 is the pure window layer and is the
// smallest token tracer. Ratios 4 and 128 additionally carry compressor state
// that is not implemented yet. This file witnesses two invariants:
//
//  1. regime classification: ratio 0 -> window-only tracer, 4/128 ->
//     compressed (unimplemented), anything else -> invalid.
//  2. fail-closed dispatch: a V4 session whose layers carry an unimplemented
//     compression ratio must REFUSE to run before any prefill/decode, rather
//     than silently falling through to the generic sliding-window Q/K/V path.

// v4FlashTracerConfig is a small synthetic DeepSeek-V4 config carrying the
// published per-layer compression schedule shape. Dims are miniature so the
// synthetic model is cheap; the identity and the ratio schedule are what the
// attention gate keys on.
func v4FlashTracerConfig(ratios []int) Config {
	cfg := Config{
		ModelType:         "deepseek_v4",
		HiddenSize:        32,
		NumLayers:         len(ratios),
		NumHeads:          4,
		NumKVHeads:        1,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
		CompressRatios:    append([]int(nil), ratios...),
	}
	cfg.Window = make([]int, cfg.NumLayers)
	for i := range cfg.Window {
		cfg.Window[i] = V4FlashWindowSize
	}
	return cfg
}

// TestV4FlashAttentionRegimeClassification pins the closed classification map.
func TestV4FlashAttentionRegimeClassification(t *testing.T) {
	cases := []struct {
		ratio int
		want  V4FlashAttentionRegime
		ok    bool
	}{
		{ratio: 0, want: V4FlashAttentionWindowOnly, ok: true},
		{ratio: 4, want: V4FlashAttentionCompressed, ok: true},
		{ratio: 128, want: V4FlashAttentionCompressed, ok: true},
		{ratio: -1, want: 0, ok: false},
		{ratio: 1, want: 0, ok: false},
		{ratio: 5, want: 0, ok: false},
		{ratio: 256, want: 0, ok: false},
	}
	for _, tc := range cases {
		got, ok := v4FlashAttentionRegime(tc.ratio)
		if ok != tc.ok || (ok && got != tc.want) {
			t.Fatalf("regime(%d) = (%v,%v), want (%v,%v)", tc.ratio, got, ok, tc.want, tc.ok)
		}
	}
}

// TestV4FlashRefusesUnimplementedCompressionBeforeGenerate is the load-bearing
// fail-closed gate. A session over a config whose schedule carries ratio 4/128
// must return the typed unimplemented error from GenerateContext, with zero
// tokens emitted, BEFORE any prefill/decode. Falling through to the generic
// sliding-window path is the exact defect this gate forbids.
func TestV4FlashRefusesUnimplementedCompressionBeforeGenerate(t *testing.T) {
	cfg := v4FlashTracerConfig([]int{0, 4, 128})
	m := NewSynthetic(cfg)
	s := m.NewSession()

	toks, err := s.GenerateContext(context.Background(), []int{1, 2, 3}, 4)
	if !errors.Is(err, ErrV4FlashAttentionUnimplemented) {
		t.Fatalf("GenerateContext error = %v, want ErrV4FlashAttentionUnimplemented", err)
	}
	if len(toks) != 0 {
		t.Fatalf("emitted %d tokens before refusing compressed attention; want 0", len(toks))
	}
	if s.Cache.Len() != 0 {
		t.Fatalf("session cache holds %d rows after refusal; want 0 (no prefill ran)", s.Cache.Len())
	}
}

// TestV4FlashRatioZeroTracerRunsThroughWindowPath proves the ratio-0 tracer is
// admitted: an all-zero schedule generates without the compressed refusal, and
// the session cache advances (the window attention path executed).
func TestV4FlashRatioZeroTracerRunsThroughWindowPath(t *testing.T) {
	cfg := v4FlashTracerConfig([]int{0, 0, 0, 0})
	m := NewSynthetic(cfg)
	s := m.NewSession()

	if err := s.refuseUnimplementedV4FlashAttention(); err != nil {
		t.Fatalf("ratio-0 tracer refused: %v", err)
	}
	toks, err := s.GenerateContext(context.Background(), []int{1, 2, 3}, 4)
	if err != nil {
		t.Fatalf("ratio-0 tracer GenerateContext error: %v", err)
	}
	if len(toks) != 4 {
		t.Fatalf("emitted %d tokens, want 4", len(toks))
	}
	if s.Cache.Len() == 0 {
		t.Fatal("session cache did not advance; window attention path did not run")
	}
}

// TestV4FlashInvalidRatioFailsClosed: a schedule outside {0,4,128} is malformed
// metadata and must refuse with the typed error rather than run.
func TestV4FlashInvalidRatioFailsClosed(t *testing.T) {
	cfg := v4FlashTracerConfig([]int{0, 5})
	s := NewSynthetic(cfg).NewSession()
	if _, err := s.GenerateContext(context.Background(), []int{1}, 1); !errors.Is(err, ErrV4FlashAttentionRatioInvalid) {
		t.Fatalf("invalid ratio error = %v, want ErrV4FlashAttentionRatioInvalid", err)
	}
}

// TestV4FlashGateLeavesNonV4FamiliesAlone: a Llama-shaped model with no
// compression schedule must not be refused by the V4 gate.
func TestV4FlashGateLeavesNonV4FamiliesAlone(t *testing.T) {
	cfg := swaTestCfg()
	cfg.ModelType = "llama"
	cfg.CompressRatios = nil
	s := NewSynthetic(cfg).NewSession()
	if err := s.refuseUnimplementedV4FlashAttention(); err != nil {
		t.Fatalf("non-V4 family refused by V4 gate: %v", err)
	}
}
