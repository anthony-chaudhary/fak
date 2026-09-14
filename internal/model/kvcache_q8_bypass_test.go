package model

import (
	"math"
	"testing"
)

// TestPrefillBatchedQ8PacksNotF32 is the #12981 regression gate for the silent
// q8->f32 fallback in the single-session Prefill batch lane. On PreNorm geometry the
// batched lane (kv.go prefillBatched) appends through appendLayerKV
// (prefill_attn.go) and then attends via preparePrefillAttention's returned keys/values.
//
// Before the fix appendLayerKV wrote raw f32 into cache.K[l]/cache.V[l]. A q8-realized
// cache keeps K/V nil and stores packed rows in kQ8/vQ8, so the f32 append populated
// nothing kvLen could see: kvLen(l) stayed 0 after a full Prefill — the cache looked
// empty to every later attend (a silent q8->f32 fallback). The test asserts, per layer,
// that the packed representation genuinely holds the panel and the f32 slices stay empty.
func TestPrefillBatchedQ8PacksNotF32(t *testing.T) {
	cfg := llamaArchConfig()
	m := NewSynthetic(cfg)
	ids := []int{1, 2, 3, 4, 5}

	// q8 arm: the cache must realize the packed layout, not silently fall back to f32.
	s := m.NewSessionWithKVPrecision(KVPrecisionQ8_0)
	if !s.Cache.quantized() {
		t.Fatalf("q8 session cache is not quantized; test would be vacuous")
	}
	logits := s.Prefill(ids)

	for l := 0; l < cfg.NumLayers; l++ {
		if n := s.Cache.kvLen(l); n != len(ids) {
			t.Errorf("layer %d: q8 kvLen=%d, want %d (prefill did not realize the q8 layout)", l, n, len(ids))
		}
		if len(s.Cache.K[l]) != 0 || len(s.Cache.V[l]) != 0 {
			t.Errorf("layer %d: q8 cache populated f32 K=%d V=%d, want 0/0", l, len(s.Cache.K[l]), len(s.Cache.V[l]))
		}
	}
	if r := s.Cache.KVCacheResidentBytes(); r <= 0 {
		t.Errorf("q8 cache resident bytes=%d, want > 0", r)
	}

	// End-to-end: prefill must return finite logits of the model's vocab width.
	if len(logits) != cfg.VocabSize {
		t.Fatalf("q8 logits len=%d, want %d", len(logits), cfg.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("q8 logits[%d]=%v is not finite", i, v)
		}
	}

	// f32 control arm: same prompt, historical layout — kvLen and raw K width must agree.
	fs := m.NewSessionWithKVPrecision(KVPrecisionFP32)
	_ = fs.Prefill(ids)
	if n := fs.Cache.kvLen(0); n != len(ids) {
		t.Fatalf("f32 kvLen(0)=%d, want %d", n, len(ids))
	}
	if got, want := len(fs.Cache.K[0]), len(ids)*fs.Cache.kvStride(); got != want {
		t.Fatalf("f32 len(K[0])=%d, want %d", got, want)
	}
}
