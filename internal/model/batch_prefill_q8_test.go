package model

import "testing"

// batch_prefill_q8_test.go — regression witness for #12981: the quantized rectangular
// batch-prefill lane (prefillEachRectQ) must be q8-KV-safe. Before the fix it appended
// raw f32 K/V rows into c.K/c.V, which are nil on a KVPrecisionQ8_0 cache, and its
// attention read those f32 slices directly — a quantized session would silently lose its
// KV state. The lane now writes through appendBatchedKV (packing q8 rows) and reads
// through attentionRows (dequantizing on q8), so a quantized batch prefill packs rows
// instead of appending to a nil slice.

// TestBatchRectPrefillQ8PacksKVRows is the focused regression: run the real
// BatchSession.PrefillEach rectangular fast path over Q8 caches and assert the KV rows
// were actually PACKED (kvLen advances, quantized stays true, f32 slices stay nil), with
// an f32 control arm confirming the unchanged lane still fills the f32 slices.
func TestBatchRectPrefillQ8PacksKVRows(t *testing.T) {
	cfg := q8TestConfig()
	cfg.VocabSize = 64
	cfg.RMSNormEps = 1e-5
	cfg.EOSTokenID = -1
	cfg.TieWordEmbeddings = true

	// Premise: this geometry routes to the rectangular quantized lane, so the test is
	// not vacuous.
	if !batchRectFastPathOK(cfg, true) {
		t.Fatal("premise broken: config no longer routes to prefillEachRectQ")
	}

	const B, P = 2, 4
	V := cfg.VocabSize
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		p := make([]int, P)
		for i := range p {
			p[i] = (b*17 + i*5 + 3) % V
		}
		prompts[b] = p
	}
	if got, ok := rectangularPrefillLen(prompts); !ok || got != P {
		t.Fatalf("premise broken: prompts not rectangular P=%d ok=%v", got, ok)
	}

	m := NewSynthetic(cfg)
	m.Quantize()

	// --- Q8 arm -------------------------------------------------------------
	bs := m.NewBatchSession(B)
	bs.SetQuant(true)
	for b := 0; b < B; b++ {
		// Re-realize each lane's cache at q8_0 (the state a quantized session runs over).
		bs.Seqs[b].Cache = NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
	}

	logits := bs.PrefillEach(prompts)
	if len(logits) != B {
		t.Fatalf("PrefillEach returned %d rows, want %d", len(logits), B)
	}
	for b := 0; b < B; b++ {
		if len(logits[b]) != V {
			t.Fatalf("PrefillEach row %d width = %d, want vocab %d", b, len(logits[b]), V)
		}
	}

	for b := 0; b < B; b++ {
		c := bs.Seqs[b].Cache
		if !c.quantized() {
			t.Fatalf("lane %d: cache must stay quantized after prefill", b)
		}
		for l := 0; l < cfg.NumLayers; l++ {
			if got := c.kvLen(l); got != P {
				t.Fatalf("lane %d layer %d: q8 kvLen = %d, want %d (f32 rows appended to a nil slice?)", b, l, got, P)
			}
		}
		if c.K[layerProbe(cfg)] != nil || c.V[layerProbe(cfg)] != nil {
			t.Fatalf("lane %d: q8 path must leave f32 K/V nil", b)
		}
		if c.KVCacheResidentBytes() <= 0 {
			t.Fatalf("lane %d: q8 resident bytes = %d, want > 0", b, c.KVCacheResidentBytes())
		}
	}

	// --- f32 control arm ----------------------------------------------------
	f32 := m.NewBatchSession(B)
	f32.SetQuant(true) // route through the quantized lane; only the CACHE precision differs
	f32Logits := f32.PrefillEach(prompts)
	if len(f32Logits) != B {
		t.Fatalf("f32 control returned %d rows, want %d", len(f32Logits), B)
	}
	for b := 0; b < B; b++ {
		if len(f32Logits[b]) != V {
			t.Fatalf("f32 control row %d width = %d, want vocab %d", b, len(f32Logits[b]), V)
		}
		c := f32.Seqs[b].Cache
		if c.quantized() {
			t.Fatalf("lane %d: f32 control cache must not be quantized", b)
		}
		if len(c.K[0]) == 0 || len(c.V[0]) == 0 {
			t.Fatalf("lane %d: f32 control K/V must be non-empty", b)
		}
		for l := 0; l < cfg.NumLayers; l++ {
			if got := c.kvLen(l); got != P {
				t.Fatalf("lane %d layer %d: f32 kvLen = %d, want %d", b, l, got, P)
			}
		}
	}
}

// layerProbe returns a valid layer index to inspect for nil f32 slices.
func layerProbe(cfg Config) int {
	if cfg.NumLayers > 0 {
		return 0
	}
	return 0
}
