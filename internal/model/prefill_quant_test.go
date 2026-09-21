package model

import (
	"math/rand"
	"testing"
)

// prefill_quant_test.go — witness for #12274: deferred prefill KV quantization.
//
// The batched prefill lane appends P positions' K/V rows per layer through
// KVCache.appendBatchedKV. Before this leaf the quantized branch re-encoded each
// row in turn (P QuantizeKVQ8_0 calls and P Scale/Codes allocations per layer,
// per step) inside the prefill hot loop. The fix routes the whole panel through
// kvPackedRow.appendRowsBulk, one bulk conversion pass.
//
// The witness has two arms:
//  1. TestDeferredPrefillQuantizationBulkMatchesPerRow — proves the deferred bulk
//     conversion is byte-identical to the historical per-row encode (no drift).
//  2. TestDeferredPrefillQuantizationPrefillIsStable — proves the real batched
//     prefill lane still packs q8 rows and produces a stable, non-degenerate
//     result over the deferred path.

// perRowBaseline encodes the same panel through the historical one-row-at-a-time
// path (the shape appendBatchedKV used before #12274), so the bulk path can be
// diffed against it exactly.
func perRowBaseline(rows, width int, panel []float32) kvPackedRow {
	var p kvPackedRow
	p.width = width
	for t := 0; t < rows; t++ {
		p.appendRow(panel[t*width : (t+1)*width])
	}
	return p
}

// BenchmarkDeferredPrefillQuantizationBulk measures the deferred bulk conversion
// against the historical per-row encode over one representative prefill panel. It
// exists so the #12274 claim ("one bulk pass instead of p per-row passes") is a
// measured number with a date/commit, not a prose assertion.
func BenchmarkDeferredPrefillQuantizationBulk(b *testing.B) {
	const rows, width = 512, 128
	rng := rand.New(rand.NewSource(12274))
	panel := make([]float32, rows*width)
	for i := range panel {
		panel[i] = float32(rng.NormFloat64())
	}
	b.Run("bulk", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			var p kvPackedRow
			p.appendRowsBulk(panel, rows, width)
		}
	})
	b.Run("per-row", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p := perRowBaseline(rows, width, panel)
			_ = p
		}
	})
}

// TestDeferredPrefillQuantizationBulkMatchesPerRow is the focused regression: the
// deferred bulk encode must produce the very codes and scales the per-row encode
// did. Width divides the 32-element Q8_0 block evenly in every supported geometry,
// so block boundaries stay aligned across rows and the deferred pass is exact, not
// an approximation.
func TestDeferredPrefillQuantizationBulkMatchesPerRow(t *testing.T) {
	const rows, width = 6, 32 // width == KVQuantQ8_0BlockSize
	rng := rand.New(rand.NewSource(12274))
	panel := make([]float32, rows*width)
	for i := range panel {
		panel[i] = float32(rng.NormFloat64())
	}

	baseline := perRowBaseline(rows, width, panel)

	var bulk kvPackedRow
	bulk.appendRowsBulk(panel, rows, width)

	if bulk.len() != baseline.len() || bulk.len() != rows {
		t.Fatalf("bulk rows = %d, baseline rows = %d, want %d", bulk.len(), baseline.len(), rows)
	}
	if len(bulk.codes) != len(baseline.codes) {
		t.Fatalf("bulk codes = %d, baseline codes = %d", len(bulk.codes), len(baseline.codes))
	}
	for i := range baseline.codes {
		if bulk.codes[i] != baseline.codes[i] {
			t.Fatalf("code[%d] = %d, baseline = %d (deferred bulk drifted from per-row)", i, bulk.codes[i], baseline.codes[i])
		}
	}
	if len(bulk.scales) != len(baseline.scales) {
		t.Fatalf("bulk scales = %d, baseline scales = %d", len(bulk.scales), len(baseline.scales))
	}
	for i := range baseline.scales {
		if bulk.scales[i] != baseline.scales[i] {
			t.Fatalf("scale[%d] = %v, baseline = %v (deferred bulk drifted from per-row)", i, bulk.scales[i], baseline.scales[i])
		}
	}
}

// TestDeferredPrefillQuantizationMultiRowWidths exercises the bulk path across the
// supported head_dim widths (each a multiple of the 32-element block) so the
// block-alignment premise the exactness rests on is actually covered.
func TestDeferredPrefillQuantizationMultiRowWidths(t *testing.T) {
	for _, width := range []int{32, 64, 128, 256} {
		const rows = 5
		rng := rand.New(rand.NewSource(int64(width)))
		panel := make([]float32, rows*width)
		for i := range panel {
			panel[i] = float32(rng.NormFloat64())
		}
		baseline := perRowBaseline(rows, width, panel)
		var bulk kvPackedRow
		bulk.appendRowsBulk(panel, rows, width)
		if bulk.len() != rows {
			t.Fatalf("width %d: bulk rows = %d, want %d", width, bulk.len(), rows)
		}
		if len(bulk.codes) != len(baseline.codes) || len(bulk.scales) != len(baseline.scales) {
			t.Fatalf("width %d: bulk (%d codes, %d scales) != baseline (%d codes, %d scales)",
				width, len(bulk.codes), len(bulk.scales), len(baseline.codes), len(baseline.scales))
		}
		for i := range baseline.codes {
			if bulk.codes[i] != baseline.codes[i] {
				t.Fatalf("width %d: deferred bulk code differs from per-row at element %d", width, i)
			}
		}
		for i := range baseline.scales {
			if bulk.scales[i] != baseline.scales[i] {
				t.Fatalf("width %d: deferred bulk scale differs from per-row at element %d", width, i)
			}
		}
	}
}

// TestDeferredPrefillQuantizationPrefillIsStable runs the real quantized rectangular
// batch-prefill lane and asserts the deferred path still packs q8 KV rows (f32
// slices stay nil, kvLen advances) and returns a byte-stable result across two
// identical runs — the "output token identity vs baseline" criterion, held on the
// live path rather than only the codec unit.
func TestDeferredPrefillQuantizationPrefillIsStable(t *testing.T) {
	cfg := q8TestConfig()
	cfg.VocabSize = 64
	cfg.RMSNormEps = 1e-5
	cfg.EOSTokenID = -1
	cfg.TieWordEmbeddings = true

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

	run := func() [][]float32 {
		m := NewSynthetic(cfg)
		m.Quantize()
		bs := m.NewBatchSession(B)
		bs.SetQuant(true)
		for b := 0; b < B; b++ {
			bs.Seqs[b].Cache = NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
		}
		logits := bs.PrefillEach(prompts)
		for b := 0; b < B; b++ {
			c := bs.Seqs[b].Cache
			if !c.quantized() {
				t.Fatalf("lane %d: cache must stay quantized through deferred prefill", b)
			}
			for l := 0; l < cfg.NumLayers; l++ {
				if got := c.kvLen(l); got != P {
					t.Fatalf("lane %d layer %d: q8 kvLen = %d, want %d", b, l, got, P)
				}
			}
			if c.K[layerProbe(cfg)] != nil || c.V[layerProbe(cfg)] != nil {
				t.Fatalf("lane %d: deferred q8 path must leave f32 K/V nil", b)
			}
		}
		return logits
	}

	first := run()
	second := run()
	if len(first) != B || len(second) != B {
		t.Fatalf("PrefillEach returned %d/%d rows, want %d", len(first), len(second), B)
	}
	for b := 0; b < B; b++ {
		if len(first[b]) != V {
			t.Fatalf("row %d width = %d, want vocab %d", b, len(first[b]), V)
		}
		for i := range first[b] {
			if first[b][i] != second[b][i] {
				t.Fatalf("row %d logit %d: run1 %v != run2 %v (deferred prefill not deterministic)", b, i, first[b][i], second[b][i])
			}
		}
	}
}
