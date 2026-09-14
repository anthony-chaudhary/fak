package model

import (
	"math"
	"testing"
)

// TestQ8KVCacheLogitParityVsF32Synthetic is the issue #12981 end-to-end parity gate:
// a Q8_0 KV-cache session and an F32 KV-cache session, both built from the SAME
// deterministic NewSynthetic weights, must agree on their next-token logits.
//
// Honesty note: the weights are synthetic, so the numeric logits are NOT
// HF-meaningful. This witnesses q8-KV INVARIANCE (argmax agreement plus
// distribution cosine on a fixed prompt set), not HF numerical parity — the latter
// remains the weight-backed oracle test's job. The value here is that the whole
// prefill->attention->decode->head path is exercised with the q8 row layout
// genuinely realized, not silently fallen back to f32.
//
// Two deliberate deviations from the llamaArchConfig() baseline, each load-bearing:
//  1. DenseMLP=true routes the prefill through the per-token lane
//     (prefillTokenLoop -> tokenHidden -> Cache.appendKV, kv.go:~1023), which is the
//     lane whose attention reads K/V through decodeRowInto on a quantized cache
//     (kv.go:~1042-1072). The default batched f32 lane (prefillBatched,
//     batch_prefill.go:198) appends straight to Cache.K/V and never touches the
//     packed q8 row-sets, so a q8 session there would silently behave as f32 — the
//     exact vacuity this test guards against.
//  2. HeadDim is widened to 16 (HiddenSize 64) so each KV row (NumKVHeads*HeadDim =
//  32. is a whole Q8_0 block. With the baseline HeadDim 8 the row is 16 wide and
//     QuantizeKVQ8_0 (kvquant.go:390) scales a whole row with one block, which on
//     synthetic weights drops the end-to-end logit cosine to ~0.95-0.99 — real q8
//     drift, but too noisy to hold the 0.999 invariance bound. Block-aligned rows
//     exercise the codec as designed.
func TestQ8KVCacheLogitParityVsF32Synthetic(t *testing.T) {
	cfg := llamaArchConfig()
	cfg.DenseMLP = true
	cfg.HeadDim = 16
	cfg.HiddenSize = 64
	m := NewSynthetic(cfg)

	prompts := [][]int{
		{1, 2, 3, 4, 5},
		{3, 17, 5, 23, 41, 2, 19},
		{2, 5, 11, 23},
		{3, 7, 13, 29},
	}
	for _, p := range prompts {
		for _, id := range p {
			if id < 0 || id >= cfg.VocabSize {
				t.Fatalf("prompt id %d out of vocab range %d", id, cfg.VocabSize)
			}
		}
	}

	for i, ids := range prompts {
		// Fresh sessions per prompt so each arm starts at position 0.
		q8 := m.NewSessionWithKVPrecision(KVPrecisionQ8_0)
		f32 := m.NewSessionWithKVPrecision(KVPrecisionFP32)

		if !q8.Cache.quantized() {
			t.Fatalf("prompt %d: q8 arm cache is not quantized (prec=%v); test would be vacuous", i, q8.KVPrecision)
		}
		if f32.Cache.quantized() {
			t.Fatalf("prompt %d: f32 control cache reports quantized", i)
		}

		gotQ8 := q8.Prefill(ids)
		gotF32 := f32.Prefill(ids)

		// The q8 arm must have genuinely appended one row per prompt position; a
		// silent f32 fallback would leave the packed row-set empty.
		if n := q8.Cache.kvLen(0); n != len(ids) {
			t.Fatalf("prompt %d: q8 cache kvLen(0)=%d, want %d (prefill did not realize the q8 layout)", i, n, len(ids))
		}
		if n := f32.Cache.kvLen(0); n != len(ids) {
			t.Fatalf("prompt %d: f32 cache kvLen(0)=%d, want %d", i, n, len(ids))
		}

		if len(gotQ8) != cfg.VocabSize || len(gotF32) != cfg.VocabSize {
			t.Fatalf("prompt %d: logits len q8=%d f32=%d, want %d", i, len(gotQ8), len(gotF32), cfg.VocabSize)
		}

		var maxDelta float64
		for j := range gotQ8 {
			d := math.Abs(float64(gotQ8[j]) - float64(gotF32[j]))
			if d > maxDelta {
				maxDelta = d
			}
		}
		cs := cosine(gotQ8, gotF32)
		amQ8, amF32 := argmax(gotQ8), argmax(gotF32)

		t.Logf("prompt %d len=%d: argmax q8=%d f32=%d cosine=%.8f max|dlogit|=%.6g",
			i, len(ids), amQ8, amF32, cs, maxDelta)

		if amQ8 != amF32 {
			t.Errorf("prompt %d: argmax mismatch q8=%d f32=%d", i, amQ8, amF32)
		}
		if cs < 0.999 {
			t.Errorf("prompt %d: cosine(q8,f32)=%.8f below 0.999", i, cs)
		}
		// Two-sided: bit-identical logits would mean the q8 arm never quantized
		// (a silent fallback), so require the q8 drift to be actually present.
		if maxDelta == 0 {
			t.Errorf("prompt %d: q8 and f32 logits are bit-identical; q8 realization did not engage", i)
		}
	}
}
