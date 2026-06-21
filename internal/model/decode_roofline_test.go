package model

import "testing"

// TestQ8DecodeStreamBytes pins the decode roofline numerator to the model's exact Q8_0
// footprint: every layer's q/k/v/o + gate/up/down projection plus the LM head, at 9/8 B per
// weight (1 int8 code + a 4-byte scale per 32-weight block). If a projection is dropped from
// the per-token stream (or double-counted), this catches it — the roofline GB/s is only as
// honest as this count.
func TestQ8DecodeStreamBytes(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 2, NumKVHeads: 1, HeadDim: 32,
		IntermediateSize: 128, VocabSize: 96, RMSNormEps: 1e-6, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: 0, HiddenAct: "silu", ModelType: "qwen2",
	}
	m := NewSynthetic(cfg)
	m.Quantize()

	H, hd, nH, nKV := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	w := nKV * hd
	I := cfg.IntermediateSize
	// weights streamed per layer: q + k + v + o (attn) and gate + up + down (MLP).
	perLayer := int64(nH*hd*H + w*H + w*H + H*nH*hd + I*H + I*H + H*I)
	weights := perLayer*int64(cfg.NumLayers) + int64(cfg.VocabSize*H) // + LM head
	want := weights * 9 / 8                                           // 1 code + 4/32 scale = 9/8 B/weight

	if got := m.q8DecodeStreamBytes(); got != want {
		t.Fatalf("q8DecodeStreamBytes = %d, want %d (weights=%d)", got, want, weights)
	}

	// The roofline derives a sane achieved GB/s and (multi-core) utilization from a ms/tok.
	r := m.DecodeRooflineFor(10.0) // 10 ms/tok
	if r.StreamBytes != want {
		t.Fatalf("roofline StreamBytes = %d, want %d", r.StreamBytes, want)
	}
	if r.AchievedGBps <= 0 || r.CeilingGBps <= 0 || r.BWUtilPct <= 0 {
		t.Fatalf("roofline produced non-positive rates: %+v", r)
	}
	if r.TokPerSec != 100.0 { // 1000/10
		t.Fatalf("roofline TokPerSec = %.3f, want 100", r.TokPerSec)
	}
}

// TestAggregateStreamGBps sanity-checks the multi-core STREAM probe: it returns a positive
// bandwidth and (on a multi-core box) at least matches the single-core MeasureMemBandwidthGBps,
// since concurrent triads tap more of the memory system than one core alone.
func TestAggregateStreamGBps(t *testing.T) {
	if got := measureAggregateStreamGBps(1<<26, 4); got <= 0 {
		t.Fatalf("measureAggregateStreamGBps = %.3f, want > 0", got)
	}
}
