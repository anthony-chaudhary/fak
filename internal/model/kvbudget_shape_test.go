package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvbudget"
)

// TestKVCacheShapeGLM52ReproducesHardcode is the issue's witness: a Config
// carrying the GLM-5.2 header numbers the triage doc §3.2 pins (the same values
// kvbudget.GLM52DSA hardcodes) produces the identical Shape and per-token KV
// footprint — the GLM-5.2 hardcode becomes a special case of the general header
// read, not a separate constant (#5242).
func TestKVCacheShapeGLM52ReproducesHardcode(t *testing.T) {
	cfg := Config{
		NumLayers:     92,
		KVLoraRank:    512,
		QKRopeHeadDim: 64,
		IndexNHeads:   64, // any > 0 declares the DSA lightning indexer
		IndexHeadDim:  128,
	}
	got := cfg.KVCacheShape()
	if want := kvbudget.GLM52DSA; got != want {
		t.Fatalf("KVCacheShape() = %+v, want GLM52DSA %+v", got, want)
	}
	// The reproduced Shape sizes to the doc §3.2 per-token footprint.
	if e := got.KVElemsPerToken(); e != 64768 {
		t.Errorf("KVElemsPerToken = %d, want 64768", e)
	}
	if b := got.KVBytesPerToken(kvbudget.F16); b != 129536.0 {
		t.Errorf("KVBytesPerToken(F16) = %v, want 129536", b)
	}
}

// TestKVCacheShapeMHA sizes a standard multi-head / grouped-query header via the
// ktransformers MHA branch: NumKVHeads × (HeadDim + VHeadDim) × NumLayers
// (kv_cache_calculator.py:121@0c2912a). A header with no separate value-head
// width squares VHeadDim to HeadDim.
func TestKVCacheShapeMHA(t *testing.T) {
	// A Llama-3-8B-shaped header: 32 layers, 8 KV heads (GQA), 128-wide heads.
	cfg := Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128}
	got := cfg.KVCacheShape()
	if got.Kind != kvbudget.MHA {
		t.Fatalf("Kind = %v, want MHA", got.Kind)
	}
	if got.VHeadDim != 128 {
		t.Errorf("VHeadDim = %d, want 128 (squared to HeadDim)", got.VHeadDim)
	}
	// 32 × 8 × (128+128) = 65536 elems/token; F16 ⇒ 131072 bytes/token.
	if e := got.KVElemsPerToken(); e != 65536 {
		t.Errorf("KVElemsPerToken = %d, want 65536", e)
	}
	if b := got.KVBytesPerToken(kvbudget.F16); b != 131072.0 {
		t.Errorf("KVBytesPerToken(F16) = %v, want 131072", b)
	}
	// A distinct value-head width (v_head_dim != head_dim) is carried, not squared.
	rect := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 128, VHeadDim: 64}.KVCacheShape()
	if e := rect.KVElemsPerToken(); e != 4*2*(128+64) {
		t.Errorf("rectangular-head KVElemsPerToken = %d, want %d", e, 4*2*(128+64))
	}
}

// TestKVCacheShapeMLANoIndexer proves a DeepSeek-style MLA header with no
// lightning indexer (IndexNHeads == 0) carries no index term — only the
// compressed latent + rope key — so MLA and MLA+NSA are distinct branches.
func TestKVCacheShapeMLANoIndexer(t *testing.T) {
	cfg := Config{NumLayers: 60, KVLoraRank: 512, QKRopeHeadDim: 64}
	got := cfg.KVCacheShape()
	if got.Kind != kvbudget.MLA {
		t.Fatalf("Kind = %v, want MLA", got.Kind)
	}
	if idx := got.IndexElemsPerToken(); idx != 0 {
		t.Errorf("IndexElemsPerToken = %d, want 0 (no indexer declared)", idx)
	}
	if e, want := got.KVElemsPerToken(), 60*(512+64); e != want {
		t.Errorf("KVElemsPerToken = %d, want %d", e, want)
	}
}
