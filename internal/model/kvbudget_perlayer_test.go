package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvbudget"
)

// gemma4WindowCadence is the public `google/gemma-4-26B-A4B-it` layer cadence the
// kvbudget per-layer doc pins: 30 layers whose layer_types repeat five
// `sliding_attention` then one `full_attention` (25 local, 5 global) with
// `sliding_window: 1024`. It is written the way applyGemma4Config writes it —
// the window for a local layer, the -1 FULL-ATTENTION SENTINEL for a global one
// (internal/ggufload/gguf_config.go:398,404) — because the sentinel round-trip is
// what these tests are about.
func gemma4WindowCadence(layers, window int) []int {
	w := make([]int, layers)
	for l := range w {
		if (l+1)%6 == 0 { // every sixth layer is global
			w[l] = -1
			continue
		}
		w[l] = window
	}
	return w
}

// gemma4ShapedConfig is that cadence on a uniform head geometry: 2 kv heads of a
// square 512-wide head, 30 layers. Uniform heads isolate the WINDOW axis, so the
// only thing that can move the byte figures below is the per-layer window.
func gemma4ShapedConfig() Config {
	return Config{
		NumLayers:  30,
		NumKVHeads: 2,
		HeadDim:    512,
		Window:     gemma4WindowCadence(30, 1024),
	}
}

// TestKVCacheShapeCarriesPerLayerWindow is the WIRING witness for #5498's consumer
// half: kvbudget.Shape gained an optional PerLayer profile, but until KVCacheShape
// populated it the field was inert — every Shape the planner reads was built here
// with PerLayer nil, so a window-capped GGUF still sized as uniformly global. A
// gemma-4-shaped Config must now produce a NON-NIL profile carrying the real
// window, with the loader's -1 sentinel mapped to kvbudget's "no window" spelling.
func TestKVCacheShapeCarriesPerLayerWindow(t *testing.T) {
	got := gemma4ShapedConfig().KVCacheShape()
	if got.PerLayer == nil {
		t.Fatalf("PerLayer = nil, want a profile: the per-layer window is not wired into KVCacheShape")
	}
	if n := len(got.PerLayer.Window); n != 30 {
		t.Fatalf("len(PerLayer.Window) = %d, want 30 (one entry per layer)", n)
	}
	// Layer 0 is local: the real 1024-token bound is carried verbatim.
	if w := got.PerLayer.Window[0]; w != 1024 {
		t.Errorf("PerLayer.Window[0] = %d, want 1024 (sliding layer keeps its bound)", w)
	}
	// Layer 5 is the first global one. The loader spells full attention -1; the
	// profile must spell it as "no window" (non-positive), NOT carry -1 into
	// LayerTokens' min(window, ctx) where it would be a nonsense one-token bound.
	if w := got.PerLayer.Window[5]; w > 0 {
		t.Errorf("PerLayer.Window[5] = %d, want a non-positive 'no window' entry (the -1 sentinel must not survive as a bound)", w)
	}
	// The sentinel mapping is only meaningful if it reaches LayerTokens: the
	// global layer must hold the WHOLE context, the local one only its window.
	const ctx = 16384
	if got.LayerTokens(5, ctx) != ctx {
		t.Errorf("LayerTokens(5, %d) = %d, want %d (global layer attends over the full context)", ctx, got.LayerTokens(5, ctx), ctx)
	}
	if got.LayerTokens(0, ctx) != 1024 {
		t.Errorf("LayerTokens(0, %d) = %d, want 1024 (local layer is window-bounded)", ctx, got.LayerTokens(0, ctx))
	}
	// Shape must stay ==-comparable: PerLayer is a POINTER precisely so a slice
	// field never lands in Shape. This line fails to COMPILE if that regresses.
	if got == (kvbudget.Shape{}) {
		t.Fatalf("Shape compared equal to its zero value")
	}
}

// TestKVCacheShapePerLayerHeadGeometry witnesses the other two per-layer axes:
// gemma4's local and global layers differ in head WIDTH (head_dim 256 vs
// global_head_dim 512) as well as extent, and perlayer.go warns that declaring a
// window without the wider global head under-counts the global layers — the
// dangerous direction for an admission gate. The profile therefore carries the
// head slice too, and mirrors it into VHeadDim because a gemma4 global layer has
// no v_proj at all (V is the raw k_proj output, gemma4.go:150-154) so V is
// exactly as wide as K on every layer.
func TestKVCacheShapePerLayerHeadGeometry(t *testing.T) {
	cfg := gemma4ShapedConfig()
	cfg.HeadDimPerLayer = make([]int, cfg.NumLayers)
	for l := range cfg.HeadDimPerLayer {
		if (l+1)%6 == 0 {
			cfg.HeadDimPerLayer[l] = 512 // global
			continue
		}
		cfg.HeadDimPerLayer[l] = 256 // local
	}
	got := cfg.KVCacheShape()
	if got.PerLayer == nil {
		t.Fatalf("PerLayer = nil, want a profile")
	}
	if got.PerLayer.HeadDim[0] != 256 || got.PerLayer.HeadDim[5] != 512 {
		t.Errorf("PerLayer.HeadDim[0,5] = %d,%d, want 256,512", got.PerLayer.HeadDim[0], got.PerLayer.HeadDim[5])
	}
	if got.PerLayer.VHeadDim == nil {
		t.Fatalf("PerLayer.VHeadDim = nil for a square-headed config, want K's per-layer width mirrored")
	}
	if got.PerLayer.VHeadDim[0] != 256 || got.PerLayer.VHeadDim[5] != 512 {
		t.Errorf("PerLayer.VHeadDim[0,5] = %d,%d, want 256,512", got.PerLayer.VHeadDim[0], got.PerLayer.VHeadDim[5])
	}
	// 25 local layers × 1024 tokens × 2 heads × (256+256) + 5 global × 16384 ×
	// 2 × (512+512) = 193_986_560 elems; F16 ⇒ 0.361328125 GiB.
	const ctx = 16384
	if e, want := got.KVElemsPerStream(ctx), 25*1024*2*(256+256)+5*ctx*2*(512+512); e != want {
		t.Errorf("KVElemsPerStream(%d) = %d, want %d", ctx, e, want)
	}
	if g, want := got.KVGiBPerStream(ctx, kvbudget.F16), 0.361328125; g != want {
		t.Errorf("KVGiBPerStream(%d, F16) = %v, want %v", ctx, g, want)
	}
	// A header that DOES declare a rectangular head carries no per-layer V width,
	// so those layers must fall back to the scalar instead of inheriting K's.
	rect := cfg
	rect.VHeadDim = 64
	if p := rect.KVCacheShape().PerLayer; p == nil || p.VHeadDim != nil {
		t.Errorf("rectangular-head PerLayer.VHeadDim = %v, want nil (no per-layer V data to carry)", p)
	}
}

// TestKVCacheShapePerLayerShrinksPerStreamBudget is the EFFECT witness: the
// wiring is only worth anything if the smaller number reaches the planner. It
// pins both ends of the same gemma-4 cadence at ctx=16384/F16 — the uniform
// figure the pre-wiring Shape produced (1.875 GiB/stream, still exactly what the
// per-TOKEN methods report, since those keep their uniform meaning) and the
// windowed one (0.41015625) — and then walks the FitRow/MaxStreams pipeline the
// admission gate actually reads to show the stream count rises rather than
// merely the byte figure falling.
func TestKVCacheShapePerLayerShrinksPerStreamBudget(t *testing.T) {
	const ctx = 16384
	got := gemma4ShapedConfig().KVCacheShape()

	// What the same Shape sized as uniformly-global: ctx × KV-bytes/token. This
	// is the pre-wiring answer, recomputed from the per-token methods rather than
	// hardcoded, so it tracks the Shape rather than a stale constant.
	uniform := float64(ctx) * got.KVBytesPerToken(kvbudget.F16) / (1 << 30)
	if uniform != 1.875 {
		t.Fatalf("uniform GiB/stream = %v, want 1.875 (the pre-wiring figure)", uniform)
	}
	windowed := got.KVGiBPerStream(ctx, kvbudget.F16)
	if windowed >= uniform {
		t.Fatalf("KVGiBPerStream = %v, want STRICTLY smaller than the uniform %v: the per-layer window is not reaching the byte figure", windowed, uniform)
	}
	// 107_520 cached token-slots (25×1024 + 5×16384) × 2 heads × 1024 elems ×
	// 2 bytes = 440_401_920 B = 0.41015625 GiB.
	if want := 0.41015625; windowed != want {
		t.Errorf("KVGiBPerStream(%d, F16) = %v, want %v", ctx, windowed, want)
	}

	// The pipeline the planner reads, at the package's own free-VRAM budget.
	// FitRow reports at 3-dp, which is the figure MaxStreams is sized against.
	rawUniform := kvbudget.MaxStreams(kvbudget.FreeVRAMGiB, uniform)
	rawWindowed := got.FitRow(ctx, kvbudget.F16).MaxStreamsRaw
	if rawWindowed <= rawUniform {
		t.Fatalf("MaxStreamsRaw windowed=%d uniform=%d, want windowed strictly larger", rawWindowed, rawUniform)
	}
	if rawUniform != 109 || rawWindowed != 502 {
		t.Errorf("MaxStreamsRaw = %d -> %d, want 109 -> 502", rawUniform, rawWindowed)
	}
}

// TestKVCacheShapeNoPerLayerDataStaysNil is a no-regression PIN, not a witness:
// it cannot fail on the pre-wiring source, because the pre-wiring source left
// PerLayer nil unconditionally. Its job is to hold the zero-value promise
// perlayer.go makes — a header with no per-layer data, or one whose per-layer
// slices say nothing the scalars do not already say, must leave PerLayer nil so
// its byte figures take the untouched ctx × bytes-per-token expression and stay
// bit-identical at any quant.
func TestKVCacheShapeNoPerLayerDataStaysNil(t *testing.T) {
	const ctx = 16384
	cases := []struct {
		name string
		cfg  Config
	}{
		{"llama-3-8b: no per-layer slices at all",
			Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128}},
		{"every layer global: the all-sentinel window declares no bound",
			Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128,
				Window: []int{-1, -1, -1, -1, -1, -1, -1, -1}}},
		{"per-layer slices that merely restate the scalars",
			Config{NumLayers: 4, NumKVHeads: 8, HeadDim: 128,
				HeadDimPerLayer:    []int{128, 128, 128, 128},
				NumKVHeadsPerLayer: []int{8, 8, 8, 8}}},
		{"MLA with no window (GLM-5.2 shape)",
			Config{NumLayers: 92, KVLoraRank: 512, QKRopeHeadDim: 64,
				IndexNHeads: 64, IndexHeadDim: 128}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.cfg.KVCacheShape()
			if got.PerLayer != nil {
				t.Fatalf("PerLayer = %+v, want nil (no per-layer refinement is declared)", got.PerLayer)
			}
			for _, q := range []kvbudget.Quant{kvbudget.F16, kvbudget.Q8_0, kvbudget.Q4} {
				want := float64(ctx) * got.KVBytesPerToken(q)
				if b := got.KVBytesPerStream(ctx, q); b != want {
					t.Errorf("KVBytesPerStream(%d, %+v) = %v, want the bit-identical %v", ctx, q, b, want)
				}
			}
		})
	}
}

// TestKVCacheShapeStaysComparable pins the property perlayer.go chose a POINTER
// for: kvbudget.Shape must remain a comparable struct, because
// kvbudget_shape_test.go compares two Shape values with != — a slice field on
// Shape would break that at COMPILE time, and so would this file.
func TestKVCacheShapeStaysComparable(t *testing.T) {
	a := gemma4ShapedConfig().KVCacheShape()
	b := gemma4ShapedConfig().KVCacheShape()
	if a == b {
		t.Errorf("two independently built Shapes compared ==; each carries its own profile pointer")
	}
	if a != a {
		t.Errorf("a Shape did not compare == to itself")
	}
	uniform := Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128}.KVCacheShape()
	if uniform != (Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128}.KVCacheShape()) {
		t.Errorf("two unrefined Shapes from the same header did not compare ==")
	}
}
