package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// contextsize_window_test.go — the CONSUMER witnesses for #5520.
//
// #5498 taught Config.KVCacheShape to carry a per-layer attention window, but nothing
// read the corrected Shape: every KV figure a user or the auto-sizer actually saw came
// from compute.EstimateKVStoreBytes over a flat compute.KVConfig that Config.ContextSizeConfig
// built without any window at all. These tests pin the wiring in both directions — the
// windowed model now charges its real per-layer cost, and every model that is NOT
// window-capped charges byte-for-byte what it charged before.

// gemma4WindowedCtx is the context the multiplier below is quoted at. The over-charge
// ratio is context-DEPENDENT (a window only bites past its own width), so any claim about
// "how many times too much" is meaningless without naming the context.
const gemma4WindowedCtx = 16384

// kvPerTokenPerLayerF32 is the per-layer, per-position cost of gemma4ShapedConfig's
// geometry at the default f32 tier: NumKVHeads*HeadDim elements in each of the three rows
// the HAL KV contract stores (pre-RoPE K, post-RoPE K, V), four bytes apiece —
// compute/kvprecision.go:97. Spelled out so the byte totals below are checkable by hand.
const kvPerTokenPerLayerF32 = 2 * 512 * 3 * 4 // = 12288

// TestContextSizeConfigChargesPerLayerWindow is the #5520 witness: the projection every
// user- and scheduler-visible KV figure flows through must charge a windowed layer
// min(window, ctx) positions, not the whole context.
//
// The arithmetic, on the pinned public gemma-4-26B-A4B-it cadence (30 layers, five
// sliding_attention at sliding_window=1024 per full_attention layer, so 25 local + 5
// global) at a 16384-token context:
//
//	uniform  layer-slots = 30 * 16384                 = 491520
//	windowed layer-slots = 25 * 1024  +  5 * 16384    = 107520
//	ratio                = 491520 / 107520            = 32/7 ~= 4.571x
//
// The ratio is not a constant of the model: it is 8/3 ~= 2.67x at ctx=4096, 32/7 ~= 4.57x
// at 16384, ~5.78x at 131072, and tends to 30/5 = 6x as the context grows. The 4.5x the
// issue quotes is the ~16K figure.
func TestContextSizeConfigChargesPerLayerWindow(t *testing.T) {
	t.Setenv("FAK_HYBRID_KV", "1")
	cfg := gemma4ShapedConfig()
	kv := cfg.ContextSizeConfig().KV

	// The projection must carry a window entry for EVERY layer the estimator iterates:
	// windowCappedPositions loops l < cfg.NumLayers and reads any layer past the end of
	// WindowPerLayer as full attention (compute/capacity.go:135-141), so a slice shorter than
	// NumLayers silently re-charges the tail at the uniform rate — the exact bug this test
	// exists to catch, at any layer count.
	if len(kv.WindowPerLayer) != kv.NumLayers {
		t.Fatalf("WindowPerLayer = %v (len %d), want one entry per layer (NumLayers = %d): the corrected KVCacheShape is not reaching compute.KVConfig",
			kv.WindowPerLayer, len(kv.WindowPerLayer), kv.NumLayers)
	}
	// And each entry is the SOURCE cadence with kvWindowPerLayer's normalization applied
	// (kvbudget_shape.go:122-127): a positive window verbatim, the loader's -1
	// full-attention sentinel flattened to the non-positive "no window" spelling
	// WindowPerLayer reads.
	for l, got := range kv.WindowPerLayer {
		wantWindow := cfg.Window[l]
		if wantWindow < 0 {
			wantWindow = 0 // the -1 full-attention sentinel is not a one-token bound
		}
		if got != wantWindow {
			t.Fatalf("WindowPerLayer[%d] = %d, want %d (source cadence Config.Window[%d] = %d)", l, got, wantWindow, l, cfg.Window[l])
		}
	}

	const uniformSlots = 30 * gemma4WindowedCtx           // 491520
	const windowedSlots = 25*1024 + 5*gemma4WindowedCtx   // 107520
	wantUniform := int64(uniformSlots) * kvPerTokenPerLayerF32
	wantWindowed := int64(windowedSlots) * kvPerTokenPerLayerF32

	got := compute.EstimateKVStoreBytes(kv, gemma4WindowedCtx)
	if got == wantUniform {
		t.Fatalf("EstimateKVStoreBytes = %d, the UNIFORM over-estimate: the per-layer window is not being charged", got)
	}
	if got != wantWindowed {
		t.Fatalf("EstimateKVStoreBytes(ctx=%d) = %d, want %d (25 layers capped at 1024 + 5 global at full ctx)",
			gemma4WindowedCtx, got, wantWindowed)
	}

	// The quoted multiplier, asserted exactly in integers so no float rounding can hide a
	// drift: uniform/windowed == 32/7.
	if wantUniform*7 != got*32 {
		t.Errorf("over-charge ratio is not 32/7: uniform %d vs windowed %d", wantUniform, got)
	}
}

// TestContextSizeConfigGateOffStaysUniform pins the gate. The saving is only REAL once the
// cache drops aged-out positions on a windowed layer, and Session.TrimToWindow has no
// production caller yet (swa.go:14), so the default must keep charging the conservative
// uniform reservation — charging less than the cache allocates would trade a refusal for
// an OOM. With FAK_HYBRID_KV unset the projection carries no window and the byte figure is
// bit-identical to the pre-#5520 one.
func TestContextSizeConfigGateOffStaysUniform(t *testing.T) {
	t.Setenv("FAK_HYBRID_KV", "")
	kv := gemma4ShapedConfig().ContextSizeConfig().KV

	if kv.WindowPerLayer != nil {
		t.Fatalf("gate off: WindowPerLayer = %v, want nil", kv.WindowPerLayer)
	}
	want := int64(30*gemma4WindowedCtx) * kvPerTokenPerLayerF32
	if got := compute.EstimateKVStoreBytes(kv, gemma4WindowedCtx); got != want {
		t.Fatalf("gate off: EstimateKVStoreBytes = %d, want the uniform %d", got, want)
	}
}

// TestContextSizeConfigNoWindowUnchanged is the no-regression half of the green gate: a
// model that is uniformly global (every Llama/Qwen checkpoint) must project the SAME
// KVConfig and the SAME bytes with the gate ON as it did before #5520 — at every context
// and at both precision tiers. A refinement that says nothing new must not exist at all.
func TestContextSizeConfigNoWindowUnchanged(t *testing.T) {
	t.Setenv("FAK_HYBRID_KV", "1")
	plain := Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, RopeTheta: 10000}
	kv := plain.ContextSizeConfig().KV

	if kv.WindowPerLayer != nil {
		t.Fatalf("uniformly-global model: WindowPerLayer = %v, want nil", kv.WindowPerLayer)
	}
	perLayer := int64(8 * 128 * 3 * 4)
	for _, ctx := range []int{1, 512, 4096, 131072} {
		for _, prec := range []compute.KVPrecision{compute.KVPrecisionF32, compute.KVPrecisionQ8} {
			cfg := kv
			cfg.Precision = prec
			bare := compute.KVConfig{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, RopeTheta: 10000, Precision: prec}
			if got, want := compute.EstimateKVStoreBytes(cfg, ctx), compute.EstimateKVStoreBytes(bare, ctx); got != want {
				t.Errorf("ctx=%d prec=%v: projected %d, want the window-free %d", ctx, prec, got, want)
			}
		}
	}
	if got, want := compute.EstimateKVStoreBytes(kv, 4096), int64(32*4096)*perLayer; got != want {
		t.Errorf("f32 ctx=4096: %d, want %d", got, want)
	}
}

// TestContextSizeConfigWindowWiderThanContextUnchanged covers the other identity edge: a
// window-capped model asked for a context SHORTER than its window is charged exactly the
// uniform figure, because no layer's bound actually bites there. This is what keeps the
// gate from silently changing short-context serving for a gemma-4-class model.
func TestContextSizeConfigWindowWiderThanContextUnchanged(t *testing.T) {
	t.Setenv("FAK_HYBRID_KV", "1")
	kv := gemma4ShapedConfig().ContextSizeConfig().KV // window 1024
	for _, ctx := range []int{1, 512, 1024} {
		want := int64(30*ctx) * kvPerTokenPerLayerF32
		if got := compute.EstimateKVStoreBytes(kv, ctx); got != want {
			t.Errorf("ctx=%d (at or below the 1024 window): %d, want the uniform %d", ctx, got, want)
		}
	}
	// One token past the window is the first context where the charge diverges.
	if got, want := compute.EstimateKVStoreBytes(kv, 1025), int64(30*1025)*kvPerTokenPerLayerF32; got >= want {
		t.Errorf("ctx=1025: %d, want strictly below the uniform %d", got, want)
	}
}

// TestContextSizeConfigAutoSizerDerivationUnchanged pins WHERE this fix lands, and where it
// deliberately does not.
//
// AutoSizeContextPlan has two halves. The CHARGE half (PerContextMemoryPlan ->
// EstimateKVStoreMemoryPlan) is what the user-visible `kv=` line and the admission demand row
// read, and that is the half the window discount reaches. The DERIVATION half
// (contextsizer.go:87 largestFittingContext) computes a single per-token rate as
// EstimateKVStoreBytes(KV, 1) and divides the budget by it — and at tokens=1 no positive
// window can bite, since a window caps a layer only once the context grows strictly PAST it.
// So the derived context-token count is bit-identical with the gate on, and stays the
// conservative uniform-rate figure.
//
// That asymmetry is the SAFE direction and this test exists to keep it that way: making the
// derivation window-aware would hand back a LARGER context, which is only sound once the
// cache actually drops aged-out positions (it does not yet — swa.go:14). A future change that
// linearizes the derivation over a windowed rate turns this test red before it can turn a
// refusal into an OOM.
func TestContextSizeConfigAutoSizerDerivationUnchanged(t *testing.T) {
	cfg := gemma4ShapedConfig()
	cfg.MaxPositionEmbeddings = 131072
	const avail = 8 << 30 // 8 GiB ceiling: big enough to clear the 512-token floor, small
	// enough that the derivation is a real largest-fit rather than a clamp to MaxContext.

	t.Setenv("FAK_HYBRID_KV", "")
	offTokens, offPlan := compute.AutoSizeContextPlan(cfg.ContextSizeConfig(), nil, avail, -1)

	t.Setenv("FAK_HYBRID_KV", "1")
	onTokens, onPlan := compute.AutoSizeContextPlan(cfg.ContextSizeConfig(), nil, avail, -1)

	if offTokens <= compute.MinAutoContextTokens || offTokens >= cfg.MaxPositionEmbeddings {
		t.Fatalf("derived %d tokens, which is a clamp endpoint — the test would be vacuous; retune avail", offTokens)
	}
	if onTokens != offTokens {
		t.Fatalf("gate flipped the DERIVED context: %d with the window vs %d without. largestFittingContext must keep dividing by the uniform per-token rate until the cache trims, or this reserves less than it allocates",
			onTokens, offTokens)
	}
	// The charge at that same token count is where the discount does land.
	offKV, onKV := offPlan.ByClass()[compute.MemoryKVCache], onPlan.ByClass()[compute.MemoryKVCache]
	if offKV <= 0 || onKV <= 0 {
		t.Fatalf("no kv_cache demand row: off=%d on=%d", offKV, onKV)
	}
	if onKV >= offKV {
		t.Errorf("kv_cache demand at %d tokens: %d with the window, %d without — the discount is not reaching the plan", onTokens, onKV, offKV)
	}
	wantOn := int64(25*1024+5*onTokens) * kvPerTokenPerLayerF32
	if onKV != wantOn {
		t.Errorf("kv_cache demand = %d, want %d (25 layers capped at 1024 + 5 global at %d)", onKV, wantOn, onTokens)
	}
}
