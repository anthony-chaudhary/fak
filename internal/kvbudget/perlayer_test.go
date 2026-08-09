package kvbudget

import "testing"

// gemma4Windows is the per-layer window vector of a gemma-4-shaped interleaved
// architecture: 30 layers whose layer_types repeat five `sliding_attention` then
// one `full_attention` (so layers 5, 11, 17, 23, 29 are global) with
// `sliding_window: 1024` — the public `google/gemma-4-26B-A4B-it` cadence the
// issue (#5498) cites. A 0 entry is a global layer (full causal attention).
func gemma4Windows() []int {
	w := make([]int, 30)
	for l := range w {
		if (l+1)%6 == 0 {
			continue // full_attention layer: no cap
		}
		w[l] = 1024
	}
	return w
}

// TestWindowCappedShapeFitsMoreStreams is the witness for #5498: the SAME cache
// geometry, differing only in whether its layers declare a sliding-window cap,
// must yield a strictly SMALLER KV footprint per stream and a strictly LARGER
// max-streams fit. Before the per-layer window existed the calculator charged
// every layer for the full ctx it will never hold, so MaxStreams under-reported
// and an admission gate refused streams that fit.
func TestWindowCappedShapeFitsMoreStreams(t *testing.T) {
	const ctx = 16384
	uniform := Shape{Kind: MHA, Layers: 30, NumKVHeads: 4, HeadDim: 256, VHeadDim: 256}
	windowed := uniform
	windowed.PerLayer = &LayerProfile{Window: gemma4Windows()}

	// Uniform: all 30 layers × 16384 tokens × 4 heads × (256+256) elems
	// = 1_006_632_960 elems ⇒ ×2 B (F16) = 2 GiB… exactly 1.875 GiB.
	if got, want := uniform.KVElemsPerStream(ctx), 30*ctx*4*512; got != want {
		t.Fatalf("uniform KVElemsPerStream = %d, want %d", got, want)
	}
	if got, want := uniform.KVGiBPerStream(ctx, F16), 1.875; got != want {
		t.Fatalf("uniform KVGiBPerStream = %v, want %v", got, want)
	}
	// Windowed: 25 sliding layers hold min(1024, ctx) = 1024 tokens each and only
	// the 5 global layers hold the full ctx ⇒ 25×1024 + 5×16384 = 107_520 token
	// slots instead of 30×16384 = 491_520.
	if got, want := windowed.KVElemsPerStream(ctx), (25*1024+5*ctx)*4*512; got != want {
		t.Fatalf("windowed KVElemsPerStream = %d, want %d", got, want)
	}
	if got, want := windowed.KVGiBPerStream(ctx, F16), 0.41015625; got != want {
		t.Fatalf("windowed KVGiBPerStream = %v, want %v", got, want)
	}

	// (1) strictly smaller per-stream footprint.
	if !(windowed.KVGiBPerStream(ctx, F16) < uniform.KVGiBPerStream(ctx, F16)) {
		t.Errorf("windowed KVGiBPerStream %v is not < uniform %v — the window is not being charged",
			windowed.KVGiBPerStream(ctx, F16), uniform.KVGiBPerStream(ctx, F16))
	}
	// (2) strictly larger max-streams fit, through the same FitRow/MaxStreams
	// pipeline the planner reads — this is the number that was under-reported.
	uRow, wRow := uniform.FitRow(ctx, F16), windowed.FitRow(ctx, F16)
	if !(wRow.MaxStreamsRaw > uRow.MaxStreamsRaw) {
		t.Errorf("windowed MaxStreamsRaw = %d, want > uniform %d", wRow.MaxStreamsRaw, uRow.MaxStreamsRaw)
	}
	if !(wRow.MaxStreamsUsable > uRow.MaxStreamsUsable) {
		t.Errorf("windowed MaxStreamsUsable = %d, want > uniform %d", wRow.MaxStreamsUsable, uRow.MaxStreamsUsable)
	}
	// The exact cells, so a silent change of method is caught, not just the sign.
	if uRow.KVGiBPerStream != 1.875 || uRow.MaxStreamsRaw != 109 || uRow.MaxStreamsUsable != 88 {
		t.Errorf("uniform row = %+v, want {KV 1.875, raw 109, usable 88}", uRow)
	}
	if wRow.KVGiBPerStream != 0.41 || wRow.MaxStreamsRaw != 502 || wRow.MaxStreamsUsable != 402 {
		t.Errorf("windowed row = %+v, want {KV 0.41, raw 502, usable 402}", wRow)
	}
}

// TestWindowedShapeIsConstantPlusLinear pins the SHAPE of the corrected curve,
// not just its sign: a window-capped arch's worst case is `constant + linear` —
// the sliding layers contribute a fixed amount bounded by their window (flat in
// ctx) and only the global layers grow with ctx.
func TestWindowedShapeIsConstantPlusLinear(t *testing.T) {
	const ctx = 16384
	windowed := Shape{Kind: MHA, Layers: 30, NumKVHeads: 4, HeadDim: 256, VHeadDim: 256,
		PerLayer: &LayerProfile{Window: gemma4Windows()}}
	perLayerPerToken := 4 * (256 + 256) // kv heads × (K + V) width

	// The flat term: 25 sliding layers × their 1024-token window, whatever ctx is.
	constant := 25 * 1024 * perLayerPerToken
	// The linear term: 5 global layers × ctx.
	linear := 5 * ctx * perLayerPerToken
	if got, want := windowed.KVElemsPerStream(ctx), constant+linear; got != want {
		t.Fatalf("KVElemsPerStream(%d) = %d, want constant %d + linear %d", ctx, got, constant, linear)
	}
	// Doubling ctx must grow ONLY the linear term — the 25 sliding layers are
	// already saturated at their window and add nothing.
	grew := windowed.KVElemsPerStream(2*ctx) - windowed.KVElemsPerStream(ctx)
	if want := 5 * ctx * perLayerPerToken; grew != want {
		t.Errorf("doubling ctx grew the cache by %d elems, want %d (the 5 global layers only)", grew, want)
	}
	// Below the window nothing is capped: every layer holds the whole context.
	if got, want := windowed.KVElemsPerStream(512), 30*512*perLayerPerToken; got != want {
		t.Errorf("KVElemsPerStream(512) = %d, want the uncapped %d (ctx < window)", got, want)
	}
}

// TestLayerTokensWindowBound pins the one place the bound is applied:
// min(window, ctx), with "absent or non-positive" meaning full attention.
func TestLayerTokensWindowBound(t *testing.T) {
	s := Shape{Kind: MHA, Layers: 4, NumKVHeads: 1, HeadDim: 8, VHeadDim: 8,
		PerLayer: &LayerProfile{Window: []int{1024, 0, -1}}} // layer 3: past the end
	cases := []struct{ layer, ctx, want int }{
		{0, 4096, 1024}, // capped: window < ctx
		{0, 512, 512},   // uncapped: ctx < window
		{0, 1024, 1024}, // exactly at the window
		{1, 4096, 4096}, // 0 entry ⇒ full attention
		{2, 4096, 4096}, // -1 entry ⇒ full attention (model.Config's sentinel)
		{3, 4096, 4096}, // past the end of the slice ⇒ full attention
	}
	for _, c := range cases {
		if got := s.LayerTokens(c.layer, c.ctx); got != c.want {
			t.Errorf("LayerTokens(layer=%d, ctx=%d) = %d, want %d", c.layer, c.ctx, got, c.want)
		}
	}
	// No profile at all: every layer holds the whole context.
	bare := Shape{Kind: MHA, Layers: 4}
	if got := bare.LayerTokens(0, 4096); got != 4096 {
		t.Errorf("profile-less LayerTokens = %d, want 4096", got)
	}
}

// TestPerLayerHeadGeometryOverridesScalars covers the other half of the
// interleaved-attention geometry: gemma-4's local and global layers differ in
// head WIDTH (`head_dim: 256` vs `global_head_dim: 512`) and kv-head count as
// well as in extent, so a window declared without the wider global head would
// UNDER-count the global layers — the dangerous direction for an admission gate.
func TestPerLayerHeadGeometryOverridesScalars(t *testing.T) {
	const ctx = 4096
	s := Shape{Kind: MHA, Layers: 6, NumKVHeads: 4, HeadDim: 256, VHeadDim: 256,
		PerLayer: &LayerProfile{
			// Five sliding layers at the scalar geometry, then one global layer
			// that is uncapped AND twice as wide with a single kv head.
			Window:     []int{1024, 1024, 1024, 1024, 1024, 0},
			NumKVHeads: []int{0, 0, 0, 0, 0, 1},
			HeadDim:    []int{0, 0, 0, 0, 0, 512},
			VHeadDim:   []int{0, 0, 0, 0, 0, 512},
		}}
	sliding := 5 * 1024 * 4 * (256 + 256) // capped extent, scalar head geometry
	global := 1 * ctx * 1 * (512 + 512)   // full extent, overridden head geometry
	if got, want := s.KVElemsPerStream(ctx), sliding+global; got != want {
		t.Fatalf("KVElemsPerStream = %d, want sliding %d + global %d", got, sliding, global)
	}
	// A zero entry falls back to the scalar, so declaring only the window keeps
	// every head figure exactly as the uniform Shape had it.
	windowOnly := s
	windowOnly.PerLayer = &LayerProfile{Window: s.PerLayer.Window}
	if got, want := windowOnly.KVElemsPerStream(ctx), (5*1024+ctx)*4*(256+256); got != want {
		t.Errorf("window-only KVElemsPerStream = %d, want %d", got, want)
	}
}

// TestWindowedMLAShapeIsBounded proves the bound reaches the MLA + DSA-indexer
// branch too, not only MHA: a window caps the latent + rope key term and the
// indexer key term alike.
func TestWindowedMLAShapeIsBounded(t *testing.T) {
	const ctx = 16384
	mla := GLM52DSA
	// A hypothetical window-capped variant of the same geometry (GLM-5.2 itself
	// declares none): every layer bounded at 4096.
	win := make([]int, mla.Layers)
	for l := range win {
		win[l] = 4096
	}
	mla.PerLayer = &LayerProfile{Window: win}
	if got, want := mla.MLAElemsPerStream(ctx), 92*4096*(512+64); got != want {
		t.Errorf("windowed MLAElemsPerStream = %d, want %d", got, want)
	}
	if got, want := mla.IndexElemsPerStream(ctx), 92*4096*128; got != want {
		t.Errorf("windowed IndexElemsPerStream = %d, want %d", got, want)
	}
	// A quarter of the context per layer ⇒ a quarter of the footprint.
	if got, want := mla.KVGiBPerStream(ctx, F16), GLM52DSA.KVGiBPerStream(ctx, F16)/4; got != want {
		t.Errorf("windowed KVGiBPerStream = %v, want %v", got, want)
	}
	if got, want := mla.MLAGiBPerStream(ctx, F16), GLM52DSA.MLAGiBPerStream(ctx, F16)/4; got != want {
		t.Errorf("windowed MLAGiBPerStream = %v, want %v", got, want)
	}
}

// TestPerLayerZeroValueIsBitIdentical is the no-regression PIN (not a witness):
// a Shape that declares no per-layer profile — every Shape that exists today,
// including GLM52DSA and everything model.Config.KVCacheShape builds — must
// produce the pre-refinement float bit-for-bit, at every quant and context. The
// `want` expressions below are the literal formulas the package used before
// Shape.PerLayer existed (`float64(ctx) × bytes-per-token ÷ GiB`), so this fails
// if the refinement ever re-associates the arithmetic on the uniform path.
//
// By construction this test passes both before and after the change; it exists
// to keep the correctness fix for window-capped archs from re-baselining every
// model, not to demonstrate the fix.
func TestPerLayerZeroValueIsBitIdentical(t *testing.T) {
	shapes := map[string]Shape{
		"glm52dsa":     GLM52DSA,
		"mha_llama":    {Kind: MHA, Layers: 32, NumKVHeads: 8, HeadDim: 128, VHeadDim: 128},
		"mha_rect":     {Kind: MHA, Layers: 4, NumKVHeads: 2, HeadDim: 128, VHeadDim: 64},
		"mla_noindex":  {Layers: 60, KVLoraRank: 512, QKRopeHeadDim: 64},
		"empty_layers": {Kind: MHA},
	}
	// A deliberately non-dyadic bytes-per-element: with 2 / 1 / 0.5 the multiply
	// is exact whatever the order, so only an odd quant can catch a reassociated
	// expression.
	quants := []Quant{F16, Q8_0, Q4, {Name: "odd", BytesPerElem: 0.6}}
	ctxs := []int{1, 4096, 8192, 16384, 131072}
	for name, base := range shapes {
		for _, variant := range []struct {
			label string
			s     Shape
		}{
			{"nil_profile", base},
			{"empty_profile", func() Shape { s := base; s.PerLayer = &LayerProfile{}; return s }()},
		} {
			s := variant.s
			for _, q := range quants {
				for _, ctx := range ctxs {
					// The pre-refinement expressions, verbatim.
					wantKV := float64(ctx) * s.KVBytesPerToken(q) / GiB
					wantMLA := float64(ctx) * s.MLABytesPerToken(q) / GiB
					if got := s.KVGiBPerStream(ctx, q); got != wantKV {
						t.Errorf("%s/%s KVGiBPerStream(%d, %s) = %v, want %v (bit-identical)",
							name, variant.label, ctx, q.Name, got, wantKV)
					}
					if got := s.MLAGiBPerStream(ctx, q); got != wantMLA {
						t.Errorf("%s/%s MLAGiBPerStream(%d, %s) = %v, want %v (bit-identical)",
							name, variant.label, ctx, q.Name, got, wantMLA)
					}
					// FitRow — and so MaxStreams — must be untouched as well.
					wantRow := Row{
						Ctx:              ctx,
						KVGiBPerStream:   reportGiB(wantKV),
						MLAGiBPerStream:  reportGiB(wantMLA),
						MaxStreamsRaw:    MaxStreams(FreeVRAMGiB, reportGiB(wantKV)),
						MaxStreamsUsable: MaxStreams(UsableVRAMGiB, reportGiB(wantKV)),
					}
					if got := s.FitRow(ctx, q); got != wantRow {
						t.Errorf("%s/%s FitRow(%d, %s) = %+v, want %+v",
							name, variant.label, ctx, q.Name, got, wantRow)
					}
				}
				// The element count reduces to the uniform per-token figure too.
				for _, ctx := range ctxs {
					if got, want := s.KVElemsPerStream(ctx), ctx*s.KVElemsPerToken(); got != want {
						t.Errorf("%s/%s KVElemsPerStream(%d) = %d, want ctx × per-token %d",
							name, variant.label, ctx, got, want)
					}
				}
			}
		}
	}
	// The landed doc's own cells, restated against the pre-refinement method.
	if got := DocTable()[0]; got.KVGiBPerStream != 0.494 || got.MaxStreamsRaw != 417 || got.MaxStreamsUsable != 334 {
		t.Errorf("DocTable 4k row = %+v, want {KV 0.494, raw 417, usable 334}", got)
	}
	// Shape must stay COMPARABLE — internal/model's KVCacheShape test compares
	// two Shapes with `!=`, which a slice field would have broken.
	got, want := GLM52DSA, Shape{Layers: 92, KVLoraRank: 512, QKRopeHeadDim: 64, IndexLayers: 92, IndexHeadDim: 128}
	if got != want {
		t.Errorf("GLM52DSA = %+v, want %+v", got, want)
	}
}
