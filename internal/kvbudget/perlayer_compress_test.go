package kvbudget

import (
	"math"
	"testing"
)

// v4FlashPinnedRatios is the full 46-entry `compress_ratios` of the pinned
// DeepSeek-V4 Flash fixture config (43 decoder layers + the MTP tail). Only the
// first NumLayers (43) entries size decoder KV; the forward loop never visits
// the tail, so the test Shape slices it off exactly as the model side does.
var v4FlashPinnedRatios = []int{
	0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
	4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
	4, 128, 4, 0, 0, 0,
}

const (
	v4FlashTestLayers     = 43
	v4FlashTestWindow     = 128 // every layer keeps a 128-row sliding window
	v4FlashTestRowElems   = 512 // head_dim: one shared K=V latent row, rope inside
	v4FlashTestIndexElems = 128 // index_head_dim: one indexer key per ratio-4 row
)

// v4FlashTestShape builds the V4-Flash-shaped Shape the spec pins, entirely in
// the test: MLA with a 512-wide row and no separate rope term, a 128-row window
// on all 43 layers, the pinned compression schedule, and an indexer key of 128
// on the ratio-4 (overlapping, indexed) layers only.
func v4FlashTestShape() Shape {
	ratios := append([]int(nil), v4FlashPinnedRatios[:v4FlashTestLayers]...)
	win := make([]int, v4FlashTestLayers)
	idx := make([]int, v4FlashTestLayers)
	for l := range win {
		win[l] = v4FlashTestWindow
		if ratios[l] == 4 {
			idx[l] = v4FlashTestIndexElems
		}
	}
	return Shape{
		Kind:          MLA,
		Layers:        v4FlashTestLayers,
		KVLoraRank:    v4FlashTestRowElems,
		QKRopeHeadDim: 0,
		PerLayer: &LayerProfile{
			Window:        win,
			CompressRatio: ratios,
			IndexHeadDim:  idx,
		},
	}
}

// handCeil is the test's own ceiling division for positive operands.
func handCeil(a, b int) int { return (a + b - 1) / b }

// TestV4FlashScheduleCounts is the precondition every golden below rests on:
// the first 43 pinned ratios are 2 uncompressed, 21 ratio-4 (even layers ≥ 2)
// and 20 ratio-128 (odd layers ≥ 3).
func TestV4FlashScheduleCounts(t *testing.T) {
	s := v4FlashTestShape()
	r := s.PerLayer.CompressRatio
	if len(r) != 43 {
		t.Fatalf("len(CompressRatio) = %d, want 43", len(r))
	}
	counts := map[int]int{}
	for l, ratio := range r {
		counts[ratio]++
		var want int
		switch {
		case l < 2:
			want = 0
		case l%2 == 0:
			want = 4
		default:
			want = 128
		}
		if ratio != want {
			t.Errorf("CompressRatio[%d] = %d, want %d", l, ratio, want)
		}
		wantIdx := 0
		if ratio == 4 {
			wantIdx = 128
		}
		if got := s.PerLayer.IndexHeadDim[l]; got != wantIdx {
			t.Errorf("IndexHeadDim[%d] = %d, want %d", l, got, wantIdx)
		}
	}
	if counts[0] != 2 || counts[4] != 21 || counts[128] != 20 || len(counts) != 3 {
		t.Fatalf("ratio counts = %v, want {0:2 4:21 128:20}", counts)
	}
}

// TestV4FlashKVBytesGolden pins KVBytesPerStream(ctx, F32) against the spec's
// golden table (window + compressed + indexer rows, f32), recomputed
// independently for this test [SW-VERIFIED]. 32,769 pins the ceil: a floor would
// drop one ratio-4 row on 21 layers and one ratio-128 row on 20.
func TestV4FlashKVBytesGolden(t *testing.T) {
	if F32.Name != "f32" || F32.BytesPerElem != 4 {
		t.Fatalf("F32 = %+v, want {f32 4}", F32)
	}
	s := v4FlashTestShape()
	golden := []struct {
		ctx  int
		want int64
	}{
		{1, 182_784},
		{127, 12_945_408},
		{128, 13_033_472},
		{129, 13_128_192},
		{8192, 123_994_112},
		{32768, 462_159_872},
		{32769, 462_254_592},
		{131072, 1_814_822_912},
		{1048576, 14_439_677_952},
	}
	for _, g := range golden {
		got := s.KVBytesPerStream(g.ctx, F32)
		if got != float64(g.want) {
			t.Errorf("KVBytesPerStream(%d, F32) = %.0f, want %d", g.ctx, got, g.want)
		}
		if elems := s.KVElemsPerStream(g.ctx); int64(elems)*4 != g.want {
			t.Errorf("KVElemsPerStream(%d) = %d, want %d (= bytes / 4)", g.ctx, elems, g.want/4)
		}
	}
	// Past the window, on a multiple of the largest ratio, every compressed row is
	// whole: the curve is exactly window (43×128 rows × 2048 B) + 13,760 B/token,
	// the per-token slope Σ_{r4}(2048+512)/4 + Σ_{r128} 2048/128 = 21×640 + 20×16.
	const windowBytes = 43 * 128 * 512 * 4
	const slope = 21*(512*4+128*4)/4 + 20*(512*4)/128
	if slope != 13_760 || windowBytes != 11_272_192 {
		t.Fatalf("hand constants slope=%d window=%d, want 13760/11272192", slope, windowBytes)
	}
	for _, ctx := range []int{128, 256, 8192, 32768, 131072, 1048576} {
		if got, want := s.KVBytesPerStream(ctx, F32), float64(windowBytes+slope*ctx); got != want {
			t.Errorf("KVBytesPerStream(%d, F32) = %.0f, want window+slope·ctx = %.0f", ctx, got, want)
		}
	}
	// The spec's affine bound: bytes(t) ≤ 11,366,912 + 13,760·t for every t ≥ 0
	// (fixed = window + one ceil-slack row+index per compressing layer).
	const fixedBound = windowBytes + 21*(2048+512) + 20*2048
	if fixedBound != 11_366_912 {
		t.Fatalf("fixed bound = %d, want 11366912", fixedBound)
	}
	for _, ctx := range []int{0, 1, 2, 3, 5, 127, 128, 129, 255, 257, 1000, 8191, 8193, 32769, 131071} {
		if got, bound := s.KVBytesPerStream(ctx, F32), float64(fixedBound+slope*ctx); got > bound {
			t.Errorf("KVBytesPerStream(%d, F32) = %.0f exceeds the affine bound %.0f", ctx, got, bound)
		}
	}
}

// TestV4FlashElemsDecomposition derives the 8,192-token figure by hand, term by
// term, so a silent change of method (not just of total) is caught:
//
//	MLA rows  = window 43×128 + ratio-4 21×ceil(8192/4) + ratio-128 20×ceil(8192/128)
//	          = 5,504 + 43,008 + 1,280 = 49,792 rows × 512 elems
//	indexer   = ratio-4 layers only: 21 × 2,048 rows × 128 elems
func TestV4FlashElemsDecomposition(t *testing.T) {
	const ctx = 8192
	s := v4FlashTestShape()
	window := 43 * 128
	r4 := 21 * handCeil(ctx, 4)
	r128 := 20 * handCeil(ctx, 128)
	if window != 5504 || r4 != 43008 || r128 != 1280 {
		t.Fatalf("hand rows window=%d r4=%d r128=%d, want 5504/43008/1280", window, r4, r128)
	}
	wantMLA := (window + r4 + r128) * 512  // 25,493,504
	wantIdx := 21 * handCeil(ctx, 4) * 128 // 5,505,024
	if wantMLA != 25_493_504 || wantIdx != 5_505_024 {
		t.Fatalf("hand elems mla=%d idx=%d, want 25493504/5505024", wantMLA, wantIdx)
	}
	if got := s.MLAElemsPerStream(ctx); got != wantMLA {
		t.Errorf("MLAElemsPerStream(%d) = %d, want %d", ctx, got, wantMLA)
	}
	if got := s.IndexElemsPerStream(ctx); got != wantIdx {
		t.Errorf("IndexElemsPerStream(%d) = %d, want %d (ratio-4 layers only)", ctx, got, wantIdx)
	}
	if got := s.KVElemsPerStream(ctx); got != wantMLA+wantIdx {
		t.Errorf("KVElemsPerStream(%d) = %d, want MLA %d + index %d", ctx, got, wantMLA, wantIdx)
	}
	if got, want := s.MLABytesPerStream(ctx, F32), float64(wantMLA*4); got != want {
		t.Errorf("MLABytesPerStream(%d, F32) = %.0f, want %.0f", ctx, got, want)
	}
	// Removing the new fields collapses the same profile to window-only: every
	// layer then holds just its 128 rows and nothing indexes (IndexLayers is 0).
	// This is the undercount the schedule charge exists to correct.
	windowOnly := s
	windowOnly.PerLayer = &LayerProfile{Window: s.PerLayer.Window}
	if got, want := windowOnly.KVBytesPerStream(ctx, F32), float64(43*128*512*4); got != want {
		t.Errorf("window-only KVBytesPerStream(%d, F32) = %.0f, want %.0f", ctx, got, want)
	}
	if !(s.KVBytesPerStream(ctx, F32) > windowOnly.KVBytesPerStream(ctx, F32)) {
		t.Errorf("compressed %.0f not > window-only %.0f", s.KVBytesPerStream(ctx, F32), windowOnly.KVBytesPerStream(ctx, F32))
	}
}

// TestCompressedRowsCeil pins CompressedRows = ceil(ctx/r) for r > 0 and ctx > 0,
// else 0, and LayerTokens = min(window, ctx) + CompressedRows on top.
func TestCompressedRowsCeil(t *testing.T) {
	s := v4FlashTestShape()
	rows := []struct{ layer, ctx, want int }{
		{2, 32769, 8193}, // ratio 4: ceil(32769/4)
		{3, 32769, 257},  // ratio 128: ceil(32769/128)
		{2, 32768, 8192}, // exact multiple: no slack row
		{3, 32768, 256},
		{2, 1, 1}, // one token already opens a compressed row
		{3, 1, 1},
		{3, 129, 2},
		{0, 32769, 0}, // ratio 0: compresses nothing
		{1, 32769, 0},
		{2, 0, 0},      // ctx 0: nothing
		{3, 0, 0},      // ctx 0: nothing
		{2, -5, 0},     // negative ctx: nothing
		{43, 32769, 0}, // past the end of the schedule
		{-1, 32769, 0}, // negative layer
	}
	for _, c := range rows {
		if got := s.CompressedRows(c.layer, c.ctx); got != c.want {
			t.Errorf("CompressedRows(layer=%d, ctx=%d) = %d, want %d", c.layer, c.ctx, got, c.want)
		}
	}
	tokens := []struct{ layer, ctx, want int }{
		{2, 1, 1 + 1},
		{2, 127, 127 + 32},
		{2, 128, 128 + 32},
		{2, 129, 128 + 33},
		{2, 32769, 128 + 8193},
		{3, 129, 128 + 2},
		{3, 32769, 128 + 257},
		{0, 32769, 128}, // uncompressed: window only
		{1, 50, 50},     // below the window, uncompressed
		{2, 0, 0},
		{3, 0, 0},
	}
	for _, c := range tokens {
		if got := s.LayerTokens(c.layer, c.ctx); got != c.want {
			t.Errorf("LayerTokens(layer=%d, ctx=%d) = %d, want %d", c.layer, c.ctx, got, c.want)
		}
	}
	// Non-positive entries opt out just like a non-positive Window entry.
	neg := Shape{Kind: MLA, Layers: 3, KVLoraRank: 8,
		PerLayer: &LayerProfile{CompressRatio: []int{-4, 0, 4}}}
	for l, want := range []int{0, 0, 3} {
		if got := neg.CompressedRows(l, 10); got != want {
			t.Errorf("CompressedRows(layer=%d, ctx=10) with ratio %d = %d, want %d",
				l, neg.PerLayer.CompressRatio[l], got, want)
		}
	}
	// No profile at all: nothing compresses.
	if got := (Shape{Kind: MLA, Layers: 4}).CompressedRows(0, 4096); got != 0 {
		t.Errorf("profile-less CompressedRows = %d, want 0", got)
	}
}

// TestCompressRatioWithoutWindow covers a schedule declared with no Window: the
// compressed rows ride on top of full attention, so the n×ctx shortcut must not
// be taken just because Window is absent.
func TestCompressRatioWithoutWindow(t *testing.T) {
	s := Shape{Kind: MLA, Layers: 2, KVLoraRank: 4,
		PerLayer: &LayerProfile{CompressRatio: []int{4}}} // layer 1: past the end
	if got, want := s.LayerTokens(0, 10), 10+3; got != want {
		t.Errorf("LayerTokens(0, 10) = %d, want %d (ctx + ceil(10/4))", got, want)
	}
	if got, want := s.LayerTokens(1, 10), 10; got != want {
		t.Errorf("LayerTokens(1, 10) = %d, want %d", got, want)
	}
	if got, want := s.MLAElemsPerStream(10), (13+10)*4; got != want {
		t.Errorf("MLAElemsPerStream(10) = %d, want %d", got, want)
	}
	if got, want := s.KVBytesPerStream(10, F16), float64((13+10)*4*2); got != want {
		t.Errorf("KVBytesPerStream(10, F16) = %v, want %v", got, want)
	}
}

// TestPerLayerIndexHeadDimFallback pins the per-layer indexer width rule: a
// positive entry wins; otherwise the scalar IndexHeadDim applies to layers below
// IndexLayers and nothing to the rest. A compressing layer indexes its
// compressed rows; any other layer indexes its retained tokens.
func TestPerLayerIndexHeadDimFallback(t *testing.T) {
	const ctx = 10
	s := Shape{Kind: MLA, Layers: 4, KVLoraRank: 8, IndexLayers: 2, IndexHeadDim: 4,
		PerLayer: &LayerProfile{
			Window:       []int{3, 0, 0, 0},
			IndexHeadDim: []int{0, 16, 0, 32},
		}}
	// l0: scalar 4 over min(3,10)=3 rows; l1: 16 × 10; l2: past IndexLayers ⇒ 0;
	// l3: 32 × 10.
	if got, want := s.IndexElemsPerStream(ctx), 3*4+10*16+0+10*32; got != want {
		t.Errorf("IndexElemsPerStream(%d) = %d, want %d", ctx, got, want)
	}
	// A compressing layer with no per-layer width takes the scalar per compressed
	// row; an uncompressed one per retained token.
	c := Shape{Kind: MLA, Layers: 2, KVLoraRank: 8, IndexLayers: 2, IndexHeadDim: 4,
		PerLayer: &LayerProfile{CompressRatio: []int{4, 0}}}
	if got, want := c.IndexElemsPerStream(ctx), handCeil(ctx, 4)*4+ctx*4; got != want {
		t.Errorf("compressing IndexElemsPerStream(%d) = %d, want %d", ctx, got, want)
	}
}

// TestCompressFieldsEmptyIsBitIdentical is the no-regression pin for the two new
// LayerProfile slices: a profile whose only content is EMPTY (non-nil)
// CompressRatio / IndexHeadDim slices must answer bit-for-bit what a nil profile
// answers — and what the pre-refinement ctx × bytes-per-token expression answers
// — for GLM52DSA and an MHA shape at every quant.
func TestCompressFieldsEmptyIsBitIdentical(t *testing.T) {
	shapes := map[string]Shape{
		"glm52dsa":  GLM52DSA,
		"mha_llama": {Kind: MHA, Layers: 32, NumKVHeads: 8, HeadDim: 128, VHeadDim: 128},
	}
	quants := []Quant{F16, Q8_0, Q4, F32, {Name: "odd", BytesPerElem: 0.6}}
	ctxs := []int{1, 127, 4096, 32769, 131072}
	for name, base := range shapes {
		empty := base
		empty.PerLayer = &LayerProfile{CompressRatio: []int{}, IndexHeadDim: []int{}}
		for _, q := range quants {
			for _, ctx := range ctxs {
				pairs := []struct {
					what             string
					viaNil, viaEmpty float64
					legacy           float64
				}{
					{"KVBytesPerStream", base.KVBytesPerStream(ctx, q), empty.KVBytesPerStream(ctx, q),
						float64(ctx) * base.KVBytesPerToken(q)},
					{"MLABytesPerStream", base.MLABytesPerStream(ctx, q), empty.MLABytesPerStream(ctx, q),
						float64(ctx) * base.MLABytesPerToken(q)},
					{"KVGiBPerStream", base.KVGiBPerStream(ctx, q), empty.KVGiBPerStream(ctx, q),
						float64(ctx) * base.KVBytesPerToken(q) / GiB},
					{"MLAGiBPerStream", base.MLAGiBPerStream(ctx, q), empty.MLAGiBPerStream(ctx, q),
						float64(ctx) * base.MLABytesPerToken(q) / GiB},
				}
				for _, p := range pairs {
					if math.Float64bits(p.viaEmpty) != math.Float64bits(p.viaNil) {
						t.Errorf("%s/%s %s(%d): empty profile %v != nil profile %v (bits)",
							name, q.Name, p.what, ctx, p.viaEmpty, p.viaNil)
					}
					if math.Float64bits(p.viaEmpty) != math.Float64bits(p.legacy) {
						t.Errorf("%s/%s %s(%d): empty profile %v != legacy expression %v (bits)",
							name, q.Name, p.what, ctx, p.viaEmpty, p.legacy)
					}
				}
				if got, want := empty.FitRow(ctx, q), base.FitRow(ctx, q); got != want {
					t.Errorf("%s/%s FitRow(%d): empty %+v != nil %+v", name, q.Name, ctx, got, want)
				}
			}
		}
		for _, ctx := range ctxs {
			if got, want := empty.KVElemsPerStream(ctx), ctx*base.KVElemsPerToken(); got != want {
				t.Errorf("%s KVElemsPerStream(%d) = %d, want %d", name, ctx, got, want)
			}
			if got, want := empty.IndexElemsPerStream(ctx), ctx*base.IndexElemsPerToken(); got != want {
				t.Errorf("%s IndexElemsPerStream(%d) = %d, want %d", name, ctx, got, want)
			}
			if got, want := empty.MLAElemsPerStream(ctx), ctx*base.MLAElemsPerToken(); got != want {
				t.Errorf("%s MLAElemsPerStream(%d) = %d, want %d", name, ctx, got, want)
			}
			for l := 0; l < 3; l++ {
				if got := empty.CompressedRows(l, ctx); got != 0 {
					t.Errorf("%s CompressedRows(%d, %d) = %d, want 0", name, l, ctx, got)
				}
				if got := empty.LayerTokens(l, ctx); got != ctx {
					t.Errorf("%s LayerTokens(%d, %d) = %d, want %d", name, l, ctx, got, ctx)
				}
			}
		}
	}
}

// TestWindowOnlyProfileUnchangedByCompressFields keeps the #5498 window path
// honest: a window-only profile answers its hand-derived pre-compression figure,
// and adding explicitly empty CompressRatio / IndexHeadDim slices changes
// nothing, bit for bit. The indexer still follows the FIRST IndexLayers windows.
func TestWindowOnlyProfileUnchangedByCompressFields(t *testing.T) {
	const ctx = 10
	win := []int{3, 0, 5, 0}
	s := Shape{Kind: MLA, Layers: 4, KVLoraRank: 8, QKRopeHeadDim: 2, IndexLayers: 2, IndexHeadDim: 4,
		PerLayer: &LayerProfile{Window: win}}
	withEmpty := s
	withEmpty.PerLayer = &LayerProfile{Window: win, CompressRatio: []int{}, IndexHeadDim: []int{}}
	// MLA: (3 + 10 + 5 + 10) rows × (8 + 2); index: first two layers (3 + 10) × 4.
	wantMLA, wantIdx := (3+10+5+10)*10, (3+10)*4
	for label, sh := range map[string]Shape{"window_only": s, "window_plus_empty": withEmpty} {
		if got := sh.MLAElemsPerStream(ctx); got != wantMLA {
			t.Errorf("%s MLAElemsPerStream = %d, want %d", label, got, wantMLA)
		}
		if got := sh.IndexElemsPerStream(ctx); got != wantIdx {
			t.Errorf("%s IndexElemsPerStream = %d, want %d", label, got, wantIdx)
		}
		for _, q := range []Quant{F16, Q8_0, Q4, F32, {Name: "odd", BytesPerElem: 0.6}} {
			want := float64(wantMLA+wantIdx) * q.BytesPerElem
			if got := sh.KVBytesPerStream(ctx, q); math.Float64bits(got) != math.Float64bits(want) {
				t.Errorf("%s KVBytesPerStream(%d, %s) = %v, want %v", label, ctx, q.Name, got, want)
			}
		}
	}
	// And the gemma-4 MHA window path, both ways, at a long context.
	mha := Shape{Kind: MHA, Layers: 30, NumKVHeads: 4, HeadDim: 256, VHeadDim: 256,
		PerLayer: &LayerProfile{Window: gemma4Windows()}}
	mhaEmpty := mha
	mhaEmpty.PerLayer = &LayerProfile{Window: gemma4Windows(), CompressRatio: []int{}, IndexHeadDim: []int{}}
	const long = 16384
	if got, want := mha.KVElemsPerStream(long), (25*1024+5*long)*4*512; got != want {
		t.Errorf("gemma window-only KVElemsPerStream = %d, want %d", got, want)
	}
	for _, q := range []Quant{F16, Q8_0, Q4, F32} {
		a, b := mha.KVBytesPerStream(long, q), mhaEmpty.KVBytesPerStream(long, q)
		if math.Float64bits(a) != math.Float64bits(b) {
			t.Errorf("gemma %s: window-only %v != window+empty %v", q.Name, a, b)
		}
	}
}

// TestShapeStaysComparable guards the property callers rely on: Shape remains
// usable with == and as a map key now that LayerProfile carries more slices
// (the profile hangs off a pointer, so the slices never enter the comparison).
func TestShapeStaysComparable(t *testing.T) {
	a := v4FlashTestShape()
	b := a
	if a != b {
		t.Fatal("a copy of the V4 Shape is not == to itself")
	}
	if c := v4FlashTestShape(); a == c {
		t.Error("two Shapes with distinct profile pointers compared ==, want pointer identity")
	}
	seen := map[Shape]int{a: 1, GLM52DSA: 2}
	if seen[b] != 1 || seen[GLM52DSA] != 2 {
		t.Errorf("Shape map lookups = %d/%d, want 1/2", seen[b], seen[GLM52DSA])
	}
}
