package kvmmu

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// recStateTilesHybridCfg is a synthetic hybrid model exercising all three KV
// memory groups: two full-attention layers, two sliding-window layers, and two
// recurrent Gated-DeltaNet linear-attention layers. Its LayerTypes and Window
// are hand-set (no checkpoint load) so the classification and tiling are
// exercised in isolation, mirroring the qwen35 tests' small geometry.
func recStateTilesHybridCfg() model.Config {
	return model.Config{
		NumLayers:  6,
		NumHeads:   4,
		NumKVHeads: 2,
		HeadDim:    8,
		// layer 0: full; layer 1: window (w=4); layer 2: recurrent;
		// layer 3: full; layer 4: recurrent; layer 5: window (w=4).
		LayerTypes: []string{"full_attention", "sliding_attention", "linear_attention",
			"full_attention", "linear_attention", "sliding_attention"},
		Window: []int{-1, 4, -1, -1, -1, 4},
		// Gated-DeltaNet geometry for the recurrent layers' O(1) state.
		LinearNumKeyHeads:   2,
		LinearNumValueHeads: 3,
		LinearKeyHeadDim:    4,
		LinearValueHeadDim:  4,
		LinearConvKernelDim: 3,
	}
}

// TestRecurrentStateTilesLayoutContract pins the mamba-view layout descriptor:
// the element/byte arithmetic of one recurrent-state tile and the typed
// refusals for unsupported dtypes, nonpositive geometry, and payload overflow.
func TestRecurrentStateTilesLayoutContract(t *testing.T) {
	l, err := NewStateTileLayout("f32", 3, 4, 4)
	if err != nil {
		t.Fatalf("NewStateTileLayout(f32,3,4,4) = %v, want ok", err)
	}
	if l.Dtype != "f32" || l.ElementBytes != 4 || l.Heads != 3 ||
		l.KeyHeadDim != 4 || l.ValueHeadDim != 4 {
		t.Fatalf("layout identity fields = %+v, want f32/4B/3 heads/4x4", l)
	}
	if l.RowElements != 16 || l.Elements != 48 || l.Bytes != 192 {
		t.Fatalf("layout arithmetic = row %d / elements %d / bytes %d, want 16/48/192",
			l.RowElements, l.Elements, l.Bytes)
	}

	// Half-precision accounting: same geometry, half the bytes.
	h, err := NewStateTileLayout("bf16", 3, 4, 4)
	if err != nil {
		t.Fatalf("NewStateTileLayout(bf16,3,4,4) = %v, want ok", err)
	}
	if h.ElementBytes != 2 || h.Bytes != 96 {
		t.Fatalf("bf16 layout = %d B/elem, %d B total, want 2/96", h.ElementBytes, h.Bytes)
	}

	cases := []struct {
		name            string
		dty             string
		heads, kHd, vHd int
		want            error
	}{
		{"unsupported dtype", "f64", 3, 4, 4, ErrUnsupportedStateDtype},
		{"zero heads", "f32", 0, 4, 4, ErrNonPositiveGeometry},
		{"negative head dim", "f32", 3, -4, 4, ErrNonPositiveGeometry},
		{"element product overflow", "f32", 1 << 62, 4, 4, ErrStateTileOverflow},
		{"byte product overflow", "f32", 300_000_000_000_000_000, 4, 4, ErrStateTileOverflow},
	}
	for _, tc := range cases {
		_, err := NewStateTileLayout(tc.dty, tc.heads, tc.kHd, tc.vHd)
		if !errors.Is(err, tc.want) {
			t.Errorf("%s: error = %v, want errors.Is(%v)", tc.name, err, tc.want)
		}
	}

	// The digest is a stable content identity: equal layouts agree, any field
	// difference disagrees.
	if l.Digest() == "" {
		t.Fatal("layout digest is empty")
	}
	other, _ := NewStateTileLayout("f32", 3, 4, 5)
	if l.Digest() == other.Digest() {
		t.Error("digests collide across differing value head dims")
	}
	same, _ := NewStateTileLayout("F32 ", 3, 4, 4) // normalized dtype, same geometry
	if l.Digest() != same.Digest() {
		t.Error("digests differ for the same normalized geometry")
	}
}

// TestRecurrentStateTilesAddressRoundTrip pins the tile addressing contract:
// every (layer, chunk) pair encodes to a unique slab offset and decodes back
// to itself; misaligned, negative, and out-of-range offsets refuse.
func TestRecurrentStateTilesAddressRoundTrip(t *testing.T) {
	cfg := recStateTilesHybridCfg()
	layout, err := NewStateTileLayout("f32", 3, 4, 4)
	if err != nil {
		t.Fatalf("NewStateTileLayout = %v", err)
	}
	const chunks = 2
	plan, err := NewStateTilePlan("kimi-k3.1-hybrid", cfg, layout, chunks)
	if err != nil {
		t.Fatalf("NewStateTilePlan = %v", err)
	}
	if plan.TileCount() != 4 || plan.SlabBytes() != 768 {
		t.Fatalf("plan = %d tiles / %d B slab, want 4 / 768", plan.TileCount(), plan.SlabBytes())
	}

	want := []struct {
		layer, chunk int
		offset       int64
	}{
		{2, 0, 0},
		{2, 1, 192},
		{4, 0, 384},
		{4, 1, 576},
	}
	seen := map[string]bool{}
	for _, wc := range want {
		off, ok := plan.Address(wc.layer, wc.chunk)
		if !ok || off != wc.offset {
			t.Fatalf("Address(%d,%d) = (%d,%v), want (%d,true)", wc.layer, wc.chunk, off, ok, wc.offset)
		}
		gotLayer, gotChunk, ok := plan.Decode(off)
		if !ok || gotLayer != wc.layer || gotChunk != wc.chunk {
			t.Fatalf("Decode(%d) = (%d,%d,%v), want (%d,%d,true)", off, gotLayer, gotChunk, ok, wc.layer, wc.chunk)
		}
		tile, ok := plan.Tile(gotLayer, gotChunk)
		if !ok {
			t.Fatalf("Tile(%d,%d) = !ok after round trip", gotLayer, gotChunk)
		}
		if tile.Group != model.KVGroupRecurrent || tile.Layout != layout {
			t.Fatalf("tile metadata = group %v layout %+v, want recurrent group + plan layout",
				tile.Group, tile.Layout)
		}
		if !strings.HasPrefix(tile.Key, "kvmmu/recstate/v1/kimi-k3.1-hybrid/") {
			t.Fatalf("tile key %q lost the model identity prefix", tile.Key)
		}
		if seen[tile.Key] {
			t.Fatalf("tile keys collide: %q reused", tile.Key)
		}
		seen[tile.Key] = true
	}

	// Determinism: an independently built plan assigns the same keys.
	again, err := NewStateTilePlan("kimi-k3.1-hybrid", cfg, layout, chunks)
	if err != nil {
		t.Fatalf("rebuild NewStateTilePlan = %v", err)
	}
	first, _ := plan.Tile(2, 0)
	second, _ := again.Tile(2, 0)
	if first.Key != second.Key {
		t.Fatalf("tile key is not deterministic across plans: %q vs %q", first.Key, second.Key)
	}

	badAddresses := []struct {
		layer, chunk int
	}{
		{1, 0},  // sliding-window layer: not a recurrent layer of this plan
		{0, 0},  // full-attention layer
		{2, 2},  // chunk past the tiling factor
		{2, -1}, // negative chunk
		{-1, 0}, // negative layer
	}
	for _, bc := range badAddresses {
		if off, ok := plan.Address(bc.layer, bc.chunk); ok {
			t.Errorf("Address(%d,%d) = (%d,true), want refusal", bc.layer, bc.chunk, off)
		}
	}
	for _, off := range []int64{-1, 191, 767, 768} {
		if l, c, ok := plan.Decode(off); ok {
			t.Errorf("Decode(%d) = (%d,%d,true), want refusal", off, l, c)
		}
	}

	// A model with no recurrent layers yields an empty (valid) plan.
	plain := model.Config{NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 8}
	empty, err := NewStateTilePlan("plain", plain, layout, chunks)
	if err != nil {
		t.Fatalf("NewStateTilePlan(plain) = %v, want ok empty plan", err)
	}
	if empty.TileCount() != 0 || empty.SlabBytes() != 0 {
		t.Fatalf("empty plan = %d tiles / %d B, want 0 / 0", empty.TileCount(), empty.SlabBytes())
	}
	if _, ok := empty.Address(0, 0); ok {
		t.Error("empty plan addressed a tile")
	}
	if _, _, ok := empty.Decode(0); ok {
		t.Error("empty plan decoded an offset")
	}
}

// TestRecurrentStateTilesGroupListingIncludesStateTiles pins the unified
// cacheable-object listing: ordinary KV groups appear as token spans, and the
// recurrent group appears as its state tiles — each carrying its own key and
// the distinct mamba-view layout the token spans do not have.
func TestRecurrentStateTilesGroupListingIncludesStateTiles(t *testing.T) {
	cfg := recStateTilesHybridCfg()
	layout, err := NewStateTileLayout("f32", 3, 4, 4)
	if err != nil {
		t.Fatalf("NewStateTileLayout = %v", err)
	}
	plan, err := NewStateTilePlan("kimi-k3.1-hybrid", cfg, layout, 2)
	if err != nil {
		t.Fatalf("NewStateTilePlan = %v", err)
	}

	entries := plan.CacheableListing()
	if len(entries) != 6 {
		t.Fatalf("listing = %d rows, want 2 token spans + 4 state tiles", len(entries))
	}

	// Row 0: the full-attention token span over layers 0 and 3.
	e0 := entries[0]
	if e0.Group != model.KVGroupFull || e0.Kind != ListingKVSpan || e0.Tile != nil {
		t.Fatalf("row 0 = %+v, want full-group token span without a tile", e0)
	}
	if len(e0.Layers) != 2 || e0.Layers[0] != 0 || e0.Layers[1] != 3 {
		t.Fatalf("full span layers = %v, want [0 3]", e0.Layers)
	}

	// Row 1: the sliding-window token span over layers 1 and 5.
	e1 := entries[1]
	if e1.Group != model.KVGroupSlidingWindow || e1.Kind != ListingKVSpan || e1.Tile != nil {
		t.Fatalf("row 1 = %+v, want window-group token span without a tile", e1)
	}
	if len(e1.Layers) != 2 || e1.Layers[0] != 1 || e1.Layers[1] != 5 {
		t.Fatalf("window span layers = %v, want [1 5]", e1.Layers)
	}

	// Rows 2-5: the recurrent state tiles, in layer-then-chunk order, each a
	// first-class object with its own key and the distinct state layout.
	wantTiles := []struct{ layer, chunk int }{{2, 0}, {2, 1}, {4, 0}, {4, 1}}
	for i, wt := range wantTiles {
		e := entries[2+i]
		if e.Group != model.KVGroupRecurrent || e.Kind != ListingStateTile || e.Tile == nil {
			t.Fatalf("row %d = %+v, want recurrent-group state tile", 2+i, e)
		}
		tile := e.Tile
		if tile.Layer != wt.layer || tile.Chunk != wt.chunk {
			t.Fatalf("row %d tile = layer %d chunk %d, want layer %d chunk %d",
				2+i, tile.Layer, tile.Chunk, wt.layer, wt.chunk)
		}
		if tile.Key == "" {
			t.Fatalf("row %d state tile carries no key", 2+i)
		}
		if tile.Layout.Elements != 48 || tile.Layout.Bytes != 192 {
			t.Fatalf("row %d tile layout = %d elems / %d B, want the distinct 48/192 state layout",
				2+i, tile.Layout.Elements, tile.Layout.Bytes)
		}
		if tile.Group != model.KVGroupRecurrent {
			t.Fatalf("row %d tile group = %v, want recurrent", 2+i, tile.Group)
		}
	}
}

// TestRecurrentStateTilesPlanRefusals pins the typed plan-level refusals:
// missing model identity, a sub-one tiling factor, and an empty layout.
func TestRecurrentStateTilesPlanRefusals(t *testing.T) {
	cfg := recStateTilesHybridCfg()
	layout, err := NewStateTileLayout("f32", 3, 4, 4)
	if err != nil {
		t.Fatalf("NewStateTileLayout = %v", err)
	}

	cases := []struct {
		name  string
		model string
		lay   StateTileLayout
		chunk int
		want  error
	}{
		{"empty model id", "", layout, 1, ErrEmptyModelID},
		{"blank model id", "   ", layout, 1, ErrEmptyModelID},
		{"zero chunks", "m", layout, 0, ErrInvalidChunks},
		{"negative chunks", "m", layout, -1, ErrInvalidChunks},
		{"empty layout", "m", StateTileLayout{}, 1, ErrNonPositiveGeometry},
	}
	for _, tc := range cases {
		_, err := NewStateTilePlan(tc.model, cfg, tc.lay, tc.chunk)
		if !errors.Is(err, tc.want) {
			t.Errorf("%s: error = %v, want errors.Is(%v)", tc.name, err, tc.want)
		}
	}
}
