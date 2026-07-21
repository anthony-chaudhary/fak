package l3kv

import (
	"errors"
	"testing"
)

// TestSplitLayerShardsOneLayerOverCeiling: a single layer already larger than the
// per-object ceiling is fail-closed with the typed unshardable error, never a
// shard the backend would refuse.
func TestSplitLayerShardsOneLayerOverCeiling(t *testing.T) {
	shards, err := SplitLayerShards([]int64{10, 300, 20}, 240)
	if !errors.Is(err, errLayerOverCeiling) {
		t.Fatalf("want errLayerOverCeiling, got err=%v shards=%v", err, shards)
	}
	if shards != nil {
		t.Fatalf("want nil shards on fail-closed, got %v", shards)
	}
}

// TestSplitLayerShardsFitsInOne: everything under the ceiling → exactly one shard
// spanning all layers with the summed byte size.
func TestSplitLayerShardsFitsInOne(t *testing.T) {
	shards, err := SplitLayerShards([]int64{50, 60, 70}, 240)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(shards) != 1 {
		t.Fatalf("want 1 shard, got %d: %v", len(shards), shards)
	}
	want := LayerShard{FirstLayer: 0, LastLayer: 2, Bytes: 180}
	if shards[0] != want {
		t.Fatalf("want %+v, got %+v", want, shards[0])
	}
	if shards[0].Layers() != 3 {
		t.Fatalf("want Layers()=3, got %d", shards[0].Layers())
	}
}

// TestSplitLayerShardsMinimalContiguousGroups: N layers that need K shards yield
// exactly K contiguous groups, correct boundaries, none over the ceiling, and a
// full byte-exact partition of the input.
func TestSplitLayerShardsMinimalContiguousGroups(t *testing.T) {
	// Six 100-byte layers under a 240 ceiling: greedy packs 2 per shard (200 ok,
	// 300 would overflow) → 3 shards of [0..1],[2..3],[4..5].
	layerBytes := []int64{100, 100, 100, 100, 100, 100}
	shards, err := SplitLayerShards(layerBytes, 240)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	want := []LayerShard{
		{FirstLayer: 0, LastLayer: 1, Bytes: 200},
		{FirstLayer: 2, LastLayer: 3, Bytes: 200},
		{FirstLayer: 4, LastLayer: 5, Bytes: 200},
	}
	if len(shards) != len(want) {
		t.Fatalf("want %d shards, got %d: %v", len(want), len(shards), shards)
	}
	var total int64
	prevLast := -1
	for i, s := range shards {
		if s != want[i] {
			t.Fatalf("shard %d: want %+v, got %+v", i, want[i], s)
		}
		if s.Bytes > 240 {
			t.Fatalf("shard %d byte size %d over ceiling", i, s.Bytes)
		}
		if s.FirstLayer != prevLast+1 {
			t.Fatalf("shard %d not contiguous: first=%d prevLast=%d", i, s.FirstLayer, prevLast)
		}
		prevLast = s.LastLayer
		total += s.Bytes
	}
	if prevLast != len(layerBytes)-1 {
		t.Fatalf("shards do not cover all layers: lastLayer=%d want=%d", prevLast, len(layerBytes)-1)
	}
	if total != 600 {
		t.Fatalf("byte total %d != input sum 600", total)
	}
}

// TestSplitLayerShardsExactCeiling: a layer whose size exactly equals the ceiling
// is allowed (inclusive) and closes its own shard.
func TestSplitLayerShardsExactCeiling(t *testing.T) {
	// [80, 240, 80] under 240: 80 fits, 240 exactly fills its own shard (80+240
	// would overflow so a boundary opens before it), then 80 opens the third.
	shards, err := SplitLayerShards([]int64{80, 240, 80}, 240)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	want := []LayerShard{
		{FirstLayer: 0, LastLayer: 0, Bytes: 80},
		{FirstLayer: 1, LastLayer: 1, Bytes: 240},
		{FirstLayer: 2, LastLayer: 2, Bytes: 80},
	}
	if len(shards) != len(want) {
		t.Fatalf("want %d shards, got %d: %v", len(want), len(shards), shards)
	}
	for i := range want {
		if shards[i] != want[i] {
			t.Fatalf("shard %d: want %+v, got %+v", i, want[i], shards[i])
		}
	}
}

// TestSplitLayerShardsBadCeiling: a non-positive ceiling is refused up front.
func TestSplitLayerShardsBadCeiling(t *testing.T) {
	for _, cap := range []int64{0, -1} {
		if _, err := SplitLayerShards([]int64{10}, cap); !errors.Is(err, errBadCeiling) {
			t.Fatalf("cap=%d: want errBadCeiling, got %v", cap, err)
		}
	}
}

// TestSplitLayerShardsNegativeLayer: a layer with a negative byte size is refused.
func TestSplitLayerShardsNegativeLayer(t *testing.T) {
	if _, err := SplitLayerShards([]int64{10, -5, 20}, 240); !errors.Is(err, errNegativeLayer) {
		t.Fatalf("want errNegativeLayer, got %v", err)
	}
}

// TestSplitLayerShardsEmpty: zero layers yield zero shards and no error.
func TestSplitLayerShardsEmpty(t *testing.T) {
	shards, err := SplitLayerShards(nil, 240)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(shards) != 0 {
		t.Fatalf("want 0 shards, got %d: %v", len(shards), shards)
	}
}

// TestSplitUniformLayers: the equal-size convenience shards a 512-unit object of
// uniform layers under a 240 ceiling into layer-aligned shards that reassemble
// byte-exact, and shares the fail-closed one-layer-over-ceiling rule.
func TestSplitUniformLayers(t *testing.T) {
	// 32 layers x 16 bytes = 512 total under a 240 ceiling: greedy packs 15 layers
	// (240) per shard → 15,15,2 → 3 shards, none over ceiling, full coverage.
	shards, err := SplitUniformLayers(32, 16, 240)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(shards) < 3 {
		t.Fatalf("512 bytes under 240 ceiling should need >=3 shards, got %d: %v", len(shards), shards)
	}
	var total int64
	prevLast := -1
	for i, s := range shards {
		if s.Bytes > 240 {
			t.Fatalf("shard %d byte size %d over ceiling", i, s.Bytes)
		}
		if s.FirstLayer != prevLast+1 {
			t.Fatalf("shard %d not contiguous: first=%d prevLast=%d", i, s.FirstLayer, prevLast)
		}
		prevLast = s.LastLayer
		total += s.Bytes
	}
	if prevLast != 31 {
		t.Fatalf("shards do not cover all 32 layers: lastLayer=%d", prevLast)
	}
	if total != 512 {
		t.Fatalf("byte total %d != 512", total)
	}

	// One layer over the ceiling is still fail-closed through the convenience.
	if _, err := SplitUniformLayers(4, 300, 240); !errors.Is(err, errLayerOverCeiling) {
		t.Fatalf("want errLayerOverCeiling, got %v", err)
	}
	// Negative layer count is refused.
	if _, err := SplitUniformLayers(-1, 16, 240); !errors.Is(err, errNegativeLayer) {
		t.Fatalf("want errNegativeLayer for negative layer count, got %v", err)
	}
}
