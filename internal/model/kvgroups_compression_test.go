package model

import "testing"

// kvgroups_compression_test.go — the #13617 regression: the FAK_HYBRID_KV layer-group
// residency plane must never size a model that declares a per-layer compression schedule
// as window-only. #13556 landed that guard for the two model-level sizing surfaces it
// names (the KV-shape and header-sizing surfaces); the arithmetic
// in kvgroups.go classifies a layer through windowForLayer() alone, so a DeepSeek-V4
// Flash / V4.1 config (sliding_window: 128 plus a compress_ratios schedule) caps EVERY
// layer at min(window, ctx) and drops the compressed rows and indexer keys entirely.
//
// These tests exercise the surfaces the issue names — KVLayerGroupOf,
// KVLayerResidentPositions, KVLayerResidentFloats, ResidentKVFloats and
// KVGroupSavingsAt — and must pass for the non-compressed models the plane was built for
// (that no-regression arm lives in kvgroups_test.go's hybridKVGroupCfg).

// compressionScheduleCfg is a minimal, deterministic compressed-schedule model: two
// layers, both declaring a window, with a per-layer compress_ratios schedule. It mirrors
// the DeepSeek-V4 shape (a sliding_window that lands in Config.Window plus a schedule)
// without depending on a checkpoint fixture, so the residency classification is
// exercised in isolation.
func compressionScheduleCfg() Config {
	return Config{
		NumLayers:      2,
		NumHeads:       4,
		NumKVHeads:     1,
		HeadDim:        8,
		Window:         []int{128, 128},
		CompressRatios: []int{2, 0},
	}
}

// TestKVLayerGroupCompressionScheduleNeverWindowOnly pins the fix on the minimal config:
// a compressed-schedule layer must not be sized window-only, because windowForLayer()
// omits the compressed rows the layer retains.
//
// fak-test:runtime fast est=1ms
func TestKVLayerGroupCompressionScheduleNeverWindowOnly(t *testing.T) {
	cfg := compressionScheduleCfg()
	if !cfg.hasKVCompressionSchedule() {
		t.Fatal("fixture does not declare a compression schedule; the guard arm is vacuous")
	}

	if got := cfg.KVLayerGroupOf(0); got == KVGroupSlidingWindow {
		t.Fatalf("compressing layer 0 classified %v; windowForLayer() alone omits its compressed rows", got)
	}
	const ctx = 4096
	if got, want := cfg.KVLayerResidentPositions(0, ctx), ctx; got != want {
		t.Fatalf("compressing layer 0 resident positions = %d, want %d (full ctx, not window-capped)", got, want)
	}
	if got, want := cfg.KVLayerResidentFloats(0, ctx), ctx*cfg.NumKVHeads*cfg.HeadDim*kvGroupPlanes; got != want {
		t.Fatalf("compressing layer 0 resident floats = %d, want %d", got, want)
	}
}

// TestResidentKVFloatsV41CompressionNeverWindowOnly is the issue's headline witness on
// the published V4.1 fixture: with the gate on, ResidentKVFloats must not equal the
// window-only figure the unguarded classifier produces.
//
// fak-test:runtime fast est=1ms
func TestResidentKVFloatsV41CompressionNeverWindowOnly(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	if !cfg.hasKVCompressionSchedule() {
		t.Fatal("V4.1 fixture does not declare a compression schedule; the witness is vacuous")
	}
	t.Setenv("FAK_HYBRID_KV", "1")

	const ctx = 1048576
	// The window-only figure is what the classifier produces when it reads Window alone.
	windowOnly := 0
	for l := 0; l < cfg.NumLayers; l++ {
		w := cfg.windowForLayer(l)
		if w <= 0 || w > ctx {
			w = ctx
		}
		windowOnly += w * cfg.NumKVHeads * cfg.HeadDim * kvGroupPlanes
	}
	if windowOnly == 0 || windowOnly == cfg.UniformKVFloats(ctx) {
		t.Fatalf("window-only baseline %d is degenerate (uniform=%d); witness vacuous", windowOnly, cfg.UniformKVFloats(ctx))
	}
	if got := cfg.ResidentKVFloats(ctx); got == windowOnly {
		t.Fatalf("ResidentKVFloats(%d) = %d equals the window-only figure; the compression schedule was dropped", ctx, got)
	}
}
