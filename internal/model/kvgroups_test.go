package model

import "testing"

// hybridKVGroupCfg is a synthetic hybrid model exercising all three KV memory groups: a
// recurrent Gated-DeltaNet linear-attention layer, a sliding-window layer, and a full-causal
// layer. Its LayerTypes and Window are hand-set (no checkpoint load) so the classification
// and budget arithmetic are exercised in isolation. The linear-attn geometry mirrors the
// qwen35 tests' small dims so recurrentStateFloats has a concrete, checkable value.
func hybridKVGroupCfg() Config {
	return Config{
		NumLayers:  3,
		NumHeads:   4,
		NumKVHeads: 2,
		HeadDim:    8,
		// layer 0: recurrent state-space; layer 1: sliding window (w=4); layer 2: full.
		LayerTypes: []string{"linear_attention", "sliding_attention", "full_attention"},
		Window:     []int{-1, 4, -1},
		// Gated-DeltaNet geometry for the recurrent layer's O(1) state.
		LinearNumKeyHeads:   2,
		LinearNumValueHeads: 2,
		LinearKeyHeadDim:    4,
		LinearValueHeadDim:  4,
		LinearConvKernelDim: 3,
	}
}

// TestKVLayerGroupClassification pins the vocabulary: each layer classifies into the group
// its LayerTypes/Window imply, recurrent-before-window-before-full.
func TestKVLayerGroupClassification(t *testing.T) {
	cfg := hybridKVGroupCfg()
	want := []KVLayerGroup{KVGroupRecurrent, KVGroupSlidingWindow, KVGroupFull}
	for l, g := range want {
		if got := cfg.KVLayerGroupOf(l); got != g {
			t.Fatalf("layer %d group = %v, want %v", l, got, g)
		}
	}
	// A non-hybrid (no windows, no linear layers) model is uniformly full-attention.
	plain := Config{NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 8}
	for l := 0; l < plain.NumLayers; l++ {
		if got := plain.KVLayerGroupOf(l); got != KVGroupFull {
			t.Fatalf("plain layer %d group = %v, want full-attention", l, got)
		}
	}
}

// TestKVGroupBudgetArithmetic is the #2241 R1 witness: per-group budget arithmetic is
// correct across context lengths — window layers CAP at the window size, recurrent layers
// are O(1) (constant) in context, and full layers grow linearly.
func TestKVGroupBudgetArithmetic(t *testing.T) {
	cfg := hybridKVGroupCfg()
	stride := cfg.NumKVHeads * cfg.HeadDim // 16
	const window = 4

	// Recurrent state is O(1): recurrent matrix (nV*kHd*vHd = 2*4*4 = 32) + conv window
	// ((K-1)*convDim). convDim = 2*keyDim + valDim = 2*(2*4) + (2*4) = 24; (3-1)*24 = 48.
	wantRecurrent := 32 + 48
	if got := cfg.recurrentStateFloats(); got != wantRecurrent {
		t.Fatalf("recurrentStateFloats = %d, want %d", got, wantRecurrent)
	}

	for _, ctx := range []int{0, 1, 2, 4, 8, 16, 64, 256} {
		b := cfg.KVGroupBudgetAt(ctx)

		if b.FullLayers != 1 || b.WindowLayers != 1 || b.RecurrentLayers != 1 {
			t.Fatalf("ctx=%d group layer counts = full %d / window %d / recurrent %d, want 1/1/1",
				ctx, b.FullLayers, b.WindowLayers, b.RecurrentLayers)
		}

		// Full-attention layer: grows linearly with ctx (ctx positions * stride * 3 planes).
		wantFull := ctx * stride * kvGroupPlanes
		if b.FullFloats != wantFull {
			t.Fatalf("ctx=%d full floats = %d, want %d (linear in ctx)", ctx, b.FullFloats, wantFull)
		}

		// Sliding-window layer: caps at the window size once ctx exceeds it.
		residentWindow := ctx
		if residentWindow > window {
			residentWindow = window
		}
		wantWindow := residentWindow * stride * kvGroupPlanes
		if b.WindowFloats != wantWindow {
			t.Fatalf("ctx=%d window floats = %d, want %d (cap at window=%d)", ctx, b.WindowFloats, wantWindow, window)
		}

		// Recurrent layer: constant in ctx.
		if b.RecurrentFloats != wantRecurrent {
			t.Fatalf("ctx=%d recurrent floats = %d, want %d (O(1) in ctx)", ctx, b.RecurrentFloats, wantRecurrent)
		}
	}

	// The window group must literally stop growing past the window; the full group must not.
	small, large := cfg.KVGroupBudgetAt(window), cfg.KVGroupBudgetAt(window*100)
	if small.WindowFloats != large.WindowFloats {
		t.Fatalf("window floats grew past window: %d -> %d", small.WindowFloats, large.WindowFloats)
	}
	if large.FullFloats <= small.FullFloats {
		t.Fatalf("full floats did not grow with ctx: %d -> %d", small.FullFloats, large.FullFloats)
	}
	if small.RecurrentFloats != large.RecurrentFloats {
		t.Fatalf("recurrent floats changed with ctx: %d -> %d", small.RecurrentFloats, large.RecurrentFloats)
	}
}

// TestKVGroupBudgetBeatsUniform witnesses the memory-plane payoff direction: at a context
// well past the window, the grouped budget is strictly smaller than the uniform allocation
// (window layers shed history they can't attend to; recurrent layers shed spans they never
// had). The resident-bytes benchmark on a real Qwen3.6-GDN model is R3 and out of scope; this
// asserts only the arithmetic's direction on the synthetic hybrid.
func TestKVGroupBudgetBeatsUniform(t *testing.T) {
	cfg := hybridKVGroupCfg()
	const ctx = 1024
	grouped := cfg.KVGroupBudgetAt(ctx).TotalFloats()
	uniform := cfg.UniformKVFloats(ctx)
	if grouped >= uniform {
		t.Fatalf("grouped budget %d not smaller than uniform %d at ctx=%d", grouped, uniform, ctx)
	}
	// The full layer alone must match its uniform share (grouped changes nothing for it).
	stride := cfg.NumKVHeads * cfg.HeadDim
	wantFullShare := ctx * stride * kvGroupPlanes
	if got := cfg.KVGroupBudgetAt(ctx).FullFloats; got != wantFullShare {
		t.Fatalf("full-layer grouped share = %d, want uniform share %d", got, wantFullShare)
	}
}

// TestResidentKVFloatsGateDefaultOff proves the plane is gated: with FAK_HYBRID_KV unset the
// seam reports the uniform allocation (today's behavior, unchanged); with it set to 1 the
// seam reports the smaller grouped total. This is the flag the #2241 R1 done-condition names.
func TestResidentKVFloatsGateDefaultOff(t *testing.T) {
	cfg := hybridKVGroupCfg()
	const ctx = 512

	if HybridKVGroupsEnabled() {
		t.Fatal("hybrid KV groups must default OFF")
	}
	if got, want := cfg.ResidentKVFloats(ctx), cfg.UniformKVFloats(ctx); got != want {
		t.Fatalf("gate off: ResidentKVFloats = %d, want uniform %d", got, want)
	}

	t.Setenv("FAK_HYBRID_KV", "1")
	if !HybridKVGroupsEnabled() {
		t.Fatal("FAK_HYBRID_KV=1 must enable hybrid KV groups")
	}
	if got, want := cfg.ResidentKVFloats(ctx), cfg.KVGroupBudgetAt(ctx).TotalFloats(); got != want {
		t.Fatalf("gate on: ResidentKVFloats = %d, want grouped %d", got, want)
	}
	if cfg.ResidentKVFloats(ctx) >= cfg.UniformKVFloats(ctx) {
		t.Fatalf("gate on: grouped %d not smaller than uniform %d", cfg.ResidentKVFloats(ctx), cfg.UniformKVFloats(ctx))
	}
}

// TestGroupBudgetBlocks witnesses the PagedKVPool-facing block arithmetic: token-indexed
// groups cost ceil(positions/blockTokens) blocks per layer, and the recurrent group costs 0
// blocks (its O(1) state is not paged into the token-block pool).
func TestGroupBudgetBlocks(t *testing.T) {
	cfg := hybridKVGroupCfg()
	const blockTokens = 4
	pool := NewPagedKVPool(cfg, blockTokens)

	// ctx=10: full layer needs ceil(10/4)=3 blocks; window layer caps at 4 -> ceil(4/4)=1
	// block; recurrent layer 0 blocks.
	full, window, recurrent := pool.GroupBudgetBlocks(cfg, 10)
	if full != 3 || window != 1 || recurrent != 0 {
		t.Fatalf("GroupBudgetBlocks(10) = full %d / window %d / recurrent %d, want 3/1/0", full, window, recurrent)
	}

	// Past the window, the window group's block demand stays flat while full keeps climbing.
	full1, window1, _ := pool.GroupBudgetBlocks(cfg, 4)
	full2, window2, _ := pool.GroupBudgetBlocks(cfg, 400)
	if window1 != window2 {
		t.Fatalf("window block demand grew past window: %d -> %d", window1, window2)
	}
	if full2 <= full1 {
		t.Fatalf("full block demand did not grow with ctx: %d -> %d", full1, full2)
	}
}
