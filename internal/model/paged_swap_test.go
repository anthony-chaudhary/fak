package model

import "testing"

// paged_swap_test.go — #31 swap/recompute witnesses over the real PagedKV bytes.

func rebuildCacheFromPaged(cfg Config, seq *PagedKV) *KVCache {
	c := NewKVCache(cfg)
	for l := 0; l < cfg.NumLayers; l++ {
		c.K[l] = seq.GatherK(l)
		if seq.pool.SupportsRaw() {
			c.Kraw[l] = seq.GatherKraw(l)
		}
		c.V[l] = seq.GatherV(l)
	}
	c.pos = make([]int, seq.Len())
	for i := range c.pos {
		c.pos[i] = i
	}
	return c
}

func assertPagedEqualsCache(t *testing.T, tag string, seq *PagedKV, c *KVCache) {
	t.Helper()
	for l := 0; l < c.cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, tag+" K l"+itoa(l), c.K[l], seq.GatherK(l))
		if seq.pool.SupportsRaw() {
			assertFloat32BitsEqual(t, tag+" Kraw l"+itoa(l), c.Kraw[l], seq.GatherKraw(l))
		}
		assertFloat32BitsEqual(t, tag+" V l"+itoa(l), c.V[l], seq.GatherV(l))
	}
}

// TestPagedKVSwapToHostRestoreBitExact proves the #31 swap path over the real paged-KV
// representation: serialized host bytes survive after the source frees its pages, restore
// into fresh owned blocks, and resume decode with bit-identical next-token logits.
func TestPagedKVSwapToHostRestoreBitExact(t *testing.T) {
	cfg := pagedEvictCfg()
	m := NewSynthetic(cfg)
	prompt := []int{3, 17, 5, 23, 41, 2, 19}

	ref := m.NewSession()
	ref.Prefill(prompt)

	pool := NewPagedKVPoolWithRaw(cfg, 4)
	seq := snapshotCacheToPaged(pool, ref.Cache)
	blocks := seq.Blocks()
	blob, err := seq.SwapToHost()
	if err != nil {
		t.Fatalf("SwapToHost: %v", err)
	}
	if len(blob) == 0 {
		t.Fatal("SwapToHost returned an empty blob")
	}
	seq.Free()
	if got := pool.PhysicalBlocks(); got != 0 {
		t.Fatalf("source pages still resident after Free: PhysicalBlocks=%d, want 0", got)
	}

	restored, err := pool.RestoreFromHost(blob)
	if err != nil {
		t.Fatalf("RestoreFromHost: %v", err)
	}
	if restored.Len() != len(prompt) || restored.Blocks() != blocks {
		t.Fatalf("restored Len/Blocks = %d/%d, want %d/%d", restored.Len(), restored.Blocks(), len(prompt), blocks)
	}
	assertPagedEqualsCache(t, "restored", restored, ref.Cache)

	gotSess := &Session{M: m, Cache: rebuildCacheFromPaged(cfg, restored)}
	const next = 7
	wantLogits := ref.Step(next)
	gotLogits := gotSess.Step(next)
	assertFloat32BitsEqual(t, "restored next-token logits", wantLogits, gotLogits)
}

// TestPagedKVRecomputeAfterDropBitExact proves the #31 recompute path's correctness
// condition: dropping paged KV and re-prefilling the retained prompt re-derives the same
// K/Kraw/V state and logits as the never-preempted sequence.
func TestPagedKVRecomputeAfterDropBitExact(t *testing.T) {
	cfg := pagedEvictCfg()
	m := NewSynthetic(cfg)
	prompt := []int{11, 7, 29, 31, 43, 47}

	ref := m.NewSession()
	wantLogits := ref.Prefill(prompt)
	pool := NewPagedKVPoolWithRaw(cfg, 4)
	preempted := snapshotCacheToPaged(pool, ref.Cache)
	preempted.Free() // recompute mode drops the KV instead of holding host bytes
	if got := pool.PhysicalBlocks(); got != 0 {
		t.Fatalf("recompute preemption left KV pages resident: %d", got)
	}

	recomputed := m.NewSession()
	gotLogits := recomputed.Prefill(prompt)
	assertFloat32BitsEqual(t, "recomputed prefill logits", wantLogits, gotLogits)
	for l := 0; l < cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, "recomputed K l"+itoa(l), ref.Cache.K[l], recomputed.Cache.K[l])
		assertFloat32BitsEqual(t, "recomputed Kraw l"+itoa(l), ref.Cache.Kraw[l], recomputed.Cache.Kraw[l])
		assertFloat32BitsEqual(t, "recomputed V l"+itoa(l), ref.Cache.V[l], recomputed.Cache.V[l])
	}
}
