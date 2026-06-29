package model

import "fmt"

// paged_materialize.go — materialize a live contiguous *KVCache straight out of the
// paged/block KV allocator's page-table gather (pagedkv.go; paged_evict.go Kraw plane).
// This is the "paged -> live cache" bridge paired with RestoreFromHost (paged_swap.go):
// it lets a swap-restored sequence resume on the proven f32 decode path (kvcache.go /
// kv.go) with byte-for-byte the cache a contiguous prefill of the same tokens would hold.
//
// Scope (honest, file:line):
//   - [SHIPPED] paged_materialize.go ToKVCache: gather K/V (+ Kraw on a 3-plane pool)
//     into a fresh KVCache, relabel pos to logical index, and resume decode through
//     Session; paged_swap_test.go proves the restored cache drives bit-identical logits.
//   - [GAP] This helper MATERIALIZES the gather into a fresh contiguous cache (a copy). It is
//     NOT a device-side paged-attention kernel that reads non-contiguous physical pages without
//     copying; device-native paged gather remains a backend follow-on.
//   - [GAP] Dense softmax K/V/Kraw planes only. A hybrid Gated-DeltaNet recurrence, the GLM-DSA
//     index/state cache, and the MiniMax MSA key cache are not paged, so ToKVCache leaves those
//     sub-caches at their NewKVCache defaults — the same scope boundary the paged allocator and
//     paged_evict.go already draw.

// KVCacheToPaged snapshots a dense softmax KVCache into a fresh 3-plane paged sequence
// (K, V, Kraw) backed by pool. It is the production twin of the paged-evict witness setup:
// every logical token row is copied byte-for-byte into paged blocks so SwapToHost can move
// the scheduler victim's KV at page granularity.
//
// The helper is deliberately narrow. Hybrid recurrent, GLM-DSA, and MiniMax sparse-index
// caches carry extra state outside the dense K/V/Kraw rows, so snapshotting only those rows
// would fabricate a resumable cache. Those variants must use recompute until their paged
// state has a first-class representation.
func KVCacheToPaged(pool *PagedKVPool, c *KVCache) (*PagedKV, error) {
	if pool == nil {
		return nil, fmt.Errorf("model: cannot snapshot KVCache into nil PagedKVPool")
	}
	if c == nil {
		return nil, fmt.Errorf("model: cannot snapshot nil KVCache")
	}
	if !pool.SupportsRaw() {
		return nil, fmt.Errorf("model: KVCacheToPaged requires a Kraw-capable PagedKVPool")
	}
	if c.cfg.isGLMMoeDsa() || c.cfg.isMiniMaxSparseAttn() || c.cfg.IsQwen35Hybrid() {
		return nil, fmt.Errorf("model: KVCacheToPaged supports dense softmax KV only")
	}
	if pool.nLayers != c.cfg.NumLayers || pool.stride != c.kvStride() {
		return nil, fmt.Errorf("model: KVCacheToPaged geometry mismatch: pool layers=%d stride=%d, cache layers=%d stride=%d",
			pool.nLayers, pool.stride, c.cfg.NumLayers, c.kvStride())
	}
	nL, w := pool.nLayers, c.kvStride()
	seq := pool.NewSequence()
	for pos := 0; pos < c.Len(); pos++ {
		k := make([][]float32, nL)
		kraw := make([][]float32, nL)
		v := make([][]float32, nL)
		for l := 0; l < nL; l++ {
			start, end := pos*w, (pos+1)*w
			if len(c.K[l]) < end || len(c.Kraw[l]) < end || len(c.V[l]) < end {
				seq.Free()
				return nil, fmt.Errorf("model: KVCacheToPaged layer %d has short KV rows for position %d", l, pos)
			}
			k[l] = append([]float32(nil), c.K[l][start:end]...)
			kraw[l] = append([]float32(nil), c.Kraw[l][start:end]...)
			v[l] = append([]float32(nil), c.V[l][start:end]...)
		}
		seq.AppendRaw(k, kraw, v)
	}
	return seq, nil
}

// ToKVCache reconstructs a live, contiguous *KVCache holding exactly this paged sequence's K and
// V (and pre-RoPE Kraw, on a 3-plane NewPagedKVPoolWithRaw pool) in logical token order, with
// each entry's absolute position relabeled to its logical index. The result is byte-for-byte the
// dense KV a contiguous prefill of the same tokens would hold, so a Session built on it decodes
// bit-identically to the contiguous path.
//
// cfg must be the model config the cache was filled under; its layer geometry must match the
// pool the sequence was minted from. On a 2-plane K/V pool the returned cache carries no Kraw
// (it cannot later Evict — the same contract the pool itself has), but decode, which reads only
// K/V/pos, is unaffected.
func (s *PagedKV) ToKVCache(cfg Config) *KVCache {
	c := NewKVCache(cfg)
	nL := s.pool.nLayers
	for l := 0; l < nL && l < len(c.K); l++ {
		c.K[l] = s.GatherK(l)
		c.V[l] = s.GatherV(l)
		if s.pool.SupportsRaw() {
			c.Kraw[l] = s.GatherKraw(l)
		}
	}
	c.pos = make([]int, s.nTokens)
	for i := range c.pos {
		c.pos[i] = i
	}
	return c
}
