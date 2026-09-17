package agent

// inKernel_host_snapshot_reuse.go — host-session prefix reuse for recurrent hybrids.
//
// The defect this closes. On the Apple-silicon Qwen3.8-27B hybrid path the planner runs
// as a CPU session with Metal projections (p.backend == nil, p.metal == true): Metal is a
// CPU-session seam, not a compute.Backend (see executionIdentity). KV-prefix reuse on that
// path used the legacy token-id KV-clone lookup (p.tree.Lookup -> b.KV()), whose mid-edge
// split only gets a boundary KV when the model supports span eviction:
//
//	radixkv.go:624  if child.kv != nil && child.kv.CanEvict() == nil && ... { mid.kv = truncatePrefix(...) }
//
// A Qwen3.5/3.6 hybrid Gated-DeltaNet cache has an ACCUMULATED recurrent state and returns
// *RecurrentEvictUnsupportedError from CanEvict, so the split boundary carries NO KV. The
// lookup still observes the structural token prefix (cacheable = b.Plen() > 0) but has no
// restorable state, so the realized reuse is 0 — the live `cacheable=376tok reused=0tok`
// signature on the Mac. Only an EXACT full-prompt re-hit reused, because that needs no split.
//
// The fix. A hybrid KVCache.Clone DOES carry the complete recurrent+conv state (kvcache.go
// Clone -> linear: c.linear.clone()), so a *complete* state at a stable boundary is a valid
// prefix. The planner already materializes exactly that on the device path: it prefills to a
// 64-token adaptive block boundary, captures s.PrefixSnapshot(), and admits it with
// InsertSnapshot (inkernel_decode.go, inKernelAdaptiveSnapshotCheckpoint). On a backend-nil
// session PrefixSnapshot captures Cache.Clone() — which includes linear — and Restore sets
// s.Cache, so the same machinery is complete for a host hybrid. It was gated behind
// `p.backend != nil`, so the host Metal path never used it.
//
// This helper is the one predicate for "the host-session recurrent-hybrid path, where the
// snapshot tier is the ONLY restorable boundary". It is deliberately narrow: an ordinary
// (evictable) cache keeps the historical KV-clone split, byte-for-byte.

func inKernelHostSnapshotReuse(p *InKernelPlanner) bool {
	return p != nil && p.backend == nil && p.tree != nil &&
		p.m != nil && p.m.Cfg.IsQwen35Hybrid()
}
