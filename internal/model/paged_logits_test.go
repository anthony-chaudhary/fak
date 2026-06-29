package model

import "testing"

// paged_logits_test.go — #34 acceptance-#2 witness: the paged-attention GATHER produces decode
// logits BIT-IDENTICAL to the contiguous path on the same prompt/decode (temp 0), proven on a
// deliberately NON-CONTIGUOUS physical page layout with no GPU.
//
// paged_evict_test.go proves logits parity AFTER a mid-span Evict (the #33 tension). This pins
// the more basic acceptance line directly: with no eviction, materializing a live cache straight
// out of the page-table gather (PagedKV.ToKVCache, paged_materialize.go) and decoding several
// tokens reproduces the contiguous path's logits exactly — so the page table is logits-faithful
// end to end, through real attention + the LM head, not merely byte-faithful on K/V.
//
// Self-contained on purpose (its own churn / ascending / mirror helpers): the sibling-edited
// paged_evict_test.go / paged_swap_test.go must not be able to break this gate.

func pagedLogitsCfg() Config {
	return Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        97,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		BlockTopology:    PreNorm,
	}
}

// pagedLogitsChurn allocates then frees alternating decoy blocks so the pool's LIFO free list
// hands the next sequence physically NON-CONTIGUOUS, out-of-logical-order block ids — making
// "physical position != logical position" real rather than an accident of a fresh pool.
func pagedLogitsChurn(pool *PagedKVPool) {
	var decoy []int
	for i := 0; i < 6; i++ {
		decoy = append(decoy, pool.alloc())
	}
	for i, id := range decoy {
		if i%2 == 0 {
			pool.release(id) // free the even ids; the LIFO free list pops them in reverse
		}
	}
}

func pagedLogitsTableAscending(table []int) bool {
	for i := 1; i < len(table); i++ {
		if table[i] < table[i-1] {
			return false
		}
	}
	return true
}

// pagedLogitsMirror copies a contiguous KVCache verbatim into a fresh 3-plane paged sequence,
// token by token (K, pre-RoPE Kraw, V), so the paged copy starts byte-identical to the
// contiguous one — the fair starting point for proving the gather is logits-faithful.
func pagedLogitsMirror(pool *PagedKVPool, c *KVCache) *PagedKV {
	nL, w := pool.nLayers, c.kvStride()
	s := pool.NewSequence()
	for pos := 0; pos < c.Len(); pos++ {
		k := make([][]float32, nL)
		kraw := make([][]float32, nL)
		v := make([][]float32, nL)
		for l := 0; l < nL; l++ {
			k[l] = append([]float32(nil), c.K[l][pos*w:(pos+1)*w]...)
			kraw[l] = append([]float32(nil), c.Kraw[l][pos*w:(pos+1)*w]...)
			v[l] = append([]float32(nil), c.V[l][pos*w:(pos+1)*w]...)
		}
		s.AppendRaw(k, kraw, v)
	}
	return s
}

// TestPagedGatherLogitsBitIdenticalToContiguous is acceptance #2: a Session whose cache was
// MATERIALIZED from a non-contiguous paged gather decodes the same prompt to logits that are
// byte-for-byte identical to the contiguous-cache Session, at every step.
func TestPagedGatherLogitsBitIdenticalToContiguous(t *testing.T) {
	cfg := pagedLogitsCfg()
	m := NewSynthetic(cfg)

	prompt := []int{3, 17, 5, 23, 41, 2, 19, 8, 11} // 9 tokens -> 3 paged blocks at blockTokens=4

	// Contiguous reference: prefill the prompt; the return is the first decode token's logits.
	cs := m.NewSession()
	wantFirst := cs.Prefill(prompt)

	// Mirror the contiguous cache into a NON-CONTIGUOUS paged layout, then materialize a fresh
	// live cache straight out of the page-table gather (the operation acceptance #2 measures).
	pool := NewPagedKVPoolWithRaw(cfg, 4)
	pagedLogitsChurn(pool)
	seq := pagedLogitsMirror(pool, cs.Cache)
	if pagedLogitsTableAscending(seq.table) {
		t.Fatalf("weak setup: page table %v is physically contiguous; the gather must cross non-contiguous blocks", seq.table)
	}

	// Sanity: the gather is byte-faithful to the contiguous cache before any decode.
	for l := 0; l < cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, "gather K l"+itoa(l), cs.Cache.K[l], seq.GatherK(l))
		assertFloat32BitsEqual(t, "gather V l"+itoa(l), cs.Cache.V[l], seq.GatherV(l))
		assertFloat32BitsEqual(t, "gather Kraw l"+itoa(l), cs.Cache.Kraw[l], seq.GatherKraw(l))
	}

	ps := &Session{M: m, Cache: seq.ToKVCache(cfg)}
	if ps.Cache.Len() != cs.Cache.Len() {
		t.Fatalf("materialized len %d != contiguous len %d", ps.Cache.Len(), cs.Cache.Len())
	}

	// Decode several tokens in lockstep. Greedy argmax keeps both sessions on the same token
	// stream (temp 0); each step's logits must be bit-identical, and any gather/materialization
	// defect that perturbed the starting KV would surface as a non-zero bit diff here.
	tok := argmaxF32(wantFirst)
	for step := 0; step < 6; step++ {
		want := cs.Step(tok)
		got := ps.Step(tok)
		assertFloat32BitsEqual(t, "decode logits step "+itoa(step), want, got)
		tok = argmaxF32(want)
	}
}
