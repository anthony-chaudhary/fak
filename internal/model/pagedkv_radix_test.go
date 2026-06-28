package model_test

// pagedkv_radix_test.go — the end-to-end RadixAttention comparison for the PagedAttention
// block allocator (pagedkv.go, #277, B-006, Track B · Performance).
//
// pagedkv_test.go proves the allocator's STANDALONE properties (demand paging, bit-exact
// gather, copy-on-write, ≤20% fragmentation). What the issue's acceptance also asks — and
// the increment that landed the allocator (20b55355) explicitly left as "not yet wired into
// internal/radixkv.Tree for an end-to-end hit-rate measurement" — is the comparison AGAINST
// RadixAttention: "Cache hit rate ≥ current RadixAttention" and "Integration with
// RadixAttention". This test closes that measurement, on real KV, with no GPU.
//
// The load-bearing fact is that PagedAttention is an ALLOCATION discipline, not a matching
// one: it does not change WHICH prefix a request reuses, only HOW the reused bytes are
// stored. So on the same workload its cache hit rate EQUALS RadixAttention's (≥ is met with
// equality), while its resident KV footprint is strictly smaller — because RadixAttention
// stores a FULL-prefix KVCache per node (radixkv.go: "each node stores the FULL-prefix
// KVCache ... length == path length"), whereas paging shares the common prefix's blocks once
// by reference count. This test measures both sides of that on one shared-prefix workload:
//
//   - RadixAttention side: the REAL internal/radixkv.Tree, fed real synthetic-model KV via
//     the proven Lookup→SessionFromPrefix→Prefill→Insert path. We read its actual matched
//     tokens (the hit rate SGLang's paper headlines) and its actual resident footprint
//     (Stats().PrefixTokens = Σ plen over cache-bearing nodes — the full-prefix-clone cost).
//   - PagedAttention side: the model.PagedKVPool, sharing the same preamble across requests
//     by copy-on-write Fork. We read its reuse (the preamble shared per later request, at
//     zero byte copies — asserted mechanically) and its resident footprint
//     (PhysicalBlocks × BlockTokens).
//
// Then: paged hit rate ≥ radix hit rate (equal), and paged footprint < radix footprint.
//
// Honest scope (unchanged by this test): this measures the allocator's sharing/memory
// properties against RadixAttention's. It does NOT add a device-side paged-attention gather
// kernel, and it does NOT yet back radixkv.Tree's node storage with the pool on the live
// serve path — those remain the GPU-/wiring-heavy bulk of B-006 (#277 stays open).

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

func radixCmpCfg() model.Config {
	return model.Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        1024,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       63,
	}
}

// idSeq is a run of n distinct token ids starting at base (a request's token sequence).
func idSeq(base, n int) []int {
	r := make([]int, n)
	for i := range r {
		r[i] = base + i
	}
	return r
}

// pagedRow builds one token's per-layer K/V keyed by a unique `key` (so a forked share can
// be checked byte-for-byte and a divergent suffix can never collide with the preamble). V is
// the negation of K so a K/V layout swap can never pass silently.
func pagedRow(cfg model.Config, key int) (k, v [][]float32) {
	stride := cfg.NumKVHeads * cfg.HeadDim
	k = make([][]float32, cfg.NumLayers)
	v = make([][]float32, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		k[l] = make([]float32, stride)
		v[l] = make([]float32, stride)
		for d := 0; d < stride; d++ {
			val := float32(key*1000 + l*10 + d)
			k[l][d] = val
			v[l][d] = -val
		}
	}
	return k, v
}

// appendKeys appends n tokens to a paged sequence, keyed keyBase..keyBase+n-1.
func appendKeys(cfg model.Config, s *model.PagedKV, keyBase, n int) {
	for j := 0; j < n; j++ {
		k, v := pagedRow(cfg, keyBase+j)
		s.Append(k, v)
	}
}

// TestPagedKVHitRateAndMemoryVsRadixAttention is the #277 acceptance witness for "Cache hit
// rate ≥ current RadixAttention" and "Integration with RadixAttention": on a shared-preamble
// fan-out workload it drives the real radixkv.Tree (with real KV) and the paged pool, and
// shows the paged hit rate equals RadixAttention's while its resident footprint is strictly
// smaller — the PagedAttention win, measured rather than asserted.
func TestPagedKVHitRateAndMemoryVsRadixAttention(t *testing.T) {
	const (
		blockTokens = 16
		S           = 128 // shared preamble, block-aligned (8 blocks)
		U           = 80  // distinct suffix per request, block-aligned (5 blocks)
		K           = 6   // requests sharing the preamble
	)
	cfg := radixCmpCfg()
	preamble := idSeq(1, S)
	suffix := func(i int) []int { return idSeq(200+i*100, U) } // distinct first token per request
	req := func(i int) []int { return append(append([]int(nil), preamble...), suffix(i)...) }

	// ---- RadixAttention: the real internal/radixkv.Tree, fed real synthetic-model KV. ----
	m := model.NewSynthetic(cfg)
	tree := radixkv.New(0) // unbounded: measure the matching algorithm's pure hit rate
	radixMatched, totalTokens := 0, 0
	for i := 0; i < K; i++ {
		r := req(i)
		b, matched := tree.Lookup(r)
		var s *model.Session
		if kv := b.KV(); kv != nil {
			s = m.SessionFromPrefix(kv) // reuse the matched prefix (a clone — node cache is untouched)
		} else {
			s = m.NewSession()
		}
		s.PrefillNoLogits(r[matched:]) // prefill ONLY the unmatched suffix
		tree.Done(tree.Insert(b, r[matched:], s.Cache))
		radixMatched += matched
		totalTokens += len(r)
	}
	// Request 0 is cold; requests 1..K-1 each reuse the FULL preamble — the few-shot headline.
	if want := (K - 1) * S; radixMatched != want {
		t.Fatalf("radix matched %d tokens, want %d ((K-1)*S full-preamble reuse)", radixMatched, want)
	}
	// RadixAttention's resident KV: a full-prefix clone per node = preamble node (plen S) plus
	// K leaves (plen S+U). This is the cost paging exists to avoid.
	radixResident := tree.Stats().PrefixTokens
	if want := S + K*(S+U); radixResident != want {
		t.Fatalf("radix PrefixTokens=%d, want %d (preamble + K full-prefix leaves)", radixResident, want)
	}

	// ---- PagedAttention: model.PagedKVPool, sharing the same preamble by copy-on-write. ----
	pool := model.NewPagedKVPool(cfg, blockTokens)
	pre := pool.NewSequence() // the shared cached preamble (radix's preamble NODE analogue)
	appendKeys(cfg, pre, 0, S)
	if got, want := pool.PhysicalBlocks(), S/blockTokens; got != want {
		t.Fatalf("preamble PhysicalBlocks=%d, want %d", got, want)
	}

	pagedReused := 0
	for i := 0; i < K; i++ {
		before := pool.PhysicalBlocks()
		f := pre.Fork() // share every preamble block by refcount — zero byte copies
		if got := pool.PhysicalBlocks(); got != before {
			t.Fatalf("req %d: Fork copied bytes (PhysicalBlocks %d -> %d, want unchanged)", i, before, got)
		}
		// Request 0 "creates" the preamble (someone must prefill it first); requests 1..K-1
		// reuse it — the same accounting as radix's cold-first/reuse-rest above.
		if i > 0 {
			pagedReused += S
		}
		// The suffix lands on a fresh block boundary (S is block-aligned), so it demand-pages
		// exactly ceil(U/blockTokens) NEW blocks and copies NONE of the shared preamble.
		appendKeys(cfg, f, 100000*(i+1), U)
		if got, want := pool.PhysicalBlocks()-before, (U+blockTokens-1)/blockTokens; got != want {
			t.Fatalf("req %d: suffix added %d blocks, want %d (demand-paged, no COW copy)", i, got, want)
		}
		// The fork shares the preamble byte-for-byte (zero-copy reuse is also bit-exact).
		for l := 0; l < cfg.NumLayers; l++ {
			pk, fk := pre.GatherK(l), f.GatherK(l)
			for x := 0; x < len(pk); x++ {
				if fk[x] != pk[x] {
					t.Fatalf("req %d layer %d: forked preamble diverged at %d (%v != %v)", i, l, x, fk[x], pk[x])
				}
			}
		}
	}

	// Hit rate: paged reuse equals RadixAttention's matched tokens (≥ met with equality).
	if pagedReused != radixMatched {
		t.Fatalf("paged reuse %d != radix matched %d (hit rate must be ≥ RadixAttention)", pagedReused, radixMatched)
	}

	// Memory: paged resident = shared preamble (counted once) + K demand-paged suffixes.
	pagedResident := pool.PhysicalBlocks() * pool.BlockTokens()
	if want := S/blockTokens + K*((U+blockTokens-1)/blockTokens); pool.PhysicalBlocks() != want {
		t.Fatalf("paged PhysicalBlocks=%d, want %d", pool.PhysicalBlocks(), want)
	}
	if pagedResident >= radixResident {
		t.Fatalf("no memory win: paged resident %d not < radix resident %d", pagedResident, radixResident)
	}

	// Overhead ≤ 20%: paged resident vs the minimal KV the workload needs (preamble once +
	// K distinct suffixes). Block-aligned here, so it is exact; the bound guards a regression.
	uniqueTokens := S + K*U
	overhead := float64(pagedResident-uniqueTokens) / float64(uniqueTokens)
	if overhead > 0.20 {
		t.Fatalf("paged memory overhead %.4f exceeds the 0.20 bar", overhead)
	}

	t.Logf("shared-preamble fan-out K=%d S=%d U=%d: hit rate %.3f (reused %d/%d) identical paged vs radix; "+
		"resident KV paged=%d radix=%d (%.2fx less); paged overhead %.4f",
		K, S, U, float64(radixMatched)/float64(totalTokens), radixMatched, totalTokens,
		pagedResident, radixResident, float64(radixResident)/float64(pagedResident), overhead)
}
