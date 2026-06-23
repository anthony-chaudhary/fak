package radixkv

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// ---- helpers ----

func seq(base, n int) []int {
	r := make([]int, n)
	for i := range r {
		r[i] = base + i
	}
	return r
}

func distinctReq(i, n int) []int { return seq(1000+i*1000, n) }

func cat(parts ...[]int) []int {
	var out []int
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func argmax(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

func maxAbsDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

func newSyntheticTiny() *model.Model {
	return model.NewSynthetic(model.Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        64,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       63,
	})
}

// servePure runs one request through the tree with NO model (accounting only), returning
// the matched prefix length and the leased leaf to Done.
func servePure(t *Tree, req []int) (matched int, leaf *node) {
	b, m := t.Lookup(req)
	return m, t.Insert(b, req[m:], nil)
}

// ---- accounting tests (the hit-rate metric SGLang reports — model-independent) ----

// TestFewShotHitRate is the shared-prefix fan-out workload: N questions all share one
// preamble; the radix tree must DISCOVER the preamble (splitting the first request's
// compressed edge) so every later question reuses it — exactly RadixAttention's headline
// few-shot case. The reuse here is automatic: nothing declared the shared prefix.
func TestFewShotHitRate(t *testing.T) {
	tree := New(0)
	const P, Q, N = 64, 8, 5
	preamble := seq(1, P)
	var totalTok, matchedTok int
	for i := 0; i < N; i++ {
		req := cat(preamble, seq(2000+i*100, Q))
		matchedTok += tree.MatchLen(req)
		totalTok += len(req)
		_, leaf := servePure(tree, req)
		tree.Done(leaf)
	}
	// request 0 matches nothing; requests 1..N-1 each reuse the FULL preamble.
	if want := (N - 1) * P; matchedTok != want {
		t.Fatalf("matched %d tokens, want %d (full preamble reused per later request)", matchedTok, want)
	}
	hit := float64(matchedTok) / float64(totalTok)
	wantHit := float64((N-1)*P) / float64(N*(P+Q))
	if math.Abs(hit-wantHit) > 1e-9 {
		t.Fatalf("hit rate %.4f, want %.4f", hit, wantHit)
	}
	if st := tree.Stats(); st.Splits != 1 {
		t.Errorf("splits=%d, want 1 (the preamble edge split once on the 2nd request)", st.Splits)
	}
	t.Logf("few-shot N=%d P=%d Q=%d: hit rate=%.3f (matched %d / %d tokens)", N, P, Q, hit, matchedTok, totalTok)
}

// TestMultiTurnChain is the chat workload: every turn's request is the previous context
// plus a new turn, so each turn must reuse the ENTIRE previous context — the case where
// RadixAttention approaches a 100% hit rate on the growing prefix.
func TestMultiTurnChain(t *testing.T) {
	tree := New(0)
	const T = 6
	ctx := seq(1, 10)
	var totalTok, matchedTok, prevLen int
	for turn := 0; turn < T; turn++ {
		if turn > 0 {
			ctx = cat(ctx, seq(100+turn*10, 5))
		}
		req := append([]int(nil), ctx...)
		m := tree.MatchLen(req)
		if turn > 0 && m != prevLen {
			t.Fatalf("turn %d reused %d tokens, want %d (the whole previous context)", turn, m, prevLen)
		}
		matchedTok += m
		totalTok += len(req)
		prevLen = len(req)
		_, leaf := servePure(tree, req)
		tree.Done(leaf)
	}
	t.Logf("multi-turn T=%d: hit rate=%.3f (matched %d / %d tokens)", T, float64(matchedTok)/float64(totalTok), matchedTok, totalTok)
}

// TestLRUEviction is RadixAttention's memory policy: under a token budget the
// least-recently-used leaf is evicted, hot prefixes survive.
func TestLRUEviction(t *testing.T) {
	tree := New(20) // budget: 20 tokens
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)
	for _, r := range [][]int{a, b, c} {
		_, leaf := servePure(tree, r)
		tree.Done(leaf)
	}
	if m := tree.MatchLen(a); m != 0 {
		t.Errorf("a (oldest) should be evicted, but matched %d", m)
	}
	if m := tree.MatchLen(b); m != len(b) {
		t.Errorf("b should survive, matched %d/%d", m, len(b))
	}
	if m := tree.MatchLen(c); m != len(c) {
		t.Errorf("c should survive, matched %d/%d", m, len(c))
	}
	if st := tree.Stats(); st.Evictions != 1 || st.Tokens != 20 {
		t.Errorf("evictions=%d tokens=%d, want 1 and 20", st.Evictions, st.Tokens)
	}
}

// TestLRURespectsLease proves a leased prefix (a request still in flight) is NOT evicted
// even when it is the LRU candidate — the ref-count guard RadixAttention needs so it never
// reclaims a prefix being served.
func TestLRURespectsLease(t *testing.T) {
	tree := New(20)
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	_, la := servePure(tree, a) // a stays LEASED (no Done) — request "in flight"
	_, lb := servePure(tree, b)
	tree.Done(lb)
	_, lc := servePure(tree, c) // tokens 30 > 20 → must evict, but a is locked
	tree.Done(lc)

	if m := tree.MatchLen(a); m != len(a) {
		t.Errorf("leased a should survive despite being oldest, matched %d/%d", m, len(a))
	}
	if m := tree.MatchLen(b); m != 0 {
		t.Errorf("b (oldest UNLOCKED) should be evicted, matched %d", m)
	}
	if m := tree.MatchLen(c); m != len(c) {
		t.Errorf("c should survive, matched %d/%d", m, len(c))
	}
	tree.Done(la)
}

// TestPolicyEvictNode is the fak differentiator: a specific cached prefix is removed
// because POLICY says so (a quarantine verdict), not because memory ran out — the
// capability an opportunistic LRU radix cache structurally cannot offer. The shared
// preamble that a benign sibling still uses is untouched.
func TestPolicyEvictNode(t *testing.T) {
	tree := New(0)
	pre := seq(1, 8)
	good := cat(pre, seq(50, 4))
	bad := cat(pre, seq(70, 4))

	_, lg := servePure(tree, good)
	tree.Done(lg)
	_, lb := servePure(tree, bad)
	tree.Done(lb)

	freed := tree.EvictNode(lb) // quarantine the 'bad' tool-result span
	if freed != 4 {
		t.Fatalf("policy-evicted %d tokens, want 4 (bad's unique tail only)", freed)
	}
	if m := tree.MatchLen(bad); m != len(pre) {
		t.Errorf("bad's unique tail should be gone; matched %d, want %d (shared preamble survives)", m, len(pre))
	}
	if m := tree.MatchLen(good); m != len(good) {
		t.Errorf("benign sibling must be intact, matched %d/%d", m, len(good))
	}
	if st := tree.Stats(); st.PolicyEvictions != 1 {
		t.Errorf("policyEvictions=%d, want 1", st.PolicyEvictions)
	}
}

// TestEvictPrefixByTokenPath is the verdict-driven eviction seam for a caller that
// holds the poisoned TOKEN SEQUENCE rather than a *node handle (the in-kernel planner,
// which keeps no node refs). It must drop exactly the branch that cached the poison and
// spare the benign sibling sharing the same preamble — the same governance EvictNode
// gives, addressed by token path.
func TestEvictPrefixByTokenPath(t *testing.T) {
	tree := New(0)
	pre := seq(1, 8)
	good := cat(pre, seq(50, 4))
	bad := cat(pre, seq(70, 4)) // shares `pre`, diverges into a poisoned tail

	_, lg := servePure(tree, good)
	tree.Done(lg)
	_, lb := servePure(tree, bad)
	tree.Done(lb)

	freed := tree.EvictPrefix(bad) // quarantine: caller has the poisoned tokens, not a node
	if freed != 4 {
		t.Fatalf("EvictPrefix freed %d tokens, want 4 (bad's unique tail only)", freed)
	}
	if m := tree.MatchLen(bad); m != len(pre) {
		t.Errorf("poisoned tail must be gone; matched %d, want %d (shared preamble survives)", m, len(pre))
	}
	if m := tree.MatchLen(good); m != len(good) {
		t.Errorf("benign sibling must be intact, matched %d/%d", m, len(good))
	}
	if st := tree.Stats(); st.PolicyEvictions != 1 {
		t.Errorf("policyEvictions=%d, want 1", st.PolicyEvictions)
	}
}

// TestEvictPrefixMidEdgeDivergence covers the compressed-trie case the planner hits when
// the poison diverges in the MIDDLE of a cached edge: passing tokens that match only
// partway into a leaf's run must still evict that whole (poisoned) branch.
func TestEvictPrefixMidEdgeDivergence(t *testing.T) {
	tree := New(0)
	full := cat(seq(1, 8), seq(50, 8)) // one cached leaf, no split yet
	_, lf := servePure(tree, full)
	tree.Done(lf)

	// A poison token path that shares the first 8 tokens then diverges MID-EDGE.
	poison := cat(seq(1, 8), seq(99, 1))
	freed := tree.EvictPrefix(poison)
	if freed != len(full) {
		t.Fatalf("mid-edge EvictPrefix freed %d, want %d (the whole cached branch)", freed, len(full))
	}
	if m := tree.MatchLen(full); m != 0 {
		t.Errorf("the poisoned branch must be fully gone, matched %d", m)
	}
}

// TestEvictPrefixSparesUncachedPoison is the guard that keeps a clean shared prefix
// intact: when the poison's continuation was NEVER cached (the cached path ends before
// `tokens` is consumed), EvictPrefix is a no-op rather than wrongly dropping the clean
// prefix the poison happens to extend.
func TestEvictPrefixSparesUncachedPoison(t *testing.T) {
	tree := New(0)
	clean := seq(1, 8)
	_, lc := servePure(tree, clean)
	tree.Done(lc)

	poison := cat(clean, seq(70, 4)) // clean is cached; the poisoned tail is not
	if freed := tree.EvictPrefix(poison); freed != 0 {
		t.Fatalf("EvictPrefix must not evict when the poison tail was never cached, freed %d", freed)
	}
	if m := tree.MatchLen(clean); m != len(clean) {
		t.Errorf("the clean cached prefix must survive, matched %d/%d", m, len(clean))
	}
}

// TestEvictPrefixEmptyAndCold: an empty token path, and a path nothing in the tree shares,
// are both no-ops (fail-open).
func TestEvictPrefixEmptyAndCold(t *testing.T) {
	tree := New(0)
	_, l := servePure(tree, seq(1, 8))
	tree.Done(l)
	if freed := tree.EvictPrefix(nil); freed != 0 {
		t.Errorf("empty EvictPrefix freed %d, want 0", freed)
	}
	if freed := tree.EvictPrefix(seq(900, 4)); freed != 0 {
		t.Errorf("cold-path EvictPrefix freed %d, want 0", freed)
	}
	if m := tree.MatchLen(seq(1, 8)); m != 8 {
		t.Errorf("unrelated entry must be untouched, matched %d/8", m)
	}
}

func TestNodeCacheEntryDescribesTokenPrefix(t *testing.T) {
	tree := New(0)
	req := []int{1, 2, 3, 5, 8}
	b, matched := tree.Lookup(req)
	if matched != 0 {
		t.Fatalf("cold request matched %d, want 0", matched)
	}
	leaf := tree.Insert(b, req, nil)
	tree.Done(leaf)

	e := leaf.CacheEntry("synthetic-model", "synthetic-tokenizer")
	if e.Plane != cachemeta.PlaneKVPrefix || e.ID.MediaType != cachemeta.MediaKVSpan {
		t.Fatalf("bad KV cache identity: %+v", e)
	}
	if e.ID.Length != int64(len(req)) || e.ID.Digest != cachemeta.DigestTokenIDs(req) {
		t.Fatalf("bad token-prefix id: %+v", e.ID)
	}
	if e.Derivation.ModelID != "synthetic-model" || e.Derivation.TokenizerID != "synthetic-tokenizer" {
		t.Fatalf("model/tokenizer axes missing: %+v", e.Derivation)
	}
	if e.Derivation.PositionMode != cachemeta.PositionPrefixAligned {
		t.Fatalf("KV prefix should be prefix-aligned, got %q", e.Derivation.PositionMode)
	}
}

// ---- live correctness rung: reuse-through-a-split == fresh recompute, bit for bit ----

// TestReuseThroughSplitMatchesRecompute is the load-bearing witness that radixkv's
// automatic prefix reuse is SOUND, including the hard case SGLang's radix tree exists for:
// two requests that diverge in the MIDDLE of a cached run, forcing an edge split. The split
// truncates the first request's KV cache to the shared-preamble boundary (via the proven
// KVCache.Evict of the tail); serving the second request from that truncated, cloned cache
// and prefilling only its suffix must produce logits BIT-IDENTICAL to a fresh full prefill.
// If the truncation drifted a single bit this fails — so it pins radixkv's only KV math to
// internal/model's verified core.
func TestReuseThroughSplitMatchesRecompute(t *testing.T) {
	m := newSyntheticTiny()
	tree := New(0)
	preamble := seq(3, 8) // shared run; tokens < vocab
	reqA := cat(preamble, seq(20, 3))
	reqB := cat(preamble, seq(40, 4)) // diverges from A inside the preamble's compressed edge

	// Serve A cold: full prefill, insert into the (empty) tree.
	bA, mA := tree.Lookup(reqA)
	if mA != 0 {
		t.Fatalf("first request matched %d, want 0 (cold cache)", mA)
	}
	sA := m.NewSession()
	sA.Prefill(reqA[mA:])
	tree.Done(tree.Insert(bA, reqA[mA:], sA.Cache))

	// Serve B: must reuse the preamble via an edge SPLIT, then prefill only B's suffix.
	bB, mB := tree.Lookup(reqB)
	if mB != len(preamble) {
		t.Fatalf("reqB matched %d, want %d (preamble reused via split)", mB, len(preamble))
	}
	if bB.KV() == nil || bB.KV().Len() != len(preamble) {
		t.Fatalf("split boundary KV len=%v, want %d", kvLen(bB.KV()), len(preamble))
	}
	sB := m.SessionFromPrefix(bB.KV()) // clone the truncated preamble cache
	lReuse := sB.Prefill(reqB[mB:])    // prefill ONLY the suffix
	tree.Done(tree.Insert(bB, reqB[mB:], sB.Cache))

	// Recompute B fresh, from scratch.
	full := m.NewSession()
	lFull := full.Prefill(reqB)

	if a, b := argmax(lReuse), argmax(lFull); a != b {
		t.Errorf("reuse-through-split argmax %d != recompute %d", a, b)
	}
	if d := maxAbsDiff(lReuse, lFull); d != 0 {
		t.Errorf("reuse-through-split not bit-identical to recompute (max|Δ|=%.3e)", d)
	}
	if sB.Cache.Len() != len(reqB) {
		t.Errorf("reuse cache len %d != %d", sB.Cache.Len(), len(reqB))
	}
	if st := tree.Stats(); st.Splits != 1 {
		t.Errorf("splits=%d, want 1", st.Splits)
	}
	t.Logf("reuse-through-split: reused %d/%d tokens (suffix %d prefilled), max|Δ|=0, bit-identical",
		mB, len(reqB), len(reqB)-mB)
}

func kvLen(c *model.KVCache) interface{} {
	if c == nil {
		return nil
	}
	return c.Len()
}
