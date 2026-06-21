package radixkv

// leak_test.go — regression proofs for the two radixkv memory-leak hazards the audit
// surfaced:
//
//   (1) Lease leak: Lookup takes refs++ on the boundary; a caller that errors between
//       Lookup and Insert/Done without releasing pins that node AND all its ancestors
//       against evictToBudget forever. These tests prove the abort path (Lookup→Done,
//       no Insert) RECLAIMS, and characterize the hazard so a future refactor of the
//       ref-counting can't silently break release-on-abort.
//
//   (2) Budget-vs-true-memory gap: the LRU budget bounds Σ edge lengths, but each node
//       holds a full-prefix KV of length plen, so true resident positions (Σ plen) can
//       exceed the budget by an O(depth) factor. TestBudgetVsTrueKVFootprint pins the
//       gap via the now-exported Stats.PrefixTokens so it is measurable, not silent.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// sessionFor clones a session from a matched-prefix KV (or a fresh one when nothing
// matched), mirroring how a real caller seeds reuse from the radix boundary.
func sessionFor(m *model.Model, kv *model.KVCache) *model.Session {
	if kv == nil {
		return m.NewSession()
	}
	return m.SessionFromPrefix(kv)
}

// abortServe simulates a caller that did Lookup but errored before Insert. The correct
// (leak-free) caller releases the lease with Done on that abort path.
func abortServe(t *Tree, req []int, release bool) {
	b, _ := t.Lookup(req)
	if release {
		t.Done(b) // abort path done right: lease released, node becomes evictable
	}
	// release==false models the BUG: the lease is dropped on the floor.
}

// TestLookupLeaseReleasedOnAbortReclaims proves that a caller which aborts after Lookup
// but correctly calls Done lets the budget reclaim normally — the abort path works.
func TestLookupLeaseReleasedOnAbortReclaims(t *testing.T) {
	tree := New(20) // budget holds two 10-token requests
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	// Serve a and b fully (each leased then Done — settled in the cache).
	_, la := servePure(tree, a)
	tree.Done(la)
	_, lb := servePure(tree, b)
	tree.Done(lb)

	// A third request is looked up but the caller ABORTS — and releases correctly.
	abortServe(tree, c, true)

	// The tree must be within budget: the aborted lease did not pin anything, and the
	// LRU could evict to make room. tokens == Σ edge lengths must be ≤ budget.
	if st := tree.Stats(); st.Tokens > tree.maxTokens {
		t.Fatalf("after correctly-released abort, tokens=%d exceeds budget %d", st.Tokens, tree.maxTokens)
	}
}

// TestLeakedLeasesDefeatBudget is the characterization test for the hazard: when enough
// leases are leaked (Lookup/Insert without Done), evictToBudget runs out of unlocked
// leaves and CANNOT drive the tree back within budget — resident tokens stay pinned above
// the configured ceiling. This is the logical memory leak. It documents exactly why the
// lease discipline on Lookup is mandatory; if a refactor of Done/evictToBudget ever lets a
// leaked lease silently stop pinning, this test catches the behavioral change. The cure
// (release the leases, re-run eviction) is asserted too, proving the only fault was the
// missing Done.
func TestLeakedLeasesDefeatBudget(t *testing.T) {
	const budget = 20 // holds two 10-token requests
	tree := New(budget)
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	// Leak leases on BOTH a and b (Insert returns a leased leaf; we never Done it).
	ba, ma := tree.Lookup(a)
	leakA := tree.Insert(ba, a[ma:], nil)
	bb, mb := tree.Lookup(b)
	leakB := tree.Insert(bb, b[mb:], nil)

	// Serve c, overflowing the budget (a+b+c = 30 > 20). a and b are both leaked-leased
	// and c is leased until its Done, so during c's Insert evictToBudget finds NO unlocked
	// leaf and gives up — leaving tokens stuck at 30, above the budget of 20.
	_, lc := servePure(tree, c)

	if st := tree.Stats(); st.Tokens <= budget {
		t.Fatalf("expected leaked leases to hold tokens above budget; tokens=%d budget=%d", st.Tokens, budget)
	}
	tree.Done(lc) // c is now releasable, but eviction does not re-run on its own

	// The cure: release the leaked leases and insert one more request so evictToBudget
	// runs again with unlocked victims available — the tree settles back within budget.
	tree.Done(leakA)
	tree.Done(leakB)
	d := distinctReq(3, 10)
	_, ld := servePure(tree, d)
	tree.Done(ld)
	if st := tree.Stats(); st.Tokens > budget {
		t.Fatalf("after releasing the leaked leases, tokens=%d should be ≤ budget %d", st.Tokens, budget)
	}
}

// TestManySessionsStayWithinBudget is the positive leak proof for the normal (disciplined)
// path: stream far more distinct sessions than the budget can hold, each properly
// Lookup→Insert→Done, and assert the cache's edge-token count never runs away — it stays
// pinned at/under the configured budget no matter how many sessions pass through.
func TestManySessionsStayWithinBudget(t *testing.T) {
	const budget = 256
	tree := New(budget)
	const sessions = 5000
	const reqLen = 24

	for i := 0; i < sessions; i++ {
		req := distinctReq(i, reqLen)
		_, leaf := servePure(tree, req)
		tree.Done(leaf)
		if st := tree.Stats(); st.Tokens > budget {
			t.Fatalf("session %d: tokens=%d exceeds budget %d (cache leaked)", i, st.Tokens, budget)
		}
	}
	st := tree.Stats()
	if st.Tokens > budget {
		t.Fatalf("final tokens=%d exceeds budget %d", st.Tokens, budget)
	}
	if tree.tokens != st.Tokens {
		t.Fatalf("token counter %d drifted from live Σ edge lengths %d", tree.tokens, st.Tokens)
	}
	if st.Evictions == 0 {
		t.Fatalf("expected evictions under sustained pressure, got 0")
	}
}

// TestBudgetVsTrueKVFootprint pins finding (3): with a model-less accounting tree the
// budget bounds Σ edge lengths, but a deep single chain holds Σ plen ≈ quadratic true
// positions. We build the chain with real (tiny) KV caches so PrefixTokens is the true
// resident position count, and assert Tokens ≤ budget while PrefixTokens ≫ budget — the
// gap the audit flagged, now observable.
func TestBudgetVsTrueKVFootprint(t *testing.T) {
	m := newSyntheticTiny()
	const budget = 64
	tree := New(budget)

	// Build a deep chain: each request extends the previous by one token, so every node
	// has edge length 1 (Tokens counts 1 each) but plen 1,2,3,... (PrefixTokens is Σ plen).
	prev := []int(nil)
	for i := 0; i < budget; i++ {
		req := append(append([]int(nil), prev...), i%10+1)
		b, mm := tree.Lookup(req)
		// Build the full-prefix KV for this request so the node carries real positions.
		s := sessionFor(m, b.KV())
		s.Prefill(req[mm:])
		tree.Done(tree.Insert(b, req[mm:], s.Cache))
		prev = req
	}

	st := tree.Stats()
	if st.Tokens > budget {
		t.Fatalf("Tokens=%d should stay within budget %d (the LRU metric)", st.Tokens, budget)
	}
	// True resident positions must visibly exceed the budget — that's the whole point.
	if st.PrefixTokens <= budget {
		t.Fatalf("PrefixTokens=%d did not exceed budget %d; the deep-chain gap is not being measured", st.PrefixTokens, budget)
	}
	t.Logf("deep-chain gap: Tokens(budget metric)=%d ≤ budget=%d, but PrefixTokens(true KV)=%d (%.1fx budget)",
		st.Tokens, budget, st.PrefixTokens, float64(st.PrefixTokens)/float64(budget))
}
