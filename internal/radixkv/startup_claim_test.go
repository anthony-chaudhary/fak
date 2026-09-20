package radixkv

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// startup_claim_test.go — witnesses for fak#13350 (agent cache-warming CW-05). The named
// acceptance witness is TestStartupCacheClaim; the sibling tests decompose its three
// Definition-of-done checks (quota refusal keeps demand intact; expiry/invalidation makes a
// handle stale; concurrent claim/use/release is safe with no permanent pin).

// startupLeaseRefs sums every live lease (refs) across the tree's namespaces. A claim
// takes exactly one lease and Release/Done returns it, so this is 0 after all claims are
// released — the "no permanent pin" witness.
func startupLeaseRefs(t *Tree) int {
	total := 0
	var walk func(n *node)
	walk = func(n *node) {
		total += n.refs
		for _, c := range n.children {
			walk(c)
		}
	}
	t.forEachRoot(walk)
	return total
}

// smallStartupKV returns a KV cache whose OwnedPayloadBytes is a small, deterministic
// multiple of the populated row width, so a byte quota is easy to reason about.
func smallStartupKV(t *testing.T, width int) *model.KVCache {
	t.Helper()
	kv := model.NewKVCache(model.Config{NumLayers: 1})
	row := make([]float32, width)
	for i := range row {
		row[i] = float32(i)
	}
	kv.K[0] = row
	return kv
}

// TestStartupCacheClaim is the named witness. It exercises the full claim lifecycle on one
// boundary: acquire → validate → reuse-by-demand → release, and proves a claim is the
// CURRENT incarnation before release and stale after it.
func TestStartupCacheClaim(t *testing.T) {
	tree := New(0)
	sc := NewStartupCache(tree)
	kv := smallStartupKV(t, 8)
	prefix := []int{1, 2, 3, 4}

	claim, err := sc.AcquireStartup(StartupClaimConfig{SpareBytes: 1 << 20, SpareTokens: 1024}, prefix, kv)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if claim.Gen() == 0 || claim.Tokens() != len(prefix) {
		t.Fatalf("claim identity = gen %d tokens %d, want gen>0 tokens %d", claim.Gen(), claim.Tokens(), len(prefix))
	}
	if !sc.Validate(claim) {
		t.Fatal("fresh claim must validate")
	}
	// The claimed prefix is resident and usable by a demand path while leased.
	if matched := tree.MatchLenNS("", prefix); matched < len(prefix) {
		t.Fatalf("claimed prefix not resident: matched %d, want >= %d", matched, len(prefix))
	}
	// Release returns the prefix to demand residency; the handle is now stale.
	sc.Release(claim)
	if sc.Validate(claim) {
		t.Fatal("released claim must not validate")
	}
	// Idempotent: a second release must not double-decrement the lease.
	sc.Release(claim)
	if startupLeaseRefs(tree) != 0 {
		t.Fatal("tree lease invariant broken after double release")
	}
	if s := sc.Stats(); s.Claims != 1 || s.Released != 1 {
		t.Fatalf("claim stats = %+v, want 1 claim / 1 released", s)
	}
}

// TestStartupClaimQuotaRefusalKeepsDemandIntact proves an over-budget admission is refused
// without disturbing existing demand residency.
func TestStartupClaimQuotaRefusalKeepsDemandIntact(t *testing.T) {
	tree := New(0)
	// Seed a demand prefix so we can prove its residency is byte-identical after refusal.
	demandKV := smallStartupKV(t, 8)
	b, _ := tree.Lookup([]int{8, 9})
	leaf := tree.InsertCloneWithLogits(b, []int{8, 9}, demandKV, nil)
	tree.Done(leaf)
	before := tree.Stats().CPUCacheBytes
	if before == 0 {
		t.Fatal("demand seed is not resident")
	}

	sc := NewStartupCache(tree)
	// A token quota smaller than the requested prefix must refuse outright.
	_, err := sc.AcquireStartup(StartupClaimConfig{SpareTokens: 1}, []int{1, 2, 3, 4}, smallStartupKV(t, 8))
	if !errors.Is(err, ErrStartupClaimQuota) {
		t.Fatalf("over-token-quota acquire err = %v, want ErrStartupClaimQuota", err)
	}
	// A byte quota the claim cannot fit in must also refuse.
	_, err = sc.AcquireStartup(StartupClaimConfig{SpareBytes: 1}, []int{5, 6, 7}, smallStartupKV(t, 8))
	if !errors.Is(err, ErrStartupClaimQuota) {
		t.Fatalf("over-byte-quota acquire err = %v, want ErrStartupClaimQuota", err)
	}
	if after := tree.Stats().CPUCacheBytes; after != before {
		t.Fatalf("demand residency changed by refused claim: %d -> %d", before, after)
	}
	if s := sc.Stats(); s.Refused != 2 || s.Claims != 0 {
		t.Fatalf("refusal stats = %+v, want 2 refused / 0 claims", s)
	}
}

// TestStartupClaimExpiryInvalidatesHandle proves a TTL-expired claim's handle can no longer
// report warm readiness and its lease is reclaimed lazily (no timer).
func TestStartupClaimExpiryInvalidatesHandle(t *testing.T) {
	tree := New(0)
	sc := NewStartupCache(tree)
	claim, err := sc.AcquireStartup(StartupClaimConfig{SpareBytes: 1 << 20, TTL: time.Millisecond}, []int{1, 2, 3}, smallStartupKV(t, 4))
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	time.Sleep(5 * time.Millisecond)
	if sc.Validate(claim) {
		t.Fatal("expired claim must not validate")
	}
	if s := sc.Stats(); s.Expired != 1 {
		t.Fatalf("expiry stats = %+v, want 1 expired", s)
	}
	// A later acquisition sweeps the already-expired claim without re-leasing it.
	if _, err := sc.AcquireStartup(StartupClaimConfig{SpareBytes: 1 << 20}, []int{7, 8, 9}, smallStartupKV(t, 4)); err != nil {
		t.Fatalf("second acquire: %v", err)
	}
	if s := sc.Stats(); s.Expired != 1 {
		t.Fatalf("expired claim re-counted: %+v", s)
	}
}

// TestStartupClaimGenerationInvalidation proves a replacement warm over the same prefix
// mints a new generation, so the older handle stops validating.
func TestStartupClaimGenerationInvalidation(t *testing.T) {
	tree := New(0)
	sc := NewStartupCache(tree)
	cfg := StartupClaimConfig{SpareBytes: 1 << 20}
	first, err := sc.AcquireStartup(cfg, []int{1, 2, 3}, smallStartupKV(t, 4))
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if !sc.Validate(first) {
		t.Fatal("first claim must validate")
	}
	// Re-warm the same live prefix: the newer claim takes the node's generation, so the
	// older handle stops validating even though the node is untouched.
	second, err := sc.AcquireStartup(cfg, []int{1, 2, 3}, smallStartupKV(t, 4))
	if err != nil {
		t.Fatalf("second acquire: %v", err)
	}
	if sc.Validate(first) {
		t.Fatal("old generation must stay invalid after re-warm")
	}
	if !sc.Validate(second) {
		t.Fatal("new generation must validate")
	}
	// A policy eviction after both leases are returned detaches the node; the second
	// handle then no longer validates (the node is gone).
	sc.Release(first)
	sc.Release(second)
	tree.EvictPrefix([]int{1, 2, 3})
	if sc.Validate(second) {
		t.Fatal("handle must not validate after its node was evicted")
	}
}

// TestStartupClaimReleaseDoesNotStripNewerClaim proves an OLDER handle's release cannot
// strip a NEWER claim's lease over the same live node: releasing the predecessor is a
// generation-guarded no-op, so the current claim stays valid and leased.
func TestStartupClaimReleaseDoesNotStripNewerClaim(t *testing.T) {
	tree := New(0)
	sc := NewStartupCache(tree)
	cfg := StartupClaimConfig{SpareBytes: 1 << 20}

	first, err := sc.AcquireStartup(cfg, []int{1, 2, 3}, smallStartupKV(t, 4))
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	second, err := sc.AcquireStartup(cfg, []int{1, 2, 3}, smallStartupKV(t, 4))
	if err != nil {
		t.Fatalf("second acquire: %v", err)
	}
	// Releasing the older handle must return ONLY its own lease (refs 2 -> 1), leaving the
	// live second claim leased and valid.
	sc.Release(first)
	if second.Gen() == first.Gen() {
		t.Fatal("re-warm over the same prefix must mint a new generation")
	}
	if !sc.Validate(second) {
		t.Fatal("releasing an older handle stripped the live claim")
	}
	if got := startupLeaseRefs(tree); got != 1 {
		t.Fatalf("lease refs after older-handle release = %d, want 1 (second claim's lease)", got)
	}
	sc.Release(second)
	if startupLeaseRefs(tree) != 0 {
		t.Fatal("lease not returned after final release")
	}
}

// TestStartupClaimConcurrentUseAndRelease proves concurrent claim/validate/release is safe
// and leaves no permanent pin: every claim is released and the node becomes evictable.
func TestStartupClaimConcurrentUseAndRelease(t *testing.T) {
	tree := New(0)
	sc := NewStartupCacheWithLocker(tree, &sync.Mutex{})
	kv := smallStartupKV(t, 4)

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			prefix := []int{100 + i, 200 + i}
			claim, err := sc.AcquireStartup(StartupClaimConfig{SpareBytes: 1 << 20}, prefix, kv)
			if err != nil {
				return
			}
			_ = sc.Validate(claim)
			sc.Release(claim)
			sc.Release(claim) // double release must be a no-op
		}(i)
	}
	wg.Wait()

	if startupLeaseRefs(tree) != 0 {
		t.Fatal("tree invariant broken after concurrent claim churn")
	}
	if s := sc.Stats(); s.Claims != s.Released {
		t.Fatalf("claims %d != released %d — permanent pin", s.Claims, s.Released)
	}
}
