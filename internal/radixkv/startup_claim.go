package radixkv

import (
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// startup_claim.go — STARTUP KV CACHE CLAIMS (fak#13350, agent cache-warming CW-05).
//
// WarmInsert (prewarm.go) lands a byte-known prefix at the LOWEST eviction priority and
// takes NO lease, so an opportunistic warm is the first LRU victim. That is exactly right
// for a mid-session tool-latency prefetch, and exactly wrong for a STARTUP warm: the whole
// point is that the prepared native cache state SURVIVES until the first real request
// arrives. Between the warm and that request the tree may be scanned, budget-pressured, or
// simply idle — and a bare WarmInsert offers no guarantee the prepared prefix is still
// there when demand finally looks.
//
// This file adds the bounded, expiring CLAIM that closes that gap without inventing a
// second cache and without pinning an unbounded prefix:
//
//   - A claim is a LEASE (refs++) on a warm leaf, so evictToBudget cannot reclaim it (a
//     node with refs>0 is never a victim) and cannot reclaim its ancestors either.
//   - The lease is FINITE: it covers only the claimed prefix, is capped by a byte/token
//     quota that may draw only on SPARE capacity (never displacing demand residency), and
//     EXPIRES after a caller-set TTL.
//   - Expiry is LAZY. No autonomous timer mutates the lock-free Tree: an expired claim is
//     reclaimed the next time a caller validates, releases, or acquires through the same
//     serialized boundary. This is the "no autonomous timer may mutate the Tree" contract
//     the issue states — the tree stays lock-free and mutation happens only on a caller's
//     own serialized turn.
//
// IDENTITY/GENERATION. A claim handle carries the node it leased plus a monotonic
// incarnation stamp (claimSeq, the sibling of recordSeq #12848). If the claimed leaf is
// evicted, detached, replaced by a later warm, revoked, or the whole tree is churned, the
// old handle no longer validates — Validate atomically re-checks attachment and the stamp
// under the caller's lock, so a stale handle can never report warm readiness.
//
// WHAT THIS IS NOT. It is not a second cache (claims hang off the SAME tree nodes), not a
// hold on demand residency (acquisition refuses rather than evict a demand leaf), and not a
// physical cache-hit claim. It proves a prepared prefix is still resident and validates as
// the SAME incarnation; the byte/token/expiry accounting is host-free SW logic. The
// wall-clock TTFT benefit of starting a real request from a claimed prefix remains
// host-gated and is deliberately not simulated here.

// ErrStartupClaimQuota is returned when admitting a startup claim would exceed the
// claim's finite spare-capacity byte/token quota. The claim is refused; demand residency
// is left fully intact.
var ErrStartupClaimQuota = errorString("radixkv: startup claim exceeds spare-capacity quota")

// ErrStartupClaimExpired is returned by Validate/Use on a claim whose TTL has elapsed but
// whose lease has not yet been reclaimed (lazy expiry). Callers must treat it as a miss.
var ErrStartupClaimExpired = errorString("radixkv: startup claim expired")

type errorString string

func (e errorString) Error() string { return string(e) }

// StartupClaimConfig bounds one startup claim. SpareBytes/SpareTokens draw ONLY on capacity
// that is not already holding demand residency; TTL bounds how long the claim may pin. A
// zero TTL means the claim never expires on its own (the caller promises an explicit
// Release); a zero SpareBytes/SpareTokens means that axis is unbounded.
type StartupClaimConfig struct {
	SpareBytes  int64
	SpareTokens int
	TTL         time.Duration
}

// StartupClaim is a bounded, expiring lease over a prepared startup prefix. It is a value
// handle: copying it copies the reference, and Release is idempotent across copies because
// the released flag lives on the shared claim state.
type StartupClaim struct {
	state *startupClaimState
}

// startupClaimState is the shared, caller-serialized claim record. All mutation happens
// under the owning StartupCache's lock, never autonomously.
type startupClaimState struct {
	tree      *Tree
	node      *node
	gen       uint64
	tokens    int
	bytes     int64
	expiresAt time.Time
	hasExpiry bool
	released  bool
}

// Gen returns the monotonic incarnation stamp this claim validated against. Two claims
// over the same token prefix from different warms carry different generations, so a handle
// from the older warm validates false after the newer warm replaces it.
func (c StartupClaim) Gen() uint64 {
	if c.state == nil {
		return 0
	}
	return c.state.gen
}

// Tokens reports the claimed prefix length in tokens (0 for the zero claim).
func (c StartupClaim) Tokens() int {
	if c.state == nil {
		return 0
	}
	return c.state.tokens
}

// Bytes reports the claimed prefix's resident payload bytes.
func (c StartupClaim) Bytes() int64 {
	if c.state == nil {
		return 0
	}
	return c.state.bytes
}

// ExpiresAt reports the claim's deadline and whether one is set.
func (c StartupClaim) ExpiresAt() (time.Time, bool) {
	if c.state == nil || !c.state.hasExpiry {
		return time.Time{}, false
	}
	return c.state.expiresAt, true
}

// StartupCache is the externally serialized boundary over one Tree for startup claims.
// It shares a locker with every other access path over the tree (the same ScopedTree
// discipline: "the mutex also supplies Tree's required external serialization"), so claim
// acquisition, validation, release, and lazy expiry reclamation never race a demand path.
type StartupCache struct {
	lock sync.Locker
	tree *Tree

	// claimSeq mints monotonic, never-reused incarnation stamps. It advances only on a
	// successful acquisition, so two claims from different warms are distinguishable.
	claimSeq uint64

	// gens maps a live claimed node to its current claim generation: the seam that makes
	// "is this handle still the CURRENT incarnation of this node?" O(1). claimsByGen is
	// the bounded inverse index the lazy-expiry sweep walks (sized by live claims, never
	// by tree size). Both are maintained only under the boundary lock.
	gens        map[*node]uint64
	claimsByGen map[uint64]*startupClaimState

	claims         int
	releasedClaims int
	expiredClaims  int
	refusedClaims  int
}

// NewStartupCache wraps tree with its own mutex. Callers that already serialize the tree
// elsewhere must use NewStartupCacheWithLocker instead so both paths share one lock.
func NewStartupCache(tree *Tree) *StartupCache {
	return NewStartupCacheWithLocker(tree, &sync.Mutex{})
}

// NewStartupCacheWithLocker shares serialization with other access paths over tree.
func NewStartupCacheWithLocker(tree *Tree, lock sync.Locker) *StartupCache {
	if tree == nil {
		tree = New(0)
	}
	if lock == nil {
		lock = &sync.Mutex{}
	}
	return &StartupCache{lock: lock, tree: tree}
}

// StartupClaimStats is a snapshot of this boundary's claim accounting.
type StartupClaimStats struct {
	Claims   int
	Released int
	Expired  int
	Refused  int
}

// Stats returns the boundary's claim accounting (read under the lock).
func (s *StartupCache) Stats() StartupClaimStats {
	if s == nil {
		return StartupClaimStats{}
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	return StartupClaimStats{Claims: s.claims, Released: s.releasedClaims, Expired: s.expiredClaims, Refused: s.refusedClaims}
}

// AcquireStartup warms the byte-known prefix `tokens` and takes a bounded, expiring lease
// on it. It reuses WarmInsert's lowest-priority placement (so a claim can never displace a
// demand prefix to place itself), then upgrades the resulting leaf to a leased claim.
//
// The quota draws only on SPARE capacity: acquisition computes the tree's current demand
// residency and refuses if the claim would push total residency past the configured spare
// byte/token allowance, or if the prefix exceeds the quota outright. A refused acquisition
// leaves demand residency byte-identical and takes no lease.
//
// Returns a zero claim (with an error) on quota refusal. A prefix already cached by demand
// (or an earlier warm) is still claimable: the lease upgrades the existing leaf, so a
// repeat startup warm never double-counts tokens.
func (s *StartupCache) AcquireStartup(cfg StartupClaimConfig, tokens []int, kv *model.KVCache) (StartupClaim, error) {
	if s == nil || len(tokens) == 0 {
		return StartupClaim{}, nil
	}
	s.lock.Lock()
	defer s.lock.Unlock()

	s.reclaimExpiredLocked()

	incomingTokens := len(tokens)
	if cfg.SpareTokens > 0 && incomingTokens > cfg.SpareTokens {
		s.refusedClaims++
		return StartupClaim{}, ErrStartupClaimQuota
	}
	if cfg.SpareBytes > 0 {
		incomingBytes := startupClaimBytes(kv, tokens)
		if incomingBytes > cfg.SpareBytes || s.residentBytesLocked()+incomingBytes > cfg.SpareBytes {
			s.refusedClaims++
			return StartupClaim{}, ErrStartupClaimQuota
		}
	}

	// WarmInsert applies the lowest-priority placement and its own budget pass, so a
	// claim into a saturated pool is dropped before it can cost a demand token. An empty
	// return means the whole prefix was already cached (by demand or an earlier warm) —
	// in that case the existing leaf is the claim target, not a new one.
	s.tree.WarmInsert(tokens, kv)
	leaf, matched := s.lookupLeafLocked(tokens)
	if leaf == nil || matched < incomingTokens {
		// The warm did not survive (saturated pool reclaimed it immediately): a startup
		// claim cannot be honored without displacing demand, so refuse rather than pin.
		s.refusedClaims++
		return StartupClaim{}, ErrStartupClaimQuota
	}
	if leaf.kv == nil && kv != nil {
		// The matched node is a structural intermediate that owns no reusable prefix
		// payload (the token sequence ended mid-edge). Leasing it would pin nothing a
		// request could actually reuse, so refuse honestly instead of claiming a warm
		// readiness the node cannot serve.
		s.refusedClaims++
		return StartupClaim{}, ErrStartupClaimQuota
	}

	leaf.refs++ // the claim's lease: evictToBudget skips a leased node and its ancestors
	s.claimSeq++
	st := &startupClaimState{
		tree:   s.tree,
		node:   leaf,
		gen:    s.claimSeq,
		tokens: leaf.plen,
		bytes:  startupClaimBytes(kv, tokens),
	}
	if cfg.TTL > 0 {
		st.hasExpiry = true
		st.expiresAt = time.Now().Add(cfg.TTL)
	}
	if s.gens == nil {
		s.gens = map[*node]uint64{}
		s.claimsByGen = map[uint64]*startupClaimState{}
	}
	s.gens[leaf] = st.gen
	s.claimsByGen[st.gen] = st
	s.claims++
	return StartupClaim{state: st}, nil
}

// Validate atomically reports whether this claim still names a live prepared prefix: the
// leaf is still attached to the tree, still carries this claim's incarnation, and has not
// expired. An expired or stale claim validates false. Read-only when valid; an expired
// claim's lease is reclaimed lazily here.
func (s *StartupCache) Validate(c StartupClaim) bool {
	if s == nil || c.state == nil {
		return false
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	return s.validateLocked(c.state)
}

func (s *StartupCache) validateLocked(st *startupClaimState) bool {
	if st.released || st.gen == 0 || st.gen > s.claimSeq {
		return false
	}
	if st.hasExpiry && !time.Now().Before(st.expiresAt) {
		s.expireLocked(st)
		return false
	}
	if st.node == nil || !s.tree.nodeAttached(st.node) {
		return false
	}
	// The incarnation must still be the CURRENT claim over this node: a later warm or
	// replacement mints a higher stamp, so an older handle can no longer validate.
	if s.currentGenOf(st.node) != st.gen {
		return false
	}
	return true
}

// currentGenOf returns the incarnation stamp of the live claim currently holding node,
// or 0 when node is unclaimed. The boundary keeps one map so a replaced claim is legible.
func (s *StartupCache) currentGenOf(n *node) uint64 {
	if s.gens == nil {
		return 0
	}
	return s.gens[n]
}

// Release drops the claim's lease exactly once, making the prepared prefix LRU-evictable
// again. It is idempotent: a second Release (or a Release after expiry) is a no-op and
// cannot double-decrement the node lease. Releasing does NOT evict the prefix — it only
// returns it to normal demand residency so a future request can still reuse it.
func (s *StartupCache) Release(c StartupClaim) {
	if s == nil || c.state == nil {
		return
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	s.releaseLocked(c.state)
}

func (s *StartupCache) releaseLocked(st *startupClaimState) {
	if st.released {
		return
	}
	st.released = true
	s.releasedClaims++
	s.dropClaimLocked(st)
}

// expireLocked reclaims an elapsed claim. Lazy expiry is the ONLY path that releases a
// claim without a caller Release, and it runs only inside a caller's serialized turn
// (Validate/Acquire/Release/Stats) — never from an autonomous timer, so the lock-free
// Tree is never mutated out-of-band.
func (s *StartupCache) expireLocked(st *startupClaimState) {
	if st.released {
		return
	}
	st.released = true
	s.expiredClaims++
	s.dropClaimLocked(st)
}

// dropClaimLocked performs the shared un-lease so the prefix returns to normal demand
// residency. The `released` flag (set by the caller before drop) makes this exactly-once,
// so neither a double Release nor a Release-after-expiry can double-decrement.
//
// The lease is PER-CLAIM, not per-node-generation: every AcquireStartup takes its own
// refs++ and every release must return exactly that one, even when several claims share a
// node. The generation guard governs VALIDITY (validateLocked), never lease return — so an
// older handle releases ITS lease without touching a newer claim's.
func (s *StartupCache) dropClaimLocked(st *startupClaimState) {
	if s.claimsByGen != nil {
		delete(s.claimsByGen, st.gen)
	}
	if s.gens != nil && st.node != nil && s.gens[st.node] == st.gen {
		// This claim is still the node's CURRENT generation — forget it so a later validate
		// reads a clean slate. A newer claim that already overwrote the entry keeps it.
		delete(s.gens, st.node)
	}
	if st.node != nil {
		s.tree.Done(st.node)
	}
}

// reclaimExpiredLocked sweeps every tracked claim for lazily-expired leases. It is bounded
// by the number of live claims (the gens map), not by tree size.
func (s *StartupCache) reclaimExpiredLocked() {
	now := time.Now()
	for n, gen := range s.gens {
		if st := s.claimsByGen[gen]; st != nil && st.hasExpiry && !now.Before(st.expiresAt) && st.node == n {
			s.expireLocked(st)
		}
	}
}

// residentBytesLocked reports current demand residency in payload bytes: the same
// OwnedPayloadBytes accounting the CPU cache budget uses, summed across every namespace.
func (s *StartupCache) residentBytesLocked() int64 {
	return s.tree.cpuCacheBytes()
}

// lookupLeafLocked walks the tree for tokens and returns the deepest node whose plen is
// exactly the matched prefix length — the leaf a WarmInsert would have attached.
func (s *StartupCache) lookupLeafLocked(tokens []int) (*node, int) {
	boundary, matched := s.tree.boundaryFor(s.tree.rootFor(""), tokens)
	if boundary == nil {
		return nil, matched
	}
	return boundary, matched
}

// startupClaimBytes sizes a claim's resident payload: the model KV payload when present,
// else the token-count fallback (4 bytes/token) so an accounting-mode claim (nil kv) still
// carries a finite, non-zero byte footprint.
func startupClaimBytes(kv *model.KVCache, tokens []int) int64 {
	if kv != nil {
		if b := kv.OwnedPayloadBytes(); b > 0 {
			return b
		}
	}
	return int64(len(tokens)) * 4
}
