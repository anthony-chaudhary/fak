// Package radixkv is SGLang's RadixAttention prefix cache, rebuilt over fak's
// kernel-owned KV cache — the apples-to-apples answer to "how does fak compare to
// SGLang's KV-cache radix attention?".
//
// SGLang's RadixAttention (arXiv:2312.07104) is the SOTA mechanism for *automatic*
// KV-cache reuse across requests: it stores every request's KV in a radix tree
// (compressed trie) keyed by the token-id sequence, so a new request that shares a
// prefix with any prior one DISCOVERS the longest cached prefix by a tree walk and
// reuses its K/V — recomputing only its own suffix. Under memory pressure it evicts
// the least-recently-used LEAF (collapsing parents upward), keeping hot prefixes.
//
// fak already had the two HARD halves of this — a kernel-owned KVCache that can be
// CLONED to splice a known prefix into a new session (model.SessionFromPrefix) and
// EVICTED span-exactly (model.KVCache.Evict, proven == never-saw-it). What it lacked
// was the radix tree's *automatic discovery*: the existing reuse path (NewBatchFromPrefix)
// makes you DECLARE the shared prefix. RadixAttention's contribution is that you don't —
// the tree finds it, even when two requests diverge in the MIDDLE of a run (which forces
// an edge SPLIT). This package is that tree, and nothing more: it is a pure CONSUMER of
// the proven primitives (Clone, Prefill, Evict), the same posture internal/kvmmu takes —
// so the reuse it schedules is bit-identical to a full recompute (proven in the live test),
// because the only KV operations it performs are the ones internal/model already verified.
//
// Two governance modes on ONE tree, which is the whole point of doing this in fak:
//
//   - EvictToBudget — LRU eviction under a token budget. This is RadixAttention's policy
//     verbatim: opportunistic, cache-pressure-driven, hot prefixes survive. It is what we
//     benchmark head-to-head with SGLang (cache hit rate is the metric SGLang's own paper
//     headlines, and it is hardware- and model-INDEPENDENT — a function of (workload,
//     matching algorithm) only — so fak-on-CPU vs SGLang-on-GPU is a fair comparison on
//     exactly this axis, unlike raw tok/s).
//
//   - EvictNode — POLICY eviction of a named span regardless of LRU or recency. This is
//     the capability an LRU radix cache structurally cannot offer: a poison/quarantine
//     verdict (internal/ctxmmu) evicts a specific subtree from the cache, and because each
//     node's KV is a real model.KVCache, the session that cloned that prefix can Evict the
//     span bit-identically to never-having-seen it. Same primitive (prefix-addressable KV),
//     opposite governance — measured, not asserted.
//
// Faithfulness note (honest, per the repo's "labeled, not faked" rule): each node stores
// the FULL-prefix KVCache for its root→node path (length == path length), not SGLang's
// per-segment slabs in a paged pool. That costs more memory, but the metric this package
// exists to compare — cache hit rate / prefill-tokens-saved — is layout-independent, and
// storing full-prefix caches lets every reuse and every edge-split go through the *proven*
// Clone/Evict primitives unchanged (a split truncates a child's cache to the boundary via
// Evict-of-the-tail, which has no survivor to re-RoPE, so it is an exact prefix). We trade
// memory for the guarantee that not one bit of the verified KV core is re-derived here.
package radixkv

import (
	"context"
	"errors"
	"math"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var ErrSnapshotByteBudget = errors.New("radixkv: snapshot resident-byte budget exceeded")
var ErrHostSnapshotByteBudget = errors.New("radixkv: host snapshot resident-byte budget exceeded")

// SnapshotTier is the physical source that satisfied a complete-prefix lookup.
// The zero value is a miss; callers must never infer a hit without an owned
// PrefixSnapshot result.
type SnapshotTier string

const (
	SnapshotTierMiss     SnapshotTier = ""
	SnapshotTierDeviceL1 SnapshotTier = "device_l1"
	SnapshotTierHostL2   SnapshotTier = "host_dram_l2"
	SnapshotTierRemoteL3 SnapshotTier = "remote_http_l3"
)

// node is one vertex of the compressed radix tree. The edge parent→node carries `key`
// (a run of token ids); the path root→node spells the token prefix this node caches.
// children are keyed by the FIRST token of each child's edge, so each match step is an
// O(1) map lookup followed by a run compare.
type node struct {
	key      []int
	parent   *node
	children map[int]*node

	// kv is the kernel-owned KV cache for the FULL prefix root→node (every token on the
	// path, post-prefill). nil in pure-accounting mode (no model attached); set by Insert
	// to the cache the caller built, and by split to a truncated clone of a child's cache.
	kv       *model.KVCache
	snapshot *model.PrefixSnapshot // optional full device/hybrid prefix owner
	// hostSnapshot is an independently owned complete PrefixSnapshot image in
	// process-local host DRAM. It is not provider accounting: the image owns the
	// attention K/Kraw/V rows, positions, and hybrid recurrent state needed to
	// materialize a fresh backend snapshot after the hot owner is evicted.
	hostSnapshot *model.HostPrefixSnapshot
	// remoteSnapshot is a verified reference to a complete serialized
	// HostPrefixSnapshot in the configured l3kv/blobhttp tier. It owns no local
	// payload bytes and remains usable after both local physical copies are gone.
	remoteSnapshot *remoteSnapshotRef
	// logits is the final-token distribution for this exact full prefix, when the caller
	// provided it. It lets an exact-hit replay sample the first generated token without
	// refeeding the prompt's final token just to recover logits. Splits intentionally do
	// not derive logits for the shorter boundary; a full-prefix KV can be truncated, but
	// the final logits for an intermediate prefix are not available from a longer leaf.
	logits       []float32
	cachedLogits []float32 // logits owned by a complete device snapshot

	plen     int    // path length in tokens (parent.plen + len(key)); == len(kv) when kv!=nil
	refs     int    // active leases; a leaf with refs>0 is never LRU-evicted
	lastUsed uint64 // logical clock of the most recent match/insert touching this node — LRU key
	hits     int    // subsequent demand lookups that found this node resident
	chunkID  int    // physical backing page / allocation chunk identifier (0 = unassigned)
}

// ChunkID returns the physical backing page or allocation chunk identifier for this node.
func (n *node) ChunkID() int {
	if n == nil {
		return 0
	}
	return n.chunkID
}

// SetChunkID assigns the physical backing page or allocation chunk identifier for this node.
func (n *node) SetChunkID(id int) {
	if n != nil {
		n.chunkID = id
	}
}

// EvictionPolicy selects the budget-pressure victim rule. The zero value is pure LRU,
// preserving New(maxTokens)'s original behavior; cost-aware is opt-in while the remaining
// #2666 production wiring lands.
type EvictionPolicy uint8

const (
	EvictionLRU EvictionPolicy = iota
	EvictionCostAware
	EvictionPageAware
)

func (p EvictionPolicy) String() string {
	switch p {
	case EvictionCostAware:
		return "cost-aware"
	case EvictionPageAware:
		return "page-aware"
	default:
		return "lru"
	}
}

// Tree is a RadixAttention prefix cache: a radix tree of token sequences with
// longest-prefix matching, an LRU token budget, and reference counting.
type Tree struct {
	root *node
	// nsRoots holds the VIRTUAL PER-NAMESPACE ROOTS that give node identity the
	// (tokens, nsKey) shape SGLang's RadixKey does (#3889): a Lookup/Insert under
	// namespace ns walks from nsRoots[ns] (root itself for the default "" namespace),
	// so identical token prefixes under different namespaces — tenant / LoRA adapter /
	// cache-salt / tool-world — can never share a node or its cached K/V. Lazily
	// populated by rootFor; the "" namespace always uses root, so every pre-namespace
	// caller and test stays byte-identical. Eviction and Stats span EVERY root
	// (forEachRoot): the token budget is ONE global pool, only node IDENTITY is
	// namespace-scoped. See namespace.go.
	nsRoots              map[string]*node
	maxTokens            int    // LRU budget in cached tokens; 0 disables eviction (unbounded)
	maxSnapshotBytes     int64  // complete snapshot payload ceiling; 0 is unlimited
	maxHostSnapshotBytes int64  // complete host-DRAM L2 payload ceiling; 0 disables L2
	tokens               int    // total cached tokens = Σ len(node.key) over all nodes
	snapshotBytes        int64  // hot complete snapshots plus their cached final logits
	hostSnapshotBytes    int64  // complete PrefixSnapshot payloads physically owned by host DRAM
	clock                uint64 // logical access clock

	l1Hits         int
	l1Misses       int
	l1Faults       int
	l1HitTokens    int
	l2Hits         int
	l2Misses       int
	l2Faults       int
	l2HitTokens    int
	l2StageBytes   int64
	l2RestoreBytes int64
	l2Evictions    int

	remoteSnapshotStore SnapshotStore
	remoteModelID       string
	remoteBackend       compute.Backend
	remoteConfig        model.Config
	remoteSnapshotBytes int64
	remoteBreaker       *RemoteL3Breaker
	l3Hits              int
	l3Misses            int
	l3Faults            int
	l3HitTokens         int
	l3StageBytes        int64
	l3RestoreBytes      int64
	l3StageNanos        int64
	l3RestoreNanos      int64
	l3StageFaults       int
	l3RestoreFaults     int

	// policy is the legacy two-value enum, retained as the back-compat knob and the
	// selector for the cost-aware eviction counter (Stats.CostEvictions). strategy is the
	// open victim seam (#3890, eviction_strategy.go): a nil strategy is pure LRU, so a bare
	// New() tree is byte-identical to before. SetEvictionPolicy routes through the strategy
	// registry and keeps policy in sync, so both dials share one victimLeaf argmin.
	policy      EvictionPolicy
	strategy    VictimStrategy
	pageTracker PageOccupancyTracker

	// retention is the signed prefix-cache retention override (#4039): once set via
	// SetRetention it supersedes maxTokens with ctxresidency's signed grammar --
	// <0 keep-all, 0 keep-none, >0 evict-to-N -- so radixkv exposes ONE legible dial
	// at a single zero-point. maxTokens is retained as the legacy positive-only budget
	// so New() and Stats stay byte-identical for every existing caller (retentionSet
	// is false until a caller opts into the signed dial).
	retention    int
	retentionSet bool

	evictions       int // LRU leaf evictions
	costEvictions   int // cost-aware leaf evictions
	pageEvictions   int // page-aware leaf evictions
	policyEvictions int // EvictNode calls (the fak differentiator)
	splits          int // edge splits performed (a structural RadixAttention event)

	// Prefix-cache LIFECYCLE event counters (#5804, lookupobs.go). The counters above
	// book evictions and structural splits; these book the other three events a
	// warm/cold rerun needs — every lookup, its hit/miss verdict with a typed miss
	// reason, and every fill. Maintained by LookupNS and attachLeaf; read via Stats.
	lookups             int // LookupNS calls (demand reuse probes)
	lookupHits          int // lookups that matched at least one cached token
	lookupMissCold      int // misses where nothing COULD have matched (empty tree / empty request)
	lookupMissDivergent int // misses where a populated tree shared no leading token
	fills               int // leaves attached (demand Insert + prewarm WarmInsert)

	lastEvictPolicy       string
	lastEvictCandidates   int
	lastEvictLocked       int
	lastEvictVictimCost   float64
	lastEvictVictimHits   int
	lastEvictVictimTokens int
	lastEvictVictimPrefix int

	// Evict→reuse thrash detector state (#3393, thrash.go): a bounded just-evicted
	// side-map (FIFO ring + latest-record index, ≤ thrashCap entries) plus the
	// accumulated thrash counters it feeds. Observability only — never steers eviction.
	thrashRing  []thrashRecord // lazily allocated on the first budget eviction
	thrashHead  int
	thrashCount int
	thrashIndex map[uint64]thrashRecord

	thrashReuses   int    // evictions whose key was re-demanded within thrashWindow
	thrashTokens   int    // Σ victim edge tokens over those thrashes (wasted recompute)
	thrashGapTotal uint64 // Σ evict→reuse gap (logical ticks) over those thrashes
	thrashGapLast  uint64 // gap of the most recent thrash
	thrashGapMax   uint64 // largest gap counted

	// Bounded-eviction confirmed-delete ledger (#3387, plan.go): staged-unconfirmed
	// victims in FIFO staging order (≤ evictionLedgerCap entries) plus the accumulated
	// plane counters. Additive plane — evictToBudget never touches it.
	evictLedger           []evictionRecord
	boundedEvictions      int // victims staged (and detached) by PlanBoundedEviction
	ledgerConfirmed       int // victims confirmed by ConfirmEvictions, each exactly once
	ledgerConfirmedTokens int // Σ victim edge tokens over confirmed deletes

	// Value/frequency admission (#9311, admission.go). Cost-aware trees enable the
	// bounded sketch gate by default; other policies and an explicit rollback keep the
	// historical insert-always behavior. The sketch has fixed cells, while the reject
	// journal is a bounded set of candidates awaiting a later successful recovery.
	admissionEnabled            bool
	admissionSketch             *cacheprice.FrequencySketch
	admissionJournal            []admissionRecord
	admissionObservations       int
	admissionCandidates         int
	admissionAdmitted           int
	admissionRejected           int
	admissionRejectedTokens     int
	admissionRejectedBytes      int64
	admissionTelemetryFallbacks int
	admissionHotProtected       int
	admissionRecoveries         int
	admissionJournalDropped     int
	admissionRecoveryGapLast    uint64
	admissionRecoveryGapMax     uint64
	lastAdmissionFrequency      int
	lastAdmissionReason         string

	// Demote-before-drop (#3414, epic #2236). Behind an opt-in dial, evictToBudget consults
	// a demotion decider before dropping a victim leaf so that spans can spill to a lower
	// tier instead of being dropped when restore undercuts recompute.
	demoteBeforeDrop bool
	demotionDecider  DemotionDecider
	demotions        int
}

// New builds an empty prefix cache. maxTokens is the LRU budget in cached tokens; pass 0
// for an unbounded tree (no eviction — the right setting when measuring the matching
// algorithm's pure hit rate, before layering a memory bound on top).
func New(maxTokens int) *Tree {
	return &Tree{root: &node{children: map[int]*node{}}, maxTokens: maxTokens}
}

// NewWithEvictionPolicy builds a tree with an explicit budget-pressure policy. Unknown
// policy values fall back to LRU so callers cannot accidentally widen behavior.
func NewWithEvictionPolicy(maxTokens int, policy EvictionPolicy) *Tree {
	return NewWithBudgetsAndEvictionPolicy(maxTokens, 0, policy)
}

func NewWithBudgets(maxTokens int, maxSnapshotBytes int64) *Tree {
	return NewWithBudgetsAndEvictionPolicy(maxTokens, maxSnapshotBytes, EvictionLRU)
}

func NewWithBudgetsAndEvictionPolicy(maxTokens int, maxSnapshotBytes int64, policy EvictionPolicy) *Tree {
	return NewWithTierBudgetsAndEvictionPolicy(maxTokens, maxSnapshotBytes, 0, policy)
}

// NewWithTierBudgetsAndEvictionPolicy builds a tree with independent hot
// snapshot and host-DRAM L2 payload budgets. A zero host budget leaves the L2
// disabled, preserving every existing constructor's behavior.
func NewWithTierBudgetsAndEvictionPolicy(maxTokens int, maxSnapshotBytes, maxHostSnapshotBytes int64, policy EvictionPolicy) *Tree {
	if maxTokens < 0 {
		maxTokens = 0
	}
	if maxSnapshotBytes < 0 {
		maxSnapshotBytes = 0
	}
	if maxHostSnapshotBytes < 0 {
		maxHostSnapshotBytes = 0
	}
	t := &Tree{
		root:                 &node{children: map[int]*node{}},
		maxTokens:            maxTokens,
		maxSnapshotBytes:     maxSnapshotBytes,
		maxHostSnapshotBytes: maxHostSnapshotBytes,
	}
	t.SetEvictionPolicy(policy)
	return t
}

// NewWithRetention builds an empty prefix cache with the signed retention dial set
// directly (#4039): n<0 keep-all, n==0 keep-none, n>0 evict-to-N. It is the
// signed-grammar sibling of New(), whose legacy argument is positive-only (0 ==
// unbounded); this constructor lets a caller pick keep-none or keep-all explicitly.
func NewWithRetention(n int) *Tree {
	t := New(0)
	t.SetRetention(n)
	return t
}

// SetRetention sets prefix-cache retention with one signed dial, matching
// ctxresidency.SetMaxHeld's grammar so radixkv shares a single legible zero-point:
//
//	n <  0  keep-all   -- disable eviction; retain the whole tree (unbounded)
//	n == 0  keep-none  -- retain nothing; the tree is evicted back toward empty
//	n >  0  evict-to-N  -- bound the tree to N cached tokens (LRU / cost-aware)
//
// It supersedes the legacy New(maxTokens) budget and re-applies immediately, so
// lowering it trims the live tree now. Only unlocked (unleased) leaves are evictable,
// so keep-none is best-effort: a prefix currently being served survives until Done.
func (t *Tree) SetRetention(n int) {
	t.retention = n
	t.retentionSet = true
	t.evictToBudget()
}

// resolveBudget folds the retention dial (when set) or the legacy maxTokens budget
// into (tokenBudget, evicts): evicts=false is keep-all (unbounded); evicts=true bounds
// the tree to tokenBudget (0 for keep-none, N for evict-to-N).
func (t *Tree) resolveBudget() (budget int, evicts bool) {
	if t.retentionSet {
		if t.retention < 0 {
			return 0, false // keep-all
		}
		return t.retention, true // 0 keep-none, N evict-to-N
	}
	if t.maxTokens > 0 {
		return t.maxTokens, true
	}
	return 0, false // legacy 0 (or negative) == unbounded
}

// effectiveMaxTokens reports the positive token budget for Stats back-compat: N for
// evict-to-N, and 0 for both keep-all (unbounded) and keep-none -- preserving the
// field's historical "0 == unbounded" reading for every legacy New() caller (keep-none
// is only reachable through the new signed SetRetention dial). For a tree that never
// touched SetRetention this equals the old t.maxTokens exactly.
func (t *Tree) effectiveMaxTokens() int {
	if b, evicts := t.resolveBudget(); evicts {
		return b
	}
	return 0
}

// SetEvictionPolicy changes the budget-pressure victim rule for future evictions only. It
// is the legacy two-value dial; it routes through the open strategy registry (#3890) so
// both dials share one victim seam. An unknown enum value maps to "lru" (String()'s
// default), preserving the fail-closed-to-LRU contract callers relied on.
func (t *Tree) SetEvictionPolicy(policy EvictionPolicy) {
	_ = t.SetEvictionStrategy(policy.String()) // enum names are always registered — never errors
}

func (t *Tree) tick() uint64 { t.clock++; return t.clock }

// walk finds the longest prefix of tokens already in the tree, starting from the namespace
// root `start` (rootFor(ns) / rootForRead(ns)). It returns the deepest node whose full path
// is a prefix of tokens (a node boundary, nlen tokens), plus — if the next edge only
// PARTIALLY matches — the child entered (pc) and how far into its key matched (oi, with
// 0<oi<len(pc.key)). Total matched tokens = nlen + oi. A nil start (an absent namespace on a
// read/verdict path) matches nothing.
func (t *Tree) walk(start *node, tokens []int) (n *node, nlen int, pc *node, oi int) {
	n = start
	if n == nil {
		return nil, 0, nil, 0
	}
	i := 0
	for i < len(tokens) {
		ch := n.children[tokens[i]]
		if ch == nil {
			return n, i, nil, 0
		}
		// Token-by-token run-compare, deliberately NOT SGLang's gallop/binary-search
		// (#3891). That borrow amortizes Python interpreter overhead with vectorized
		// numpy slice compares; Go's []int slices.Equal is a scalar loop with the same
		// per-element cost, so galloping only swaps N cheap inline compares for O(log N)
		// slices.Equal *calls* — a wash on a fully-shared edge and ~20% slower on early
		// divergence (measured, walkbench_test.go BenchmarkCommonRunLen). The correct
		// gallop and its equivalence proof live in test scope as the retirement witness.
		j := 0
		for j < len(ch.key) && i+j < len(tokens) && tokens[i+j] == ch.key[j] {
			j++
		}
		if j == len(ch.key) {
			n, i = ch, i+j // whole edge consumed; descend
			continue
		}
		return n, i, ch, j // partial edge match: boundary is mid-edge ch
	}
	return n, i, nil, 0
}

// boundaryFor walks the longest cached prefix of tokens and, when the match lands mid-edge
// (oi>0), SPLITS that edge so a real node boundary with a reusable cache exists exactly at
// the matched length — returning that boundary node and the total matched token count. It is
// the shared "walk then split-to-boundary" preamble both the demand path (Lookup) and the
// prefetch path (WarmInsert) run before they attach a suffix; it performs the same structural
// split (t.splits bumps once, via t.split) and mutates nothing else, so both callers observe
// byte-identical tree state. It does NOT touch recency or leases — each caller applies its own
// residency policy afterward.
func (t *Tree) boundaryFor(start *node, tokens []int) (boundary *node, matched int) {
	n, nlen, pc, oi := t.walk(start, tokens)
	boundary, matched = n, nlen
	if oi > 0 {
		boundary = t.split(n, pc, oi)
		matched = nlen + oi
	}
	return boundary, matched
}

// MatchLen is MatchLenNS on the default ("") namespace — the pre-namespace signature every
// existing caller uses, unchanged.
func (t *Tree) MatchLen(tokens []int) int { return t.MatchLenNS("", tokens) }

// MatchLenNS is the read-only accounting probe for namespace ns: the number of leading
// tokens of `tokens` already cached UNDER ns (a node boundary or mid-edge), with no
// mutation, no lock, no split. Because each namespace has its own virtual root (#3889), a
// probe under ns "A" never counts a prefix cached only under ns "B" even when the token ids
// collide. An absent namespace matches nothing (0). This is the hit-rate measurement seam —
// sum it over a workload and divide by total tokens to get the cache hit rate SGLang reports.
func (t *Tree) MatchLenNS(ns string, tokens []int) int {
	_, nlen, _, oi := t.walk(t.rootForRead(ns), tokens)
	return nlen + oi
}

// split inserts an intermediate node between parent and child at offset oi into child.key,
// so a node boundary exists exactly at the matched prefix length (parent.plen+oi). The
// intermediate node's KV is the child's cache TRUNCATED to that length when the model
// supports span eviction. This is the move that lets RadixAttention reuse a prefix that
// diverges in the middle of a run, which fak's declare-the-prefix path could not.
func (t *Tree) split(parent, child *node, oi int) *node {
	first := child.key[0]
	mid := &node{
		key:      append([]int(nil), child.key[:oi]...),
		parent:   parent,
		children: map[int]*node{},
		plen:     parent.plen + oi,
		lastUsed: child.lastUsed,
		hits:     child.hits,
		chunkID:  child.chunkID,
	}
	if child.kv != nil {
		mid.kv = truncatePrefix(child.kv, mid.plen)
	}
	child.key = append([]int(nil), child.key[oi:]...)
	child.parent = mid
	mid.children[child.key[0]] = child
	parent.children[first] = mid // first == mid.key[0]; overwrites child's old slot
	t.splits++
	return mid
}

// Lookup is LookupNS on the default ("") namespace — the pre-namespace signature every
// existing caller uses, unchanged.
func (t *Tree) Lookup(tokens []int) (*node, int) { return t.LookupNS("", tokens) }

// LookupNS matches the longest cached prefix of tokens WITHIN namespace ns, SPLITTING an
// edge if the match lands mid-run so a real node boundary (with a reusable cache) exists
// there, bumps LRU recency along the matched path, and LEASES the boundary node (refs++) so
// an eviction cannot reclaim the prefix while the caller is serving from it. Returns the
// boundary node (call node.KV() for the clone-able reuse cache; nil if nothing matched) and
// the matched token count. The caller prefills tokens[matched:] and then calls Insert + Done.
//
// The walk starts from ns's virtual root (#3889), so a request under one namespace can NEVER
// be served a K/V span cached under another — even when their token prefixes are identical
// (different tenant / LoRA adapter / cache-salt / tool-world). Insert needs no namespace: it
// attaches to the boundary node LookupNS returns, which already lives under ns's root.
//
// LEASE DISCIPLINE (memory-leak contract — TestLookupLeaseMustBeReleased guards it):
// Lookup takes a lease (refs++) on the boundary. The caller MUST release it on EVERY
// path — Insert hands the lease off to the new leaf, and Done releases it directly. If a
// caller returns early between Lookup and Insert (prefill error, context cancel, panic)
// WITHOUT calling Done, the boundary stays leased forever, and because evictToBudget never
// reclaims a leased node OR its ancestors, the whole prefix subtree is pinned against the
// budget permanently — a logical memory leak. Use `defer` so the abort path cannot forget:
//
//	b, m := tree.Lookup(req)
//	leaf := b            // what we ultimately Done()
//	defer func() { tree.Done(leaf) }()
//	... prefill (may fail) ...
//	leaf = tree.Insert(b, req[m:], kv)   // moves the lease onto the leaf
func (t *Tree) LookupNS(ns string, tokens []int) (*node, int) {
	root := t.rootFor(ns)
	t.observeAdmission(root, tokens)
	// Lifecycle bookkeeping (#5804): read whether this namespace held anything BEFORE the
	// walk, so a miss can be attributed to an empty cache rather than a divergent request.
	populated := len(root.children) > 0
	boundary, matched := t.boundaryFor(root, tokens)
	t.countLookup(matched, populated && len(tokens) > 0)
	for p := boundary; p != nil; p = p.parent {
		p.lastUsed = t.clock + 1 // freshen the whole hot path
	}
	t.clock++
	if matched > 0 {
		for p := boundary; p != nil && p.parent != nil; p = p.parent {
			p.hits++ // p.parent==nil marks a namespace root, which never accrues hits
		}
	}
	// Thrash probe (#3393): was this exact request path just evicted? Demand for it is
	// back — count the evict→reuse gap. Free when the side-map is empty.
	t.probeThrash(root, tokens)
	boundary.refs++
	return boundary, matched
}

// Insert attaches the request's suffix (the tokens beyond the matched prefix) as a new
// child of the boundary node, with kv = the FULL-prefix cache the caller built for the
// whole request (matched prefix reused + suffix prefilled). It then enforces the LRU
// budget. An empty suffix means the request was already fully cached: nothing is added and
// the (still-leased) boundary node is returned. kv may be nil (pure-accounting mode).
//
// Lease handoff: Lookup leases the matched BOUNDARY so the reused prefix can't be evicted
// in the gap before this Insert; Insert moves that lease onto the new LEAF (the request's
// full path) and frees the boundary, because a non-leaf is never an eviction candidate, so
// holding the leaf protects every ancestor too. The caller therefore always Dones the node
// Insert returns, exactly when the request finishes — and the just-inserted request is
// itself protected from the eviction its own Insert may trigger.
func (t *Tree) Insert(boundary *node, suffix []int, kv *model.KVCache) *node {
	return t.InsertWithLogits(boundary, suffix, kv, nil)
}

// LookupSnapshot is Lookup plus a deep clone/restore of the longest complete
// snapshot, consulting host DRAM only after the hot snapshot tier misses.
func (t *Tree) LookupSnapshot(tokens []int) (*node, *model.PrefixSnapshot, int, error) {
	n, snap, matched, _, err := t.LookupSnapshotTiered(tokens)
	return n, snap, matched, err
}

// LookupSnapshotTiered is LookupSnapshot with truthful source-tier attribution.
func (t *Tree) LookupSnapshotTiered(tokens []int) (*node, *model.PrefixSnapshot, int, SnapshotTier, error) {
	return t.LookupSnapshotTieredContext(context.Background(), tokens)
}

// LookupSnapshotTieredContext is the request-cancellable L1→L2→L3 lookup.
func (t *Tree) LookupSnapshotTieredContext(ctx context.Context, tokens []int) (*node, *model.PrefixSnapshot, int, SnapshotTier, error) {
	n, _ := t.Lookup(tokens)
	for candidate := n; candidate != nil; candidate = candidate.parent {
		if candidate.snapshot == nil {
			continue
		}
		snap, err := candidate.snapshot.Clone()
		if err != nil {
			t.l1Faults++
			return n, nil, candidate.plen, SnapshotTierDeviceL1, err
		}
		t.l1Hits++
		t.l1HitTokens += candidate.plen
		return n, snap, candidate.plen, SnapshotTierDeviceL1, nil
	}
	t.l1Misses++
	if !t.HostL2Enabled() && !t.RemoteSnapshotEnabled() {
		return n, nil, 0, SnapshotTierMiss, nil
	}
	for candidate := n; candidate != nil; candidate = candidate.parent {
		if candidate.hostSnapshot == nil {
			continue
		}
		snap, err := candidate.hostSnapshot.Restore()
		if err != nil {
			t.l2Faults++
			return n, nil, candidate.plen, SnapshotTierHostL2, err
		}
		t.l2Hits++
		t.l2HitTokens += candidate.plen
		t.l2RestoreBytes += candidate.hostSnapshot.TransferBytes()
		return n, snap, candidate.plen, SnapshotTierHostL2, nil
	}
	if t.HostL2Enabled() {
		t.l2Misses++
	}
	if t.RemoteSnapshotEnabled() {
		for candidate := n; candidate != nil; candidate = candidate.parent {
			if candidate.remoteSnapshot == nil {
				continue
			}
			snap, found, err := t.restoreSnapshotFromRemote(ctx, "", candidate)
			if err != nil {
				return n, nil, candidate.plen, SnapshotTierRemoteL3, err
			}
			if found {
				return n, snap, candidate.plen, SnapshotTierRemoteL3, nil
			}
		}
		t.l3Misses++
	}
	return n, nil, 0, SnapshotTierMiss, nil
}

// InsertSnapshot transfers an independently owned device/hybrid snapshot to the tree. A
// value/frequency bypass is a successful cache decision: the request already computed the
// prefix, so the snapshot is closed here and the still-leased boundary is returned for Done.
// Byte-budget errors retain the historical ownership contract: ownership remains with the
// caller, which receives the error and closes the rejected snapshot.
func (t *Tree) InsertSnapshot(boundary *node, suffix []int, snap *model.PrefixSnapshot, logits []float32) (*node, error) {
	incoming := snapshotResidentBytes(snap, logits)
	if t.maxSnapshotBytes > 0 && incoming > t.maxSnapshotBytes {
		return nil, ErrSnapshotByteBudget
	}
	comparisons := t.snapshotAdmissionComparisons(boundary, suffix, incoming)
	n, admitted := t.insertWithLogits(boundary, suffix, nil, nil, comparisons)
	if !admitted {
		snap.Close()
		return n, nil
	}
	if n == nil {
		return nil, ErrSnapshotByteBudget
	}
	oldBytes := snapshotResidentBytes(n.snapshot, n.cachedLogits)
	if !t.makeSnapshotRoom(incoming-oldBytes, n) {
		return n, ErrSnapshotByteBudget
	}
	if n.snapshot != nil {
		n.snapshot.Close()
	}
	t.snapshotBytes -= oldBytes
	n.snapshot = snap
	n.cachedLogits = append([]float32(nil), logits...)
	t.snapshotBytes += incoming
	return n, nil
}

func snapshotResidentBytes(snap *model.PrefixSnapshot, logits []float32) int64 {
	if snap == nil {
		return 0
	}
	return snap.ResidentBytes() + int64(len(logits))*4
}

func (t *Tree) makeSnapshotRoom(delta int64, exclude *node) bool {
	if delta <= 0 || t.maxSnapshotBytes == 0 {
		return true
	}
	for t.snapshotBytes+delta > t.maxSnapshotBytes {
		victim := t.snapshotVictim(exclude)
		if victim == nil {
			return false
		}
		t.releaseHotSnapshot(victim)
	}
	return true
}

func (t *Tree) snapshotVictim(exclude *node) *node {
	strat := t.evictionStrategy()
	if prep, ok := strat.(TreePreparer); ok {
		prep.PrepareTree(t)
	}
	var victim *node
	var walk func(*node)
	walk = func(n *node) {
		if n != exclude && n.refs == 0 && n.snapshot != nil {
			if victim == nil || strat.Priority(n).less(strat.Priority(victim)) {
				victim = n
			}
		}
		for _, child := range n.children {
			walk(child)
		}
	}
	t.forEachRoot(walk)
	return victim
}

func (t *Tree) releaseHotSnapshot(n *node) {
	if n == nil || n.snapshot == nil {
		return
	}
	t.snapshotBytes -= snapshotResidentBytes(n.snapshot, n.cachedLogits)
	n.snapshot.Close()
	n.snapshot = nil
	if n.hostSnapshot == nil && n.remoteSnapshot == nil {
		n.cachedLogits = nil
	}
}

// InsertWithLogits is Insert plus an optional exact-prefix logits payload. The logits are
// copied on admission because quantized sessions may reuse their logits buffer on the next
// Step/Prefill. Passing nil preserves Insert's historical behavior.
func (t *Tree) InsertWithLogits(boundary *node, suffix []int, kv *model.KVCache, logits []float32) *node {
	n, _ := t.insertWithLogits(boundary, suffix, kv, logits, nil)
	return n
}

func (t *Tree) insertWithLogits(boundary *node, suffix []int, kv *model.KVCache, logits []float32, comparisons []admissionComparison) (*node, bool) {
	return t.insertWithLogitsAndChunk(boundary, suffix, kv, logits, comparisons, 0)
}

func (t *Tree) insertWithLogitsAndChunk(boundary *node, suffix []int, kv *model.KVCache, logits []float32, comparisons []admissionComparison, chunkID int) (*node, bool) {
	if boundary == nil {
		return nil, false
	}
	keyHash := t.admissionKeyHash(boundary, suffix)
	if t.admissionEnabled {
		comparisons = append(comparisons, t.tokenAdmissionComparisons(boundary, suffix)...)
	} else {
		comparisons = nil
	}
	stamp := t.clock
	if len(suffix) > 0 {
		stamp = t.tick()
	}
	if len(comparisons) > 0 && !t.admitCandidate(keyHash, comparisons) {
		return boundary, false
	}
	if len(suffix) == 0 {
		if logits != nil {
			boundary.logits = append([]float32(nil), logits...)
		}
		if chunkID != 0 {
			boundary.chunkID = chunkID
		}
		t.noteAdmissionRecovery(keyHash)
		return boundary, true // already fully cached; keep the boundary lease for the caller to Done
	}
	leaf := t.attachLeafWithChunk(boundary, suffix, kv, logits, stamp, chunkID)
	leaf.refs++ // lease the in-flight request's own leaf...
	if boundary.refs > 0 {
		boundary.refs-- // ...and release the prefix lease Lookup took (the leaf now guards the path)
	}
	t.evictToBudget()
	t.noteAdmissionRecovery(keyHash)
	return leaf, true
}

// attachLeaf builds a new leaf node hanging suffix off boundary (copying suffix
// so the tree owns the slice), links it as boundary's child, and grows the
// tree's token count — the node-construction step both Insert and WarmInsert
// (prewarm.go) share; each caller applies its own lease/recency bookkeeping and
// eviction pass on the returned leaf.
func (t *Tree) attachLeaf(boundary *node, suffix []int, kv *model.KVCache, logits []float32, lastUsed uint64) *node {
	return t.attachLeafWithChunk(boundary, suffix, kv, logits, lastUsed, 0)
}

func (t *Tree) attachLeafWithChunk(boundary *node, suffix []int, kv *model.KVCache, logits []float32, lastUsed uint64, chunkID int) *node {
	// Thrash probe (#3393): the new leaf's full path is (root→boundary)+suffix — if that
	// exact key was just evicted, this attach is the re-insert that proves the eviction
	// premature. Covers both demand Insert and WarmInsert; a Lookup-consumed entry is
	// simply absent here (consumed once).
	t.probeThrash(boundary, suffix)
	t.fills++ // one prefix-cache FILL event (#5804): demand Insert or prewarm WarmInsert
	s := append([]int(nil), suffix...)
	leaf := &node{
		key:      s,
		parent:   boundary,
		children: map[int]*node{},
		kv:       kv,
		logits:   append([]float32(nil), logits...),
		plen:     boundary.plen + len(s),
		lastUsed: lastUsed,
		chunkID:  chunkID,
	}
	if chunkID == 0 && t.pageTracker != nil {
		if id := t.pageTracker.ChunkID(leaf); id != 0 {
			leaf.chunkID = id
		}
	}
	boundary.children[s[0]] = leaf
	t.tokens += len(s)
	return leaf
}

// Done releases a lease taken by Lookup/Insert (refs--), so the node becomes LRU-evictable again.
func (t *Tree) Done(n *node) {
	if n != nil && n.refs > 0 {
		n.refs--
	}
	t.pruneEmptyNamespaceRoot(n)
}

// KV is the reusable kernel-owned cache for this node's full prefix (nil if none). Clone
// it (model.SessionFromPrefix does) before prefilling a suffix; never mutate it in place.
func (n *node) KV() *model.KVCache { return n.kv }

// Logits returns a copy of the final-token logits for this exact full prefix, or nil when
// the caller did not cache them. The copy keeps request-local sampling penalties/biases
// from mutating the shared tree payload.
func (n *node) Logits() []float32 {
	if n == nil {
		return nil
	}
	if n.snapshot != nil || n.hostSnapshot != nil || n.remoteSnapshot != nil {
		return append([]float32(nil), n.cachedLogits...)
	}
	return append([]float32(nil), n.logits...)
}

// Plen is the node's cached prefix length in tokens.
func (n *node) Plen() int { return n.plen }

// DemotionDecision indicates the chosen demotion action.
type DemotionDecision struct {
	Action string // "spill", "evict"
	Tier   string // "host", "disk"
}

// DemotionDecider is consulted by evictToBudget when demoteBeforeDrop is true (#3414).
type DemotionDecider interface {
	DecideDemotion(tokens int, hits int, bytes int64) DemotionDecision
}

// SetDemotionDecider configures a demotion decider on the tree.
func (t *Tree) SetDemotionDecider(d DemotionDecider) {
	t.demotionDecider = d
}

// EnableDemoteBeforeDrop toggles whether evictToBudget consults the demotion decider.
func (t *Tree) EnableDemoteBeforeDrop(enable bool) {
	t.demoteBeforeDrop = enable
}

// Demotions reports the number of victims successfully routed to a demotion fate.
func (t *Tree) Demotions() int {
	return t.demotions
}

// evictToBudget evicts least-recently-used, unlocked LEAVES until the cached-token count
// is within budget — RadixAttention's eviction policy. Removing a leaf can make its parent
// a leaf, which the next iteration may then evict (the upward collapse). A node that is
// leased (refs>0) is never chosen, so a prefix being served survives memory pressure.
func (t *Tree) evictToBudget() {
	budget, evicts := t.resolveBudget()
	if !evicts {
		return // keep-all: eviction disabled
	}
	for t.tokens > budget {
		v := t.victimLeaf()
		if v == nil {
			return // everything in budget-excess is locked; cannot evict further
		}
		if t.demoteBeforeDrop && t.demotionDecider != nil {
			footprintBytes := int64(len(v.key) * 4) // estimated footprint
			res := t.demotionDecider.DecideDemotion(len(v.key), v.hits, footprintBytes)
			if res.Action == "spill" {
				t.demotions++
			}
		}
		t.noteEviction(v) // thrash detector (#3393): remember the victim before it goes
		t.removeLeaf(v)
		t.evictions++
		if t.policy == EvictionCostAware {
			t.costEvictions++
		}
		if t.policy == EvictionPageAware {
			t.pageEvictions++
		}
	}
}

// victimLeaf selects the budget-pressure victim: a single argmin over the active strategy's
// Priority key across every unlocked leaf — SGLang's heappush((strategy.get_priority(node),
// node)) collapsed to one pass. One traversal, one comparison rule; the tuple ordering in
// victimKey does the tiering, so lru / cost-aware / slru all share this loop with no
// per-policy branch (#3890). Leased leaves (refs>0) are counted as locked and skipped,
// never scored, so a prefix being served is never a victim. Returns nil when every
// budget-excess leaf is locked — the signal evictToBudget bails on.
func (t *Tree) victimLeaf() *node {
	return t.selectVictimLeaf(true)
}

func (t *Tree) selectVictimLeaf(record bool) *node {
	strat := t.evictionStrategy()
	if prep, ok := strat.(TreePreparer); ok {
		prep.PrepareTree(t)
	}
	var best *node
	var bestKey victimKey
	candidates, locked := 0, 0
	var stack []*node
	t.forEachRoot(func(r *node) { // one global victim pool spanning every namespace root
		for _, c := range r.children {
			stack = append(stack, c)
		}
	})
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if len(n.children) == 0 {
			candidates++
			if n.refs > 0 {
				locked++
				continue
			}
			k := strat.Priority(n)
			if best == nil || k.less(bestKey) {
				best, bestKey = n, k
			}
			continue
		}
		for _, c := range n.children {
			stack = append(stack, c)
		}
	}
	if record {
		t.recordEvictChoice(strat.Name(), candidates, locked, best)
	}
	return best
}

func (n *node) kvSpanStats() compute.KVSpanStats {
	return compute.KVSpanStats{
		Tokens:   len(n.key),
		Bytes:    int64(max(n.plen, 1)),
		Hits:     n.hits,
		LastUsed: n.lastUsed,
		Leased:   n.refs > 0,
	}
}

// lastEvictPolicyName is the strategy name of the most recent eviction choice, defaulting
// to "lru" before any eviction has run — preserving the reading the EvictionPolicy-zero
// value gave (its String() was "lru") for every legacy Stats reader.
func (t *Tree) lastEvictPolicyName() string {
	if t.lastEvictPolicy == "" {
		return "lru"
	}
	return t.lastEvictPolicy
}

func (t *Tree) recordEvictChoice(strategyName string, candidates, locked int, victim *node) {
	t.lastEvictPolicy = strategyName
	t.lastEvictCandidates = candidates
	t.lastEvictLocked = locked
	t.lastEvictVictimCost = 0
	t.lastEvictVictimHits = 0
	t.lastEvictVictimTokens = 0
	t.lastEvictVictimPrefix = 0
	if victim == nil {
		return
	}
	st := victim.kvSpanStats()
	cost := compute.KVEvictionCost(st)
	if math.IsInf(cost, 0) {
		cost = 0
	}
	t.lastEvictVictimCost = cost
	t.lastEvictVictimHits = st.Hits
	t.lastEvictVictimTokens = st.Tokens
	t.lastEvictVictimPrefix = victim.plen
}

func (t *Tree) removeLeaf(v *node) {
	if v.parent == nil {
		return
	}
	delete(v.parent.children, v.key[0])
	t.tokens -= len(v.key)
	t.releaseSnapshotPayload(v)
	v.kv = nil
}

// EvictNode is POLICY eviction: it removes a node and its entire subtree from the cache
// regardless of LRU recency or lease state, returning the tokens freed. This is the seam a
// quarantine verdict drives — the radix-tree analogue of model.KVCache.Evict, and the
// capability an opportunistic LRU cache cannot provide: a poisoned prefix is removed
// because POLICY says so, not because memory ran out. (Callers serving from a leased node
// should still hold their own session; this drops the SHARED cached copy so no future
// request can reuse the poisoned prefix.)
func (t *Tree) EvictNode(n *node) int {
	if n == nil || n.parent == nil { // parent==nil is any namespace root (incl. the default)
		return 0
	}
	freed := subtreeTokens(n)
	t.closeSubtreeSnapshots(n)
	delete(n.parent.children, n.key[0])
	t.tokens -= freed
	t.policyEvictions++
	return freed
}

// EvictPrefix is verdict-driven eviction by TOKEN PATH: it walks the longest cached
// prefix of `tokens` and evicts the deepest cached node on that path (its whole
// subtree), returning the tokens freed. It is the EvictNode seam for a caller that
// holds the poisoned token sequence rather than a *node handle — exactly the in-kernel
// planner, which caches by token path and (because radixkv's node type is unexported)
// keeps no node references to feed EvictNode directly. A quarantine verdict passes the
// cached transcript THROUGH the poisoned span and the branch that cached it — and
// everything built on it — is dropped, so no later request can reuse the poisoned KV.
//
// It is a no-op (returns 0) when nothing along `tokens` is cached PAST where it would
// diverge — in particular, when the cached path ends BEFORE `tokens` is exhausted
// (`tokens` extends into a region that was never prefilled+cached), nothing is evicted.
// That guard is what keeps a clean shared prefix intact when the poison's continuation
// was never cached. Sibling branches that diverged before the matched node are always
// preserved (the benign-prefix guarantee), the same span-exact, sibling-sparing
// governance EvictNode provides.
//
// Contract: `tokens` must END at or within the poisoned span (the caller — a quarantine
// verdict — renders the transcript THROUGH the poisoned result, so a full match lands on
// a node whose cached KV genuinely attended to the poison). A coincidental mid-edge
// collision on the poison's first divergent token can conservatively evict a benign
// sibling branch; that is at worst a cache miss (it re-prefills), never a poison replay.
func (t *Tree) EvictPrefix(tokens []int) int { return t.EvictPrefixNS("", tokens) }

// EvictPrefixNS is EvictPrefix scoped to namespace ns (#3889): it walks ns's virtual root
// and evicts only within that namespace, so quarantining a poisoned prefix under one tenant
// / adapter / cache-salt never drops a benign identical-token prefix cached under another.
// An absent namespace is a no-op (returns 0).
func (t *Tree) EvictPrefixNS(ns string, tokens []int) int {
	if len(tokens) == 0 {
		return 0
	}
	n, nlen, pc, oi := t.walk(t.rootForRead(ns), tokens)
	_ = oi
	switch {
	case pc != nil:
		// `tokens` diverged MID-EDGE into child pc: the matched run tokens[nlen:nlen+oi)
		// is on pc's edge, so pc's root→pc path includes those (poison) tokens. Evict pc's
		// whole branch; pc's siblings (which diverged earlier) survive.
		return t.EvictNode(pc)
	case nlen == len(tokens) && n != nil && n.parent != nil:
		// The ENTIRE poisoned token path matched a cached node exactly: that node's KV
		// attended to the poison. Evict it and its subtree.
		return t.EvictNode(n)
	default:
		// nlen < len(tokens): the cached path ended before `tokens` was consumed — the
		// poison continuation was never cached here, so there is nothing poisoned to drop.
		return 0
	}
}

func (t *Tree) closeSubtreeSnapshots(n *node) {
	if n == nil {
		return
	}
	t.releaseSnapshotPayload(n)
	for _, child := range n.children {
		t.closeSubtreeSnapshots(child)
	}
}

func subtreeTokens(n *node) int {
	total := len(n.key)
	for _, c := range n.children {
		total += subtreeTokens(c)
	}
	return total
}

// truncatePrefix returns a clone of c covering only its first L positions, bit-identical
// to a prefill that stopped at L. It uses model.KVCache.TryEvict of the tail span [L, Len):
// because no position is cached after the evicted tail, Evict renumbers and re-RoPEs no
// survivor, so the result is an exact prefix. Recurrent caches that cannot be truncated
// return nil so the split remains a structural boundary and callers recompute.
func truncatePrefix(c *model.KVCache, L int) *model.KVCache {
	cp := c.Clone()
	if L < cp.Len() {
		if _, err := cp.TryEvict(L, cp.Len()-L); err != nil {
			return nil
		}
	}
	return cp
}

// Stats is a snapshot of the cache's structural state for reporting.
type Stats struct {
	Tokens                     int          // total cached tokens (Σ edge lengths) — the LRU-budget metric
	PrefixTokens               int          // Σ node.plen over nodes holding a kv — TRUE resident KV positions
	Nodes                      int          // non-root nodes
	SnapshotBytes              int64        `json:"snapshot_bytes"`
	MaxSnapshotBytes           int64        `json:"max_snapshot_bytes"`
	HostSnapshotBytes          int64        `json:"host_snapshot_bytes"`
	MaxHostSnapshotBytes       int64        `json:"max_host_snapshot_bytes"`
	DeviceSnapshotBytes        int64        `json:"device_snapshot_bytes"`
	DeviceSnapshotHostBytes    int64        `json:"device_snapshot_host_bytes"`
	DeviceSnapshotTokens       int          `json:"device_snapshot_tokens"`
	L1Hits                     int          `json:"l1_hits"`
	L1Misses                   int          `json:"l1_misses"`
	L1Faults                   int          `json:"l1_faults"`
	L1HitTokens                int          `json:"l1_hit_tokens"`
	L2Hits                     int          `json:"l2_hits"`
	L2Misses                   int          `json:"l2_misses"`
	L2Faults                   int          `json:"l2_faults"`
	L2HitTokens                int          `json:"l2_hit_tokens"`
	L2StageBytes               int64        `json:"l2_stage_bytes"`
	L2RestoreBytes             int64        `json:"l2_restore_bytes"`
	L2Evictions                int          `json:"l2_evictions"`
	L3Enabled                  bool         `json:"l3_enabled"`
	L3ReferencedBytes          int64        `json:"l3_referenced_bytes"`
	L3Hits                     int          `json:"l3_hits"`
	L3Misses                   int          `json:"l3_misses"`
	L3Faults                   int          `json:"l3_faults"`
	L3HitTokens                int          `json:"l3_hit_tokens"`
	L3StageBytes               int64        `json:"l3_stage_bytes"`
	L3RestoreBytes             int64        `json:"l3_restore_bytes"`
	L3StageNanos               int64        `json:"l3_stage_nanos"`
	L3RestoreNanos             int64        `json:"l3_restore_nanos"`
	L3StageFaults              int          `json:"l3_stage_faults"`
	L3RestoreFaults            int          `json:"l3_restore_faults"`
	L3Breaker                  BreakerStats `json:"l3_breaker"`
	L3BreakerState             BreakerState `json:"l3_breaker_state,omitempty"`
	L3BreakerConsecutiveFaults int          `json:"l3_breaker_consecutive_faults,omitempty"`
	L3BreakerTotalFaults       int          `json:"l3_breaker_total_faults,omitempty"`
	L3BreakerOpenSkips         int          `json:"l3_breaker_open_skips,omitempty"`
	L3BreakerProbesAttempted   int          `json:"l3_breaker_probes_attempted,omitempty"`
	L3BreakerProbeRecoveries   int          `json:"l3_breaker_probe_recoveries,omitempty"`
	L3BreakerProbeFailures     int          `json:"l3_breaker_probe_failures,omitempty"`
	L3BreakerOpenedAt          time.Time    `json:"l3_breaker_opened_at,omitempty"`
	Leaves                     int          // leaf nodes
	MaxDepthTokens             int          // longest cached prefix
	Evictions                  int          // LRU leaf evictions performed
	CostEvictions              int          // cost-aware leaf evictions performed
	PageEvictions              int          `json:"page_evictions,omitempty"` // page-aware leaf evictions performed
	PolicyEvictions            int          // EvictNode calls
	Splits                     int          // edge splits performed
	MaxTokens                  int          // configured LRU budget (0 = unbounded)
	EvictionPolicy             string
	ReuseHits                  int

	LastEvictPolicy       string
	LastEvictCandidates   int
	LastEvictLocked       int
	LastEvictVictimCost   float64
	LastEvictVictimHits   int
	LastEvictVictimTokens int
	LastEvictVictimPrefix int

	// Evict→reuse thrash detector (#3393, thrash.go). ThrashReuses counts budget
	// evictions whose exact key was re-demanded within thrashWindow logical ticks —
	// premature evictions, the live "budget/policy too aggressive" signal. Gaps are in
	// logical-clock ticks (deterministic, wall-clock-free).
	ThrashReuses   int    // premature evictions detected (evict→reuse within the window)
	ThrashTokens   int    // Σ victim edge tokens over those thrashes — wasted recompute
	ThrashGapTotal uint64 // Σ evict→reuse gap over those thrashes
	ThrashGapLast  uint64 // gap of the most recent thrash
	ThrashGapMax   uint64 // largest gap counted
	ThrashTracked  int    // just-evicted keys currently probe-able (≤ thrashCap)

	// Bounded-eviction plane (#3387, plan.go). BoundedEvictions counts victims staged
	// (and detached) by PlanBoundedEviction — a subset of Evictions, which counts BOTH
	// planes. LedgerConfirmed lags BoundedEvictions by exactly LedgerPending: staged
	// lifetime == BoundedEvictions, and every staged victim is confirmed exactly once.
	BoundedEvictions      int // victims evicted via the ratio-capped bounded plane
	LedgerConfirmed       int // confirmed deletes settled by ConfirmEvictions
	LedgerConfirmedTokens int // Σ victim edge tokens over confirmed deletes
	LedgerPending         int // staged-unconfirmed ledger entries (≤ evictionLedgerCap)

	// Prefix-cache lifecycle events (#5804, lookupobs.go). These count DEMAND, not
	// residency: they are monotonic for the tree's lifetime and eviction never rolls them
	// back. The three lookup buckets partition every probe exactly —
	// Lookups == LookupHits + LookupMissCold + LookupMissDivergent — which is what makes a
	// 0% hit rate diagnosable: all-cold is warmup, all-divergent is genuine non-overlap.
	Lookups             int // LookupNS calls (demand reuse probes; MatchLenNS is not counted)
	LookupHits          int // probes that matched at least one leading token
	LookupMissCold      int // misses where nothing could have matched (empty namespace / empty request)
	LookupMissDivergent int // misses against a populated namespace that shared no leading token
	Fills               int // leaves attached (demand Insert + prewarm WarmInsert)

	// Value/frequency admission (#9311, admission.go). Candidates count only inserts
	// that would displace token or snapshot residency; no-pressure fills stay off the
	// decision hot path. The fixed sketch and bounded pending journal make overhead
	// explicit, while rejects/recoveries show pollution refused and later heat earned.
	AdmissionEnabled        bool `json:"admission_enabled"`
	AdmissionSketchCells    int  `json:"admission_sketch_cells"`
	AdmissionObservations   int  `json:"admission_observations"`
	AdmissionCandidates     int  `json:"admission_candidates"`
	AdmissionAdmitted       int  `json:"admission_admitted"`
	AdmissionRejected       int  `json:"admission_rejected"`
	AdmissionRejectedTokens int  `json:"admission_rejected_tokens"` // candidate token writes bypassed
	// Candidate value-axis footprint bypassed: exact resident bytes for snapshots and
	// radixkv's existing full-prefix token proxy for regular KV nodes.
	AdmissionRejectedBytes      int64  `json:"admission_rejected_bytes"`
	AdmissionTelemetryFallbacks int    `json:"admission_telemetry_fallbacks"`
	AdmissionHotProtected       int    `json:"admission_hot_protected"`
	AdmissionRecoveries         int    `json:"admission_recoveries"`
	AdmissionJournalPending     int    `json:"admission_journal_pending"`
	AdmissionJournalDropped     int    `json:"admission_journal_dropped"`
	AdmissionRecoveryGapLast    uint64 `json:"admission_recovery_gap_last"`
	AdmissionRecoveryGapMax     uint64 `json:"admission_recovery_gap_max"`
	LastAdmissionFrequency      int    `json:"last_admission_frequency"`
	LastAdmissionReason         string `json:"last_admission_reason"`
}

// Stats walks the tree and returns its current shape.
//
// Note the deliberate Tokens vs PrefixTokens split. The LRU budget (MaxTokens) bounds
// Tokens = Σ edge lengths — SGLang's per-segment accounting, the apples-to-apples
// hit-rate metric this package exists to compare. But each node stores the FULL prefix
// root→node KV (length plen), so the TRUE resident KV footprint is PrefixTokens = Σ plen,
// which can exceed the configured budget by an O(prefix-depth) factor on a deep/narrow
// tree (a single N-token chain holds N·(N+1)/2 positions while Tokens reports only N).
// PrefixTokens makes that gap measurable instead of silent (see TestBudgetVsTrueKVFootprint).
func (t *Tree) Stats() Stats {
	var bStats BreakerStats
	if t.remoteBreaker != nil {
		bStats = t.remoteBreaker.Stats()
	} else {
		bStats = BreakerStats{
			State:          BreakerClosed,
			FaultThreshold: DefaultBreakerFaultThreshold,
			Cooldown:       DefaultBreakerCooldown,
		}
	}
	s := Stats{
		Evictions:                  t.evictions,
		CostEvictions:              t.costEvictions,
		PageEvictions:              t.pageEvictions,
		PolicyEvictions:            t.policyEvictions,
		Splits:                     t.splits,
		MaxTokens:                  t.effectiveMaxTokens(),
		SnapshotBytes:              t.snapshotBytes,
		MaxSnapshotBytes:           t.maxSnapshotBytes,
		HostSnapshotBytes:          t.hostSnapshotBytes,
		MaxHostSnapshotBytes:       t.maxHostSnapshotBytes,
		L1Hits:                     t.l1Hits,
		L1Misses:                   t.l1Misses,
		L1Faults:                   t.l1Faults,
		L1HitTokens:                t.l1HitTokens,
		L2Hits:                     t.l2Hits,
		L2Misses:                   t.l2Misses,
		L2Faults:                   t.l2Faults,
		L2HitTokens:                t.l2HitTokens,
		L2StageBytes:               t.l2StageBytes,
		L2RestoreBytes:             t.l2RestoreBytes,
		L2Evictions:                t.l2Evictions,
		L3Enabled:                  t.RemoteSnapshotEnabled(),
		L3ReferencedBytes:          t.remoteSnapshotBytes,
		L3Hits:                     t.l3Hits,
		L3Misses:                   t.l3Misses,
		L3Faults:                   t.l3Faults,
		L3HitTokens:                t.l3HitTokens,
		L3StageBytes:               t.l3StageBytes,
		L3RestoreBytes:             t.l3RestoreBytes,
		L3StageNanos:               t.l3StageNanos,
		L3RestoreNanos:             t.l3RestoreNanos,
		L3StageFaults:              t.l3StageFaults,
		L3RestoreFaults:            t.l3RestoreFaults,
		L3Breaker:                  bStats,
		L3BreakerState:             bStats.State,
		L3BreakerConsecutiveFaults: bStats.ConsecutiveFaults,
		L3BreakerTotalFaults:       bStats.TotalFaults,
		L3BreakerOpenSkips:         bStats.OpenSkips,
		L3BreakerProbesAttempted:   bStats.ProbesAttempted,
		L3BreakerProbeRecoveries:   bStats.ProbeRecoveries,
		L3BreakerProbeFailures:     bStats.ProbeFailures,
		L3BreakerOpenedAt:          bStats.OpenedAt,
		EvictionPolicy:             t.evictionStrategy().Name(),
		LastEvictPolicy:            t.lastEvictPolicyName(),
		LastEvictCandidates:        t.lastEvictCandidates,
		LastEvictLocked:            t.lastEvictLocked,
		LastEvictVictimCost:        t.lastEvictVictimCost,
		LastEvictVictimHits:        t.lastEvictVictimHits,
		LastEvictVictimTokens:      t.lastEvictVictimTokens,
		LastEvictVictimPrefix:      t.lastEvictVictimPrefix,
		ThrashReuses:               t.thrashReuses,
		ThrashTokens:               t.thrashTokens,
		ThrashGapTotal:             t.thrashGapTotal,
		ThrashGapLast:              t.thrashGapLast,
		ThrashGapMax:               t.thrashGapMax,
		ThrashTracked:              len(t.thrashIndex),
		BoundedEvictions:           t.boundedEvictions,
		LedgerConfirmed:            t.ledgerConfirmed,
		LedgerConfirmedTokens:      t.ledgerConfirmedTokens,
		LedgerPending:              len(t.evictLedger),
		Lookups:                    t.lookups,
		LookupHits:                 t.lookupHits,
		LookupMissCold:             t.lookupMissCold,
		LookupMissDivergent:        t.lookupMissDivergent,
		Fills:                      t.fills,
		AdmissionEnabled:           t.admissionEnabled,
		AdmissionSketchCells: func() int {
			if t.admissionSketch == nil {
				return 0
			}
			return t.admissionSketch.Cells()
		}(),
		AdmissionObservations:       t.admissionObservations,
		AdmissionCandidates:         t.admissionCandidates,
		AdmissionAdmitted:           t.admissionAdmitted,
		AdmissionRejected:           t.admissionRejected,
		AdmissionRejectedTokens:     t.admissionRejectedTokens,
		AdmissionRejectedBytes:      t.admissionRejectedBytes,
		AdmissionTelemetryFallbacks: t.admissionTelemetryFallbacks,
		AdmissionHotProtected:       t.admissionHotProtected,
		AdmissionRecoveries:         t.admissionRecoveries,
		AdmissionJournalPending:     t.admissionJournalPending(),
		AdmissionJournalDropped:     t.admissionJournalDropped,
		AdmissionRecoveryGapLast:    t.admissionRecoveryGapLast,
		AdmissionRecoveryGapMax:     t.admissionRecoveryGapMax,
		LastAdmissionFrequency:      t.lastAdmissionFrequency,
		LastAdmissionReason:         t.lastAdmissionReason,
	}
	var visit func(n *node)
	visit = func(n *node) {
		if n.parent != nil { // skip every namespace root (parent==nil); count real nodes once
			s.Nodes++
			s.Tokens += len(n.key)
			if n.kv != nil {
				s.PrefixTokens += n.plen
			}
			if n.snapshot != nil {
				host, device := n.snapshot.ResidencyBytes()
				s.DeviceSnapshotHostBytes += host
				s.DeviceSnapshotBytes += device
				if device > 0 {
					s.DeviceSnapshotTokens += n.plen
				}
			}
			if n.plen > s.MaxDepthTokens {
				s.MaxDepthTokens = n.plen
			}
			s.ReuseHits += n.hits
			if len(n.children) == 0 {
				s.Leaves++
			}
		}
		for _, c := range n.children {
			visit(c)
		}
	}
	t.forEachRoot(visit) // one Stats snapshot across every namespace's subtree
	return s
}
