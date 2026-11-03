// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Completion ownership adapted from vllm/v1/kv_offload/tiering/manager.py:281-358
// at bd5bbce6e45e09dbd07ac3e8c394626c059078f7; modified for Go synchronous
// snapshot acquisition: publish successful owners and release rejected owners.
//
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
// Faithfulness note (honest, per the repo's "labeled, not faked" rule): default and
// live-pointer nodes store the FULL-prefix KVCache for their root→node path, not SGLang's
// per-segment slabs in a paged pool. That costs more memory, but the metric this package
// exists to compare — cache hit rate / prefill-tokens-saved — is layout-independent, and
// storing full-prefix caches lets every reuse and every edge-split go through the *proven*
// Clone/Evict primitives unchanged (a split truncates a child's cache to the boundary via
// Evict-of-the-tail, which has no survivor to re-RoPE, so it is an exact prefix). We trade
// memory for the guarantee that not one bit of the verified KV core is re-derived here.
// Explicit clone-owned dense-FP32 selection instead shares immutable paged payloads;
// request boundaries still materialize private full-prefix Session caches. Unsupported
// layouts and the live-pointer InsertWithLogits contract retain the legacy path.
package radixkv

import (
	"context"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Node is an alias for node to make the tree node handle accessible to external callers.
type Node = node

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
	densePrefix *model.DensePrefixSnapshot // immutable clone-owned dense payload; nil on legacy insertion
	kv          *model.KVCache
	snapshot    *model.PrefixSnapshot // optional full device/hybrid prefix owner
	// hostSnapshot is an independently owned complete PrefixSnapshot image in
	// process-local host DRAM. It is not provider accounting: the image owns the
	// attention K/Kraw/V rows, positions, and hybrid recurrent state needed to
	// materialize a fresh backend snapshot after the hot owner is evicted.
	hostSnapshot *model.HostPrefixSnapshot
	// remoteSnapshot is a verified reference to a complete serialized
	// HostPrefixSnapshot in the configured l3kv/blobhttp tier. It owns no local
	// payload bytes and remains usable after both local physical copies are gone.
	remoteSnapshot *remoteSnapshotRef
	// ssdSnapshot is a verified reference to a complete serialized HostPrefixSnapshot
	// demoted to the local-SSD overflow tier when host L2 pressure evicted it. Like a
	// remote reference it owns no local payload bytes and keeps the record incarnation
	// alive after every in-memory copy is gone; its on-disk bytes count against the
	// tree's SSD budget while the reference is attached.
	ssdSnapshot *ssdSnapshotRef
	// logits is the final-token distribution for this exact full prefix, when the caller
	// provided it. It lets an exact-hit replay sample the first generated token without
	// refeeding the prompt's final token just to recover logits. Splits intentionally do
	// not derive logits for the shorter boundary; a full-prefix KV can be truncated, but
	// the final logits for an intermediate prefix are not available from a longer leaf.
	logits       []float32
	cachedLogits []float32 // logits owned by a complete device snapshot

	plen          int    // path length in tokens (parent.plen + len(key)); == len(kv) when kv!=nil
	refs          int    // active leases; a leaf with refs>0 is never LRU-evicted
	lastUsed      uint64 // logical clock of the most recent match/insert touching this node — LRU key
	createdAt     uint64 // tree-scoped node creation order, unchanged by later touches
	opportunistic bool   // admitted by prewarm; stays low priority until a demand touch
	hits          int    // subsequent demand lookups that found this node resident
	chunkID       int    // physical backing page / allocation chunk identifier (0 = unassigned)
	regimeKey     string // decode regime identifier under which this node's KV was produced

	pinned    bool              // Tier 0 coordinator prompt: immune from eviction (seg = 3)
	retention *RetentionRequest // client-declared per-request KV retention descriptor
	tier      PriorityTier      // explicit priority tier (Tier0PinnedRoot .. Tier3Probationary)
	tierSet   bool              // true if tier was explicitly assigned

	state     uint32        // lifecycle state (warm, computing prefill, failed, evicted); accessed atomically
	ready     chan struct{} // completion broadcast for in-flight prefill promises
	flightErr error         // terminal error if in-flight prefill failed or was abandoned

	// recordGen is this node's local record incarnation: a monotonically increasing
	// tree-scoped sequence stamped when InsertSnapshot ADMITS a complete local
	// snapshot record, and cleared (with hasRecord) when the last complete local
	// copy is logically evicted. A zero recordGen with hasRecord=false means the node
	// owns no live record (a structural intermediate or a pure KV node). The pair is
	// the radixkv analogue of ctxmmu's opaque monotonic allocation-incarnation handle
	// (#12848): it never moves across a split, so a handle can never be inherited by a
	// node that does not own the snapshot. Mutation is externally serialized, like the
	// tree's other counters, so plain fields (no mutex) are sufficient.
	recordGen uint64
	hasRecord bool
}

// RecordHandle is a node-owned incarnation handle for a complete local snapshot
// record. It is minted only when a complete PrefixSnapshot is admitted at a node
// and is invalidated by replacement or by logical eviction of the record's last
// complete local copy. A handle minted for a node whose record is later replaced
// or evicted no longer validates.
type RecordHandle struct {
	node *node
	gen  uint64
}

// Generation returns the monotonic record incarnation stamped into the handle.
func (h RecordHandle) Generation() uint64 { return h.gen }

// RecordHandle returns this node's current live record handle. ok is false when the
// node owns no live admitted record (a structural intermediate, a pure KV node, or a
// node whose record was replaced or logically evicted).
func (n *node) RecordHandle() (RecordHandle, bool) {
	if n == nil || !n.hasRecord || n.recordGen == 0 {
		return RecordHandle{}, false
	}
	return RecordHandle{node: n, gen: n.recordGen}, true
}

var closedReadyChan = func() <-chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}()

// State returns the current lifecycle state of this node.
func (n *node) State() NodeState {
	if n == nil {
		return NodeWarm
	}
	return NodeState(atomic.LoadUint32(&n.state))
}

// SetState updates the lifecycle state of this node atomically.
func (n *node) SetState(s NodeState) {
	if n != nil {
		atomic.StoreUint32(&n.state, uint32(s))
	}
}

// Ready returns a channel that is closed when prefill computation finishes (or immediately if already warm).
func (n *node) Ready() <-chan struct{} {
	if n == nil || n.ready == nil {
		return closedReadyChan
	}
	return n.ready
}

// WaitReady blocks until prefill computation for this node completes or ctx is cancelled.
func (n *node) WaitReady(ctx context.Context) error {
	if n == nil {
		return nil
	}
	select {
	case <-n.Ready():
		if n.flightErr != nil {
			return n.flightErr
		}
		if n.State() == NodeFailed {
			return ErrFlightAbandoned
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// FlightErr returns the error recorded if prefill failed during in-flight computation.
func (n *node) FlightErr() error {
	if n == nil {
		return nil
	}
	return n.flightErr
}

// IsComputing reports whether this node is currently undergoing prefill computation.
func (n *node) IsComputing() bool {
	return n != nil && n.State() == NodeComputingPrefill
}

// IsWarm reports whether this node is resident and has completed prefill.
func (n *node) IsWarm() bool {
	return n != nil && n.State() == NodeWarm
}

// CloneKV returns an independently cloneable handle to the node's full-prefix KV cache.
func (n *node) CloneKV() *model.KVCache {
	if n == nil {
		return nil
	}
	if n.densePrefix != nil {
		return n.densePrefix.Materialize()
	}
	if n.kv == nil {
		return nil
	}
	return n.kv.Clone()
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
	EvictionPriority
)

func (p EvictionPolicy) String() string {
	switch p {
	case EvictionCostAware:
		return "cost-aware"
	case EvictionPageAware:
		return "page-aware"
	case EvictionPriority:
		return "priority"
	default:
		return "lru"
	}
}

// EvictionHaltPredicate returns true when budget eviction should halt early
// because shared allocation capacity is already sufficient.
type EvictionHaltPredicate func() bool

// Tree is a RadixAttention prefix cache: a radix tree of token sequences with
// longest-prefix matching, an LRU token budget, and reference counting.
type Tree struct {
	maxCPUCacheBytes    int64
	sharedDensePrefixes bool // construction-time opt-in; live-pointer insertion unchanged
	cpuCacheBypasses    int64
	cpuCacheLastBypass  string
	root                *node
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
	creationSeq          uint64 // node creation clock, independent of demand/warm recency
	recordSeq            uint64 // monotonic, never-reused record incarnation sequence (#12848)

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

	remoteSnapshotStore    SnapshotStore
	remoteConfigGeneration uint64               // changes on every store/backend/config attach or detach
	remoteTransfers        *remoteTransferState // serialized pending ownership; no background worker
	remoteModelID          string
	remoteBackend          compute.Backend
	remoteConfig           model.Config
	remoteSnapshotBytes    int64
	remoteBreaker          *RemoteL3Breaker

	localSSDStore         SnapshotStore
	localSSDModelID       string
	localSSDBackend       compute.Backend
	localSSDConfig        model.Config
	localSSDBudget        int64
	localSSDSnapshotBytes int64
	l3LocalHits           int
	l3LocalMisses         int
	l3LocalFaults         int
	l3LocalHitTokens      int
	l3LocalStageBytes     int64
	l3LocalRestoreBytes   int64
	l3Hits                int
	l3Misses              int
	l3Faults              int
	l3HitTokens           int
	l3StageBytes          int64
	l3RestoreBytes        int64
	l3StageNanos          int64
	l3RestoreNanos        int64
	l3StageFaults         int
	l3RestoreFaults       int

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

	// Halt eviction early when shared allocation capacity is sufficient (#12414).
	haltPredicate                  EvictionHaltPredicate
	evictionHaltSufficientCapacity int64
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

// Retention reports the signed retention dial: negative keeps all, zero keeps
// none, and positive bounds cached tokens. Legacy unbounded budgets map to -1.
// Unlike Stats.MaxTokens, this preserves the distinction needed to restore a
// policy after a temporary pressure trim.
func (t *Tree) Retention() int {
	if t.retentionSet {
		return t.retention
	}
	if t.maxTokens > 0 {
		return t.maxTokens
	}
	return -1
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
	t.creationSeq++
	mid := &node{
		key:           append([]int(nil), child.key[:oi]...),
		parent:        parent,
		children:      map[int]*node{},
		plen:          parent.plen + oi,
		lastUsed:      child.lastUsed,
		createdAt:     t.creationSeq,
		opportunistic: child.opportunistic,
		hits:          child.hits,
		chunkID:       child.chunkID,
		regimeKey:     child.regimeKey,
		pinned:        child.pinned,
		tier:          child.tier,
		tierSet:       child.tierSet,
	}
	// Record ownership stays with the snapshot-bearing CHILD: mid is a fresh
	// structural intermediate that owns no snapshot, so it must never inherit the
	// child's record incarnation (#12848). child's recordGen/hasRecord are left
	// untouched and remain valid after the edge is re-parented.
	if child.retention != nil {
		retCopy := *child.retention
		mid.retention = &retCopy
	}
	if child.densePrefix != nil {
		if bytes, ok := child.densePrefix.PrefixMetadataBytes(mid.plen); ok && t.cpuCacheRoomWithoutEviction(bytes) {
			mid.densePrefix = child.densePrefix.ForkPrefix(mid.plen)
		}
	} else if child.kv != nil && child.kv.CanEvict() == nil && t.cpuCacheCanClone(child.kv) {
		mid.kv = truncatePrefix(child.kv, mid.plen)
	}
	if child.IsComputing() {
		mid.SetState(NodeComputingPrefill)
		mid.ready = child.ready
		mid.flightErr = child.flightErr
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

// InsertWithTier attaches suffix to boundary, assigns priority tier to the new leaf,
// and enforces the budget.
func (t *Tree) InsertWithTier(boundary *node, suffix []int, kv *model.KVCache, tier PriorityTier) *node {
	leaf := t.Insert(boundary, suffix, kv)
	if leaf != nil {
		t.SetNodeTier(leaf, tier)
	}
	return leaf
}

// InsertWithRetention attaches suffix to boundary, assigns client-declared retention descriptor
// to the new leaf, and enforces the budget.
func (t *Tree) InsertWithRetention(boundary *node, suffix []int, kv *model.KVCache, req RetentionRequest) (*node, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	leaf := t.Insert(boundary, suffix, kv)
	if leaf != nil {
		if err := t.SetNodeRetention(leaf, req); err != nil {
			return leaf, err
		}
	}
	return leaf, nil
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

func acquireSnapshotForRequest(ctx context.Context, acquire func() (*model.PrefixSnapshot, error)) (*model.PrefixSnapshot, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	snap, err := acquire()
	if canceled := ctx.Err(); canceled != nil {
		err = canceled
	}
	if err != nil {
		if snap != nil {
			snap.Close()
		}
		return nil, err
	}
	return snap, nil
}

// LookupSnapshotTieredContext is the request-cancellable L1→L2→L3 lookup.
func (t *Tree) LookupSnapshotTieredContext(ctx context.Context, tokens []int) (*node, *model.PrefixSnapshot, int, SnapshotTier, error) {
	return t.lookupSnapshotTieredContext(ctx, tokens, nil)
}

func (t *Tree) lookupSnapshotTieredContext(ctx context.Context, tokens []int, lock *remoteTransferLock) (*node, *model.PrefixSnapshot, int, SnapshotTier, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, 0, SnapshotTierMiss, err
	}
	n, _ := t.Lookup(tokens)
	for candidate := n; candidate != nil; candidate = candidate.parent {
		if candidate.snapshot == nil {
			continue
		}
		snap, err := acquireSnapshotForRequest(ctx, candidate.snapshot.CloneForRestore)
		if err != nil {
			t.l1Faults++
			return n, nil, candidate.plen, SnapshotTierDeviceL1, err
		}
		t.l1Hits++
		t.l1HitTokens += candidate.plen
		return n, snap, candidate.plen, SnapshotTierDeviceL1, nil
	}
	t.l1Misses++
	if !t.HostL2Enabled() && !t.RemoteSnapshotEnabled() && !t.LocalSSDSnapshotEnabled() {
		return n, nil, 0, SnapshotTierMiss, nil
	}
	for candidate := n; candidate != nil; candidate = candidate.parent {
		if candidate.hostSnapshot == nil {
			continue
		}
		snap, err := acquireSnapshotForRequest(ctx, candidate.hostSnapshot.Restore)
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
	// The local-SSD overflow sits below host DRAM but above the remote tier: it holds exactly
	// the images L2 pressure demoted, so a node whose hot and host copies are gone can still be
	// restored from disk without crossing the network.
	if t.LocalSSDSnapshotEnabled() {
		for candidate := n; candidate != nil; candidate = candidate.parent {
			if candidate.ssdSnapshot == nil {
				continue
			}
			snap, found, err := t.restoreSnapshotFromLocalSSD(ctx, candidate)
			if err != nil {
				return n, nil, candidate.plen, SnapshotTierLocalL3, err
			}
			if found {
				return n, snap, candidate.plen, SnapshotTierLocalL3, nil
			}
		}
		t.l3LocalMisses++
	}
	if t.RemoteSnapshotEnabled() {
		for candidate := n; candidate != nil; candidate = candidate.parent {
			if candidate.remoteSnapshot == nil {
				continue
			}
			matched := candidate.plen
			snap, found, err := t.restoreSnapshotFromRemoteWithLock(ctx, "", candidate, lock)
			if lock != nil && lock.released {
				// The transfer may have kept an ancestor record valid while an
				// administrator detached or replaced our original lookup boundary.
				// Drop only our lookup lease and reacquire a current structural
				// boundary. Never walk the old ancestry after an unlocked callback.
				t.Done(n)
				n, _ = t.Lookup(tokens)
				if err != nil {
					return n, nil, matched, SnapshotTierRemoteL3, err
				}
				if found {
					// The independently owned snapshot still matches the captured
					// prefix; job validation proved its record and token count.
					return n, snap, matched, SnapshotTierRemoteL3, nil
				}
				t.l3Misses++
				return n, nil, 0, SnapshotTierMiss, nil
			}
			if err != nil {
				return n, nil, matched, SnapshotTierRemoteL3, err
			}
			if found {
				return n, snap, matched, SnapshotTierRemoteL3, nil
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
	// Seal only after admission succeeds and before exposing the new owner.
	// Preparation allocates host metadata before retiring the previous snapshot.
	snap.PrepareForCache()
	if n.snapshot != nil && n.snapshot != snap {
		n.snapshot.Close()
	}
	t.snapshotBytes -= oldBytes
	n.snapshot = snap
	// Staged tiers belong to the previous admission, even when the token key
	// matches. Retire them only after admission succeeds and the new hot owner
	// is installed, so rejection preserves every old copy and same-owner
	// re-admission cannot close the owner that remains live.
	t.releaseHostSnapshot(n)
	t.releaseRemoteSnapshot(n)
	t.releaseSSDSnapshot(n)
	n.cachedLogits = append([]float32(nil), logits...)
	t.snapshotBytes += incoming
	// Successful admission of a complete local snapshot record mints a fresh
	// incarnation. Re-admission on the same node (even with identical tokens) mints
	// a NEW generation, which naturally invalidates every previously issued handle.
	t.mintRecord(n)
	return n, nil
}

// mintRecord stamps a fresh, never-reused record incarnation onto n. Called only
// from InsertSnapshot's admitted branch, after the snapshot is installed.
func (t *Tree) mintRecord(n *node) {
	if n == nil {
		return
	}
	t.recordSeq++
	n.recordGen = t.recordSeq
	n.hasRecord = true
}

// invalidateRecordIfNoLocalCopy clears n's record when NO complete local copy of
// its snapshot remains (hot L1, host L2, and remote L3 all absent). This is the
// boundary at which the record dies: a demotion that only moves the record between
// tiers (hot -> host) leaves a local complete copy, so the incarnation survives.
func (t *Tree) invalidateRecordIfNoLocalCopy(n *node) {
	if n == nil || !n.hasRecord {
		return
	}
	if n.snapshot == nil && n.hostSnapshot == nil && n.remoteSnapshot == nil && n.ssdSnapshot == nil {
		n.hasRecord = false
		n.recordGen = 0
	}
}

// ValidateRecord reports whether h is the CURRENT live incarnation of a record
// owned by a node that is still attached to t and still holds a complete local
// snapshot copy. A replaced, evicted, or detached record's handle returns false.
func (t *Tree) ValidateRecord(h RecordHandle) bool {
	if t == nil || h.node == nil || !h.node.hasRecord || h.gen == 0 || h.gen != h.node.recordGen {
		return false
	}
	if h.node.snapshot == nil && h.node.hostSnapshot == nil && h.node.remoteSnapshot == nil && h.node.ssdSnapshot == nil {
		return false
	}
	return t.nodeAttached(h.node)
}

// nodeAttached reports whether n is still linked into t's radix tree: walking up its
// parent links re-verifies each child slot and terminates at one of t's roots. A node
// removed by removeLeaf/EvictNode fails the child-slot check at its former parent.
func (t *Tree) nodeAttached(n *node) bool {
	if t == nil || n == nil {
		return false
	}
	for n.parent != nil {
		p := n.parent
		if len(n.key) == 0 || p.children[n.key[0]] != n {
			return false
		}
		n = p
	}
	if n == t.root {
		return true
	}
	for _, r := range t.nsRoots {
		if r == n {
			return true
		}
	}
	return false
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
	return t.snapshotVictimSkipping(exclude, nil)
}

// snapshotEvictionView projects only the insertion tick and the one lease a
// nonempty suffix transfers off its lookup boundary. Prediction never changes
// the live clock, refs, topology or snapshot ownership. Structural strategy
// preparation and intervening token-budget eviction are not simulated here.
type snapshotEvictionView struct {
	clock    uint64
	released *node
}

func (v snapshotEvictionView) refs(n *node) int {
	refs := n.refs
	if n == v.released && refs > 0 {
		refs--
	}
	return refs
}

// snapshotVictimSkipping applies the same retention rules to real byte-pressure
// eviction and the admission dry run. skipped snapshots have already been selected
// by the dry run; their nodes remain attached, just as releaseHotSnapshot leaves them.
// Priority-ordered per-tier reclaim follows NVIDIA/TensorRT-LLM, Apache-2.0,
// cpp/tensorrt_llm/batch_manager/evictionPolicy.cpp at
// f4c5c935aa891b0826f73936c4831236cb6ff836 (LRUEvictionPolicy::getFreeBlock/refresh).
// This Go adaptation keeps Fak's existing pin, tier and logical-TTL contracts.
func (t *Tree) snapshotVictimSkipping(exclude *node, skipped map[*node]bool) *node {
	return t.snapshotVictimSkippingInView(exclude, skipped, snapshotEvictionView{clock: t.clock})
}

func (t *Tree) snapshotVictimSkippingInView(exclude *node, skipped map[*node]bool, view snapshotEvictionView) *node {
	strat := t.evictionStrategy()
	if prep, ok := strat.(TreePreparer); ok {
		prep.PrepareTree(t)
	}
	// Deepest-first: a snapshot is a victim candidate only when no descendant holds an
	// evictable snapshot. An ancestor snapshot (e.g. a shared system/tools boundary)
	// restores every sibling below it, while a deeper per-request snapshot serves only
	// its own continuation, so the ancestor must outlive its descendants. Recurrent
	// state cannot be truncated, so a lost ancestor cannot be rebuilt from a leaf.
	// Among candidates the configured strategy picks as before.
	var victim *node
	var bestKey victimKey
	var walk func(*node) bool
	walk = func(n *node) bool {
		below := false
		for _, child := range n.children {
			if walk(child) {
				below = true
			}
		}
		evictable := n != exclude && !skipped[n] && view.refs(n) == 0 && !n.IsComputing() &&
			n.snapshot != nil && !t.isNodePinnedOrImmuneAt(n, view.clock)
		var key victimKey
		if evictable {
			key = strat.Priority(n)
			if seg, ok := t.nodeTierSegAt(n, view.clock); ok {
				key.seg = seg
			}
			evictable = key.seg < 3
		}
		if evictable && !below {
			// Snapshot record generations are unique across namespaces and survive
			// tier moves, giving exact priority ties a stable admission-order key.
			if victim == nil || key.less(bestKey) || (key == bestKey && n.recordGen < victim.recordGen) {
				victim, bestKey = n, key
			}
		}
		return below || evictable
	}
	t.forEachRoot(func(n *node) { walk(n) })
	return victim
}

func (t *Tree) releaseHotSnapshot(n *node) {
	if n == nil || n.snapshot == nil {
		return
	}
	t.snapshotBytes -= snapshotResidentBytes(n.snapshot, n.cachedLogits)
	n.snapshot.Close()
	n.snapshot = nil
	if n.hostSnapshot == nil && n.remoteSnapshot == nil && n.ssdSnapshot == nil {
		n.cachedLogits = nil
	}
	// The record follows the node across tiers: only drop the incarnation when this
	// was the last complete local copy (no staged host / remote reference remains).
	t.invalidateRecordIfNoLocalCopy(n)
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
	// A still-leased Lookup boundary can become stale while another request
	// inserts this edge. Keep any immutable dense owner reachable/accounted,
	// and preserve every lease on refusal. Legacy-only collisions are unchanged.
	if len(suffix) > 0 && hasDenseSubtree(boundary.children[suffix[0]]) {
		return boundary, false
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
		if logits != nil && boundary.HasKV() {
			if !t.makeCPUCacheRoom(cpuLogitsBytes(len(logits)), cpuLogitsBytes(cap(boundary.logits)), boundary) {
				return boundary, false
			}
			boundary.logits = copyCPULogits(logits)
		}
		if chunkID != 0 {
			boundary.chunkID = chunkID
		}
		t.noteAdmissionRecovery(keyHash)
		return boundary, true // already fully cached; keep the boundary lease for the caller to Done
	}
	if t.maxCPUCacheBytes > 0 && !t.makeCPUCacheRoom(cacheByteSum(kv.OwnedPayloadBytes(), cpuLogitsBytes(len(logits))), 0, boundary) {
		return boundary, false
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
	room := true
	if t.maxCPUCacheBytes > 0 {
		incoming := cacheByteSum(kv.OwnedPayloadBytes(), cpuLogitsBytes(len(logits)))
		if lastUsed == warmRecency {
			room = t.cpuCacheRoomWithoutEviction(incoming)
		} else {
			room = t.makeCPUCacheRoom(incoming, 0, boundary)
		}
	}
	if !room {
		kv, logits = nil, nil
	}
	// Thrash probe (#3393): the new leaf's full path is (root→boundary)+suffix — if that
	// exact key was just evicted, this attach is the re-insert that proves the eviction
	// premature. Covers both demand Insert and WarmInsert; a Lookup-consumed entry is
	// simply absent here (consumed once).
	t.probeThrash(boundary, suffix)
	t.fills++ // one prefix-cache FILL event (#5804): demand Insert or prewarm WarmInsert
	s := append([]int(nil), suffix...)
	t.creationSeq++
	leaf := &node{
		key:           s,
		parent:        boundary,
		children:      map[int]*node{},
		kv:            kv,
		logits:        copyCPULogits(logits),
		plen:          boundary.plen + len(s),
		lastUsed:      lastUsed,
		createdAt:     t.creationSeq,
		opportunistic: lastUsed == warmRecency,
		chunkID:       chunkID,
		regimeKey:     boundary.regimeKey,
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

// KV preserves the live-pointer contract of InsertWithLogits. An explicitly
// clone-owned dense snapshot instead materializes a private cache. Use HasKV
// for presence and CloneKV for one private materialization at a request boundary.
func (n *node) KV() *model.KVCache {
	if n.densePrefix != nil {
		return n.densePrefix.Materialize()
	}
	return n.kv
}

// Logits returns a copy of the final-token logits for this exact full prefix, or nil when
// the caller did not cache them. The copy keeps request-local sampling penalties/biases
// from mutating the shared tree payload.
func (n *node) Logits() []float32 {
	if n == nil {
		return nil
	}
	if n.snapshot != nil || n.hostSnapshot != nil || n.remoteSnapshot != nil || n.ssdSnapshot != nil {
		return append([]float32(nil), n.cachedLogits...)
	}
	return append([]float32(nil), n.logits...)
}

// Plen is the node's cached prefix length in tokens.
func (n *node) Plen() int { return n.plen }

// RegimeKey returns the decode regime under which this node was cached, or "" if unassigned.
func (n *node) RegimeKey() string {
	if n == nil {
		return ""
	}
	return n.regimeKey
}
