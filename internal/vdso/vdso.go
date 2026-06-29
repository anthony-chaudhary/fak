// Package vdso is the tool vDSO: a 3-tier local fast path that answers a tool
// call with NO engine and NO remote round-trip — the agentic analogue of the
// kernel vDSO that serves gettimeofday() from userspace without a syscall.
//
// Three tiers, consulted cheapest-first (registered at ascending FastPath tiers):
//
//	tier 1  pure registry   — the result is a pure function of args; gated on
//	                          readOnlyHint+idempotentHint, re-checked not trusted.
//	tier 2  content cache    — keyed on (tool, args-sha256, world-version); filled
//	                          from EvComplete events; a world bump invalidates it.
//	tier 3  static table     — canned answers for static tools.
//
// Soundness invariant (unit 38, also steward vdso-soundness): a cache hit equals
// a fresh call. Tier-2 enforces it by binding every key to the world-version and
// bumping the version on any write-shaped completion (so a stale read can never
// be served). The vDSO also registers an Emitter to observe completions — that is
// how the cache fills and the world advances, with no new ABI seam.
package vdso

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// DefaultCacheSize is the tier-2 LRU capacity (unit 35 RSI tweak target).
const DefaultCacheSize = 1024

// DefaultContentCacheSize is the tier-4 content dedup cache capacity.
const DefaultContentCacheSize = 512

// DefaultNodeEpochLimit bounds the finer-eraser epoch table. When pressure evicts
// a node epoch, the vDSO bumps the root epoch so old keys cannot become reachable.
const DefaultNodeEpochLimit = 8192

// DefaultRevokedWitnessLimit bounds the exact refuted-witness ledger. On overflow,
// unknown witness-bearing entries fail closed (revoke.go) rather than growing the
// ledger without bound or re-admitting stale CAS bytes.
const DefaultRevokedWitnessLimit = 8192

// PureFunc computes a tool's result purely from its argument bytes.
type PureFunc func(args []byte) (result []byte, ok bool)

// VDSO holds all three tiers + counters. The package registers one Default.
type VDSO struct {
	mu sync.Mutex

	pure   map[string]PureFunc // tier 1
	static map[string][]byte   // tier 3

	// shareable (principal.go) names tools whose result is identity-INDEPENDENT
	// public knowledge (e.g. a shared policy doc): for them the per-principal cache
	// dimension is dropped so the entry is shared ACROSS principals — the opt-in
	// cross-tenant win. Guarded by v.mu, like pure/static: written at registration,
	// read in keyLocked (which already holds v.mu). A nil map reads as "none".
	shareable map[string]bool

	cap      int
	cache    map[string]*list.Element // tier 2: key -> LRU node
	lru      *list.List               // front = most-recent
	worldVer uint64                   // the root ("*") epoch — lock-free Global hot path

	// contentCache is the tier-4 content dedup cache: a map of content hash to Ref.
	// It stores inbound content blocks (tool results, user messages) so identical
	// content sent multiple times in one request is stored once and reused.
	contentCache      map[string]abi.Ref // tier 4: content hash -> Ref
	contentCacheSize  int
	contentCacheLRU   *list.List
	contentCacheIndex map[string]*list.Element

	// Finer-eraser state (scope.go). gran selects how broadly a write invalidates;
	// nodes holds the per-namespace / per-entity epochs the hierarchical key binds;
	// subs is the coherence-bus subscriber set. All zero-valued => Global behavior.
	gran      Granularity
	nearDup   int32             // neardup.go: 1 => collapse formatting-variant args (opt-in)
	nodes     map[string]uint64 // tag (depth>=1) -> epoch; root lives in worldVer
	nodeCap   int
	nodeLRU   *list.List
	nodeIndex map[string]*list.Element
	subs      subList[Mutation] // coherence-bus observers of write mutations
	subSeq    uint64            // subscriber id allocator (mutation + revocation subs)
	mutSeq    uint64            // monotone coherence-bus sequence (one order over writes+refutations)
	mutations int64             // write-shaped completions observed (bus event count)

	// Integrity-direction eraser (revoke.go) — the mutable TRUST EPOCH layered over the
	// durable, content-addressed CAS. revoked names the external world-state witnesses a
	// refutation has retired (no entry admitted under one may be served or re-admitted);
	// trustEpoch is the monotone integrity clock (dual of worldVer); revSubs is the
	// integrity-bus subscriber set. All zero-valued => no revocation has ever fired.
	revoked         map[string]uint64 // refuted external witnesses -> trust epoch
	revokedCap      int
	revokedLRU      *list.List
	revokedIndex    map[string]*list.Element
	revokedOverflow bool   // exact ledger overflowed; unknown witnesses fail closed
	trustEpoch      uint64 // monotone refutation epoch — dual of worldVer
	revSubs         subList[Revocation]
	revocations     int64 // refutations observed (integrity-bus event count)

	// cachemeta emission (§2.5). cacheSink observes tier-2 lifecycle events as
	// cachemeta entries; witnessAdapters are per-tool external-witness extractors.
	// Both are opt-in (nil/empty = unchanged behavior) and dispatched outside v.mu.
	// regMu guards these two registries so emit/witness reads stay race-free without
	// contending v.mu on the hot Lookup path.
	regMu           sync.RWMutex
	cacheSink       func(CacheEvent)
	witnessAdapters map[string]WitnessFunc

	lookups int64
	hits    int64
	fills   int64

	// contentHits and contentFills track the content dedup tier-4 metrics.
	contentHits  int64
	contentFills int64

	// missCtr attributes every ok=false to the reason that produced it, so a low
	// hit rate is explainable ("is it write-shaped tools, missing hints, or churn?")
	// instead of collapsing to a bare miss. Lock-free: each is bumped at the exact
	// early-return that decided the miss. See MissReasons + the Miss* constants.
	missCtr struct {
		destructive      int64
		missingHints     int64
		resourceMisnamed int64
		witnessRevoked   int64
		notCached        int64
	}
}

// The closed vocabulary of vDSO miss reasons (the WHY behind ok=false):
//   - MissDestructive:      the tool is write-shaped/destructive, so it is never
//     fast-path eligible (a cached read of it would be unsound).
//   - MissMissingHints:     the call lacks readOnlyHint/idempotentHint, so the
//     soundness gate cannot prove it is cacheable.
//   - MissResourceMisnamed: a read that cannot name its entity (no fine-grained
//     write could invalidate it) — refused to the engine for soundness.
//   - MissWitnessRevoked:   a cached entry was admitted under a now-refuted
//     external witness; it is evicted and treated as a miss.
//   - MissNotCached:        cacheable, but no tier-1/tier-3/tier-2 entry answered
//     (never filled, or stranded by a write that bumped its epoch).
const (
	MissDestructive      = "DESTRUCTIVE"
	MissMissingHints     = "MISSING_HINTS"
	MissResourceMisnamed = "RESOURCE_MISNAMED"
	MissWitnessRevoked   = "WITNESS_REVOKED"
	MissNotCached        = "NOT_CACHED"
)

// missed records a lookup miss by reason, emits the first-class cachemeta MISS event
// (§2.5; a no-op unless a cache sink is installed), and returns the (nil, false) a
// Lookup caller expects — so each early-return reads `return v.missed(c, MissX)`. It
// takes the consuming call so the emitted miss is attributed to its agent/turn,
// symmetric with the consumer attribution on a hit. Every call site is OUTSIDE v.mu.
func (v *VDSO) missed(c *abi.ToolCall, reason string) (*abi.Result, bool) {
	switch reason {
	case MissDestructive:
		atomic.AddInt64(&v.missCtr.destructive, 1)
	case MissMissingHints:
		atomic.AddInt64(&v.missCtr.missingHints, 1)
	case MissResourceMisnamed:
		atomic.AddInt64(&v.missCtr.resourceMisnamed, 1)
	case MissWitnessRevoked:
		atomic.AddInt64(&v.missCtr.witnessRevoked, 1)
	default:
		atomic.AddInt64(&v.missCtr.notCached, 1)
	}
	v.emitMiss(c, reason)
	return nil, false
}

// gateMiss attributes a miss reached without a cache hit: a write-shaped tool and
// a hint-less call are distinguished from a genuine not-cached miss, so the
// dominant cause of a low hit rate is legible.
func (v *VDSO) gateMiss(c *abi.ToolCall) (*abi.Result, bool) {
	switch {
	case destructive(c):
		return v.missed(c, MissDestructive)
	case !metaTrue(c, "readOnlyHint") || !metaTrue(c, "idempotentHint"):
		return v.missed(c, MissMissingHints)
	default:
		return v.missed(c, MissNotCached)
	}
}

// MissReasons returns a cumulative snapshot of vDSO lookup misses by reason — the
// WHY behind every ok=false, rendered as fak_vdso_misses_total{reason}.
func (v *VDSO) MissReasons() map[string]uint64 {
	return map[string]uint64{
		MissDestructive:      uint64(atomic.LoadInt64(&v.missCtr.destructive)),
		MissMissingHints:     uint64(atomic.LoadInt64(&v.missCtr.missingHints)),
		MissResourceMisnamed: uint64(atomic.LoadInt64(&v.missCtr.resourceMisnamed)),
		MissWitnessRevoked:   uint64(atomic.LoadInt64(&v.missCtr.witnessRevoked)),
		MissNotCached:        uint64(atomic.LoadInt64(&v.missCtr.notCached)),
	}
}

type entry struct {
	key     string
	ref     abi.Ref
	witness string // external world-state witness this entry was admitted under ("" = none)
}

// New builds a vDSO with the given tier-2 capacity.
func New(capacity int) *VDSO {
	if capacity <= 0 {
		capacity = DefaultCacheSize
	}
	return &VDSO{
		pure:              map[string]PureFunc{},
		static:            map[string][]byte{},
		shareable:         map[string]bool{},
		cap:               capacity,
		cache:             map[string]*list.Element{},
		lru:               list.New(),
		nodes:             map[string]uint64{},
		nodeCap:           DefaultNodeEpochLimit,
		nodeLRU:           list.New(),
		nodeIndex:         map[string]*list.Element{},
		revoked:           map[string]uint64{},
		revokedCap:        DefaultRevokedWitnessLimit,
		revokedLRU:        list.New(),
		revokedIndex:      map[string]*list.Element{},
		contentCache:      map[string]abi.Ref{},
		contentCacheSize:  DefaultContentCacheSize,
		contentCacheLRU:   list.New(),
		contentCacheIndex: map[string]*list.Element{},
	}
}

// RegisterPure adds a tier-1 pure tool.
func (v *VDSO) RegisterPure(tool string, f PureFunc) {
	v.mu.Lock()
	v.pure[tool] = f
	v.mu.Unlock()
}

// RegisterStatic adds a tier-3 static answer.
func (v *VDSO) RegisterStatic(tool string, answer []byte) {
	v.mu.Lock()
	v.static[tool] = append([]byte(nil), answer...)
	v.mu.Unlock()
}

// Caps advertises nothing special.
func (v *VDSO) Caps() []abi.Capability { return nil }

// argHash content-addresses args. It canonicalizes JSON first (json.Marshal of a
// decoded map emits sorted keys), so two orderings of the same object hash to the
// SAME key (unit 26). Non-JSON args fall back to a raw-bytes hash.
func argHash(b []byte) string {
	if canon, ok := canonicalJSON(b); ok {
		b = canon
	}
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])[:24]
}

// canonicalJSON re-encodes a JSON value with map keys sorted (encoding/json sorts
// object keys on marshal), giving an order-independent canonical form.
func canonicalJSON(b []byte) ([]byte, bool) {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return nil, false
	}
	out, err := json.Marshal(v)
	if err != nil {
		return nil, false
	}
	return out, true
}

func metaTrue(c *abi.ToolCall, k string) bool {
	if c.Meta == nil {
		return false
	}
	return c.Meta[k] == "true"
}

// writeShapeNeedles are the tool-NAME substrings that mark a call write-shaped
// regardless of its hints (unit 32). This is the kernel's authoritative heuristic
// for "a name that smells like it mutates the world"; it is the SINGLE definition
// reused by the runtime override below AND by the static tool linter
// (internal/toollint), so the call-time veto and the definition-time diagnostic
// can never drift. It is a deliberate over-approximation: a read-only tool whose
// name merely contains one of these (e.g. "rundown_report") is treated as
// write-shaped and excluded from the fast path — the linter exists to surface
// exactly that name/hint tension once, at definition time.
var writeShapeNeedles = []string{"write", "edit", "delete", "patch", "exec", "run", "book", "update", "cancel", "send"}

// IsWriteShaped reports whether a tool NAME is write-shaped by the kernel's
// authoritative heuristic. Exported so the static tool linter predicts the exact
// fast-path veto the runtime applies, from one shared rule.
func IsWriteShaped(tool string) bool {
	t := strings.ToLower(tool)
	for _, p := range writeShapeNeedles {
		if strings.Contains(t, p) {
			return true
		}
	}
	return false
}

// WriteShapeNeedles returns a copy of the write-shape name substrings (forensics /
// linter messages). The slice is a copy; mutating it does not affect the kernel.
func WriteShapeNeedles() []string { return append([]string(nil), writeShapeNeedles...) }

// destructive re-checks the routing input the model gave us (unit 32): an
// annotation routes a call to the vDSO, but a write-shaped name or an explicit
// destructive flag OVERRIDES the hint — we never trust the annotation alone.
func destructive(c *abi.ToolCall) bool {
	if metaTrue(c, "destructive") {
		return true
	}
	return IsWriteShaped(c.Tool)
}

func (v *VDSO) bytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

func (v *VDSO) put(ctx context.Context, b []byte, taint abi.TaintLabel) abi.Ref {
	if res := abi.ActiveResolver(); res != nil {
		if ref, err := res.Put(ctx, b); err == nil {
			ref.Taint = taint
			return ref
		}
	}
	return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b)), Taint: taint}
}

// servedTaint computes the taint of a LOCALLY-computed serve, closing the vDSO's
// one-hop taint-laundering hole (unit 32 dual). A tier-1 pure result is at most as
// trustworthy as its ARGS — a pure function cannot wash taint off its input. But
// abi.TaintTainted is the enum ZERO value, so an unstamped Ref is indistinguishable
// from a deliberately-tainted one; trusting Tainted-or-unstamped would over-taint the
// many internal pure calls. We therefore DOWNGRADE only on POSITIVE proof of sealed
// data — Quarantined args — exactly the discipline the IFC SinkGate uses (ifc.go). So
// calculate{<quarantined arg>} can no longer emerge Trusted in one hop, while an
// ordinary unstamped pure call stays adjudicated-safe. Tier-3 static answers are
// args-independent (canned, kernel-provided) and pass TaintTrusted explicitly; tier-2
// hits never reach here — they serve the stored producer taint (no laundering).
func servedTaint(c *abi.ToolCall) abi.TaintLabel {
	if c != nil && c.Args.Taint == abi.TaintQuarantined {
		return abi.TaintQuarantined
	}
	return abi.TaintTrusted
}

// Lookup is the FastPath entry (unit 30: consulted before the adjudicator). It
// tries tier 1, then tier 3, then tier 2; a miss returns ok=false.
func (v *VDSO) Lookup(ctx context.Context, c *abi.ToolCall) (*abi.Result, bool) {
	atomic.AddInt64(&v.lookups, 1)

	// tier 1: pure registry, gated on read-only+idempotent and not destructive.
	if metaTrue(c, "readOnlyHint") && metaTrue(c, "idempotentHint") && !destructive(c) {
		v.mu.Lock()
		f, ok := v.pure[c.Tool]
		v.mu.Unlock()
		if ok {
			if out, served := f(v.bytes(ctx, c.Args)); served {
				return v.served(ctx, c, out, 1, servedTaint(c)), true
			}
		}
	}

	// tier 3: static table.
	v.mu.Lock()
	ans, ok := v.static[c.Tool]
	v.mu.Unlock()
	if ok {
		res := v.served(ctx, c, ans, 3, abi.TaintTrusted)
		// §2.5 tier-3 emission: a static-table serve is a first-class cachemeta hit,
		// attributed to the consuming agent/turn (consumerOpt is nil for an anonymous
		// call). Emitted OUTSIDE v.mu, after served() pinned the payload Ref.
		v.emitStaticHit(c, res.Payload, consumerOpt(c))
		return res, true
	}

	// tier 2: content-addressed cache, gated identically and world-versioned.
	if metaTrue(c, "readOnlyHint") && metaTrue(c, "idempotentHint") && !destructive(c) {
		args := v.bytes(ctx, c.Args)
		// Resource-mode soundness gate: refuse to serve a read that can't name its
		// entity (it would be invalidated by no entity-fine write) — go to the engine.
		if v.resourceMisnamed(c, args) {
			return v.missed(c, MissResourceMisnamed)
		}
		v.mu.Lock()
		key := v.keyLocked(c, args)
		if el, ok := v.cache[key]; ok {
			e := el.Value.(*entry)
			// Integrity gate (revoke.go): an entry admitted under a now-REFUTED external
			// witness is evicted on the spot and treated as a miss — the durable CAS would
			// otherwise re-serve the same poisoned bytes the consistency key still matches.
			// Sound: this only ever turns a hit into a miss (-> engine, fresh).
			if v.revokedLocked(e.witness) {
				v.lru.Remove(el)
				delete(v.cache, e.key)
				abi.UnpinResolved(e.ref) // left the cache (under v.mu) -> release its CAS pin
				rk, rref, rwit := e.key, e.ref, e.witness
				v.mu.Unlock()
				v.emitCache(CacheRevoke, rk, rref, rwit)
				return v.missed(c, MissWitnessRevoked)
			}
			v.lru.MoveToFront(el)
			ref := e.ref
			hk, href, hwit := e.key, e.ref, e.witness
			v.mu.Unlock()
			atomic.AddInt64(&v.hits, 1)
			// §2.5 consumer tracking: a HIT names the agent/turn that reused the entry
			// (consumerOpt is nil for an anonymous call, so no empty consumer is recorded).
			v.emitCache(CacheHit, hk, href, hwit, consumerOpt(c))
			return &abi.Result{Call: c, Payload: ref, Status: abi.StatusOK,
				Meta: map[string]string{"served_by": "vdso", "tier": "2"}}, true
		}
		v.mu.Unlock()
	}
	return v.gateMiss(c)
}

// served wraps a locally-computed tier-1/tier-3 answer. tierN tags which tier
// served it ("1" pure, "3" static) so a consumer (e.g. the turn-tax benchmark)
// can attribute a local serve to its tier from the result alone — the same
// "tier" key tier-2 already carries. taint is the label stamped on the served
// payload (servedTaint for tier-1; TaintTrusted for the args-independent tier-3).
func (v *VDSO) served(ctx context.Context, c *abi.ToolCall, out []byte, tierN int, taint abi.TaintLabel) *abi.Result {
	atomic.AddInt64(&v.hits, 1)
	return &abi.Result{Call: c, Payload: v.put(ctx, out, taint), Status: abi.StatusOK,
		Meta: map[string]string{"served_by": "vdso", "tier": strconv.Itoa(tierN)}}
}

// keyLocked builds the tier-2 cache key — tool : argHash : epoch-stamp — and MUST be
// called with v.mu held. Computing the key and touching the cache map under the same
// lock makes a hit airtight against a concurrent epoch bump: the bump either fully
// precedes the key build (a miss, correct) or follows the map check (it strands a key
// we already decided on, never the one we serve). Global stamps the root epoch only
// (v0.1 behavior — one scalar, any write strands every entry); finer modes stamp the
// epoch of every node on the read's root->leaf chain, joined with '.' so distinct
// chains can never alias (e.g. [1,2] -> "1.2" never collides with [12] -> "12").
func (v *VDSO) keyLocked(c *abi.ToolCall, args []byte) string {
	h := v.argHashFor(args)
	// Per-principal isolation (principal.go): scope the hash to the caller's principal
	// so a DIFFERENT principal can neither be served nor fill this entry — closing the
	// cross-tenant cache leak + the hit/miss timing oracle. A nil/empty principal or a
	// tool declared Shareable leaves h untouched, so the key is BYTE-IDENTICAL to the
	// single-tenant v0.1 key (default sharing and cross-tenant PUBLIC sharing both
	// preserved). v.shareable is read under v.mu, already held here — like v.pure.
	if p := principalOf(c); p != "" && !v.shareable[c.Tool] {
		h = scopeHash(p, h)
	}
	base := c.Tool + ":" + h
	if v.GranularityOf() == Global {
		return base + ":" + atou(atomic.LoadUint64(&v.worldVer))
	}
	chain := v.readChain(c, args)
	var sb strings.Builder
	sb.Grow(len(base) + 1 + 6*len(chain))
	sb.WriteString(base)
	sb.WriteByte(':')
	for i, tag := range chain {
		if i > 0 {
			sb.WriteByte('.')
		}
		sb.WriteString(atou(v.epochLocked(tag)))
	}
	return sb.String()
}

// Emit observes completions: it fills the tier-2 cache for read-only+idempotent
// calls and bumps the world-version on any write-shaped completion (cache
// invalidation, unit 28).
func (v *VDSO) Emit(ev abi.Event) {
	if ev.Kind != abi.EvComplete || ev.Call == nil || ev.Result == nil {
		return
	}
	c, r := ev.Call, ev.Result
	if r.Status != abi.StatusOK {
		return
	}
	if destructive(c) {
		// The finer eraser: bump only the epoch(s) of the tag(s) this write touches,
		// then publish the mutation on the coherence bus. In Global mode writeTags is
		// always ["*"], so this reduces to the v0.1 worldVer++ full flush; in finer
		// modes it strands only the affected subtree and leaves siblings warm.
		var wargs []byte
		if v.GranularityOf() != Global {
			wargs = v.bytes(context.Background(), c.Args)
		}
		tags := v.writeTags(c, wargs)
		v.bumpAndPublish(c, tags)
		return
	}
	if !(metaTrue(c, "readOnlyHint") && metaTrue(c, "idempotentHint")) {
		return
	}
	// already served by the vDSO? don't re-store.
	if r.Meta != nil && r.Meta["served_by"] == "vdso" {
		return
	}
	args := v.bytes(context.Background(), c.Args)
	// Resource-mode soundness gate: a known-namespace read that can't name its entity
	// is not cacheable (an entity-fine write would miss it) — don't store it.
	if v.resourceMisnamed(c, args) {
		return
	}
	// Temporal-cache negative-result guard (neardup.go): in near-dup mode a negative
	// answer ("not found" / empty) is never stored, so a formatting-variant query can
	// never be served a stale negative that has since flipped positive.
	if v.NearDupOf() && negativeResult(v.bytes(context.Background(), r.Payload)) {
		return
	}
	wit := v.resolveWitness(c, r)
	// The fill + LRU-evict run under v.mu. We collect cachemeta emit jobs inside
	// the lock (where the entry identities live) and dispatch them OUTSIDE the lock
	// via this IIFE — the `defer v.mu.Unlock()` keeps the early-return unlock
	// semantics byte-identical, so a sink that re-enters the vDSO cannot deadlock.
	fillJobs, evictJobs := func() (fill, evicted []emitJob) {
		v.mu.Lock()
		defer v.mu.Unlock()
		// Integrity gate (revoke.go): never RE-ADMIT under a witness a refutation retired —
		// the durable CAS makes the poisoned bytes content-stable, so without this an evicted
		// entry would silently repopulate on the next read.
		if v.revokedLocked(wit) {
			return nil, nil
		}
		key := v.keyLocked(c, args)
		if _, ok := v.cache[key]; ok {
			return nil, nil
		}
		el := v.lru.PushFront(&entry{key: key, ref: r.Payload, witness: wit})
		v.cache[key] = el
		atomic.AddInt64(&v.fills, 1)
		// Pin the CAS bytes UNDER v.mu, before the entry is reachable to any Lookup,
		// so a concurrent eviction on a bounded store cannot drop a digest this tier-2
		// entry will resolve on a later hit (the soundness race). The blob store is a
		// leaf — it never re-enters the vDSO — so this foreign call under v.mu cannot
		// deadlock, unlike emitCache (which a sink may re-enter, so that stays outside).
		abi.PinResolved(r.Payload)
		fill = []emitJob{{key: key, ref: r.Payload, witness: wit}}
		for v.lru.Len() > v.cap { // LRU eviction (unit 36)
			back := v.lru.Back()
			if back == nil {
				break
			}
			ce := back.Value.(*entry)
			v.lru.Remove(back)
			delete(v.cache, ce.key)
			abi.UnpinResolved(ce.ref) // left the cache -> release its CAS pin
			evicted = append(evicted, emitJob{key: ce.key, ref: ce.ref, witness: ce.witness})
		}
		return fill, evicted
	}()
	for _, j := range fillJobs {
		v.emitCache(CacheFill, j.key, j.ref, j.witness)
	}
	for _, j := range evictJobs {
		v.emitCache(CacheEvict, j.key, j.ref, j.witness)
	}
}

// emitJob carries a tier-2 identity (key + payload ref + witness) from a locked
// cache mutation to the outside-the-lock emission path.
type emitJob struct {
	key     string
	ref     abi.Ref
	witness string
}

// BumpWorld manually advances the root ("*") epoch — the panic-button full flush and
// the per-epoch trial-isolation reset the fleet benchmark relies on. Every read binds
// the root, so this invalidates the whole cache at any granularity. It takes v.mu so a
// finer-mode keyLocked cannot tear the chain against it. It is an INTERNAL reset hook
// and deliberately does NOT publish on the coherence bus (it is not a real-world
// mutation); a genuine global-scope write reaches the bus through Emit -> bump(["*"]).
func (v *VDSO) BumpWorld() {
	v.mu.Lock()
	atomic.AddUint64(&v.worldVer, 1)
	v.mu.Unlock()
}

// WorldVersion reports the current version.
func (v *VDSO) WorldVersion() uint64 { return atomic.LoadUint64(&v.worldVer) }

// Stats reports lookups, hits, fills, and the hit rate (unit 31, 33).
func (v *VDSO) Stats() (lookups, hits, fills int64, hitRate float64) {
	l := atomic.LoadInt64(&v.lookups)
	h := atomic.LoadInt64(&v.hits)
	f := atomic.LoadInt64(&v.fills)
	if l > 0 {
		hitRate = float64(h) / float64(l)
	}
	return l, h, f, hitRate
}

// PureTools returns the names of the registered tier-1 pure tools, sorted. The
// static tool linter enumerates this to check that each pure registration is
// actually REACHABLE under the tier-1 hint gate (a pure tool whose hints can never
// satisfy readOnly+idempotent+!destructive is dead code). Returns a fresh slice.
func (v *VDSO) PureTools() []string {
	v.mu.Lock()
	out := make([]string, 0, len(v.pure))
	for t := range v.pure {
		out = append(out, t)
	}
	v.mu.Unlock()
	sort.Strings(out)
	return out
}

// StaticTools returns the names of the registered tier-3 static tools, sorted. The
// linter enumerates this to flag a canned answer registered for a write-shaped name
// — tier-3 is served UNCONDITIONALLY (Lookup applies no destructive gate to it), so
// a static answer for a "send"/"delete" tool would silently swallow the write.
// Returns a fresh slice.
func (v *VDSO) StaticTools() []string {
	v.mu.Lock()
	out := make([]string, 0, len(v.static))
	for t := range v.static {
		out = append(out, t)
	}
	v.mu.Unlock()
	sort.Strings(out)
	return out
}

func atou(n uint64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// ----------------------------------------------------------------------------
// Default instance + registration.
// ----------------------------------------------------------------------------

// Default is the registered vDSO.
var Default = New(DefaultCacheSize)

func init() {
	// Seed a tier-1 pure tool and a tier-3 static tool so v0.1 has a live fast path.
	Default.RegisterPure("calculate", calcSum)
	Default.RegisterStatic("list_all_airports", []byte(`{"airports":["SFO","JFK","LAX","ORD","SEA","BOS","ATL","DFW"]}`))

	abi.RegisterFastPath(1, tier{Default, 1}) // pure
	abi.RegisterFastPath(3, tier{Default, 3}) // static + cache (Lookup handles all)
	abi.RegisterEmitter(Default)
	abi.RegisterCapability("vdso.v1")
}

// tier wraps the VDSO so a single Lookup covers all three; registering twice at
// different tiers keeps the FastPath ordering contract honest without duplicating
// the lookup logic.
type tier struct {
	v *VDSO
	n int
}

// Caps delegates to the wrapped VDSO, advertising nothing special.
func (t tier) Caps() []abi.Capability { return t.v.Caps() }
func (t tier) Lookup(ctx context.Context, c *abi.ToolCall) (*abi.Result, bool) {
	// Only the first registered tier actually serves; the second registration is a
	// no-op so a single Lookup isn't run twice per call.
	if t.n != 1 {
		return nil, false
	}
	return t.v.Lookup(ctx, c)
}

// calcSum is a trivially-pure tool: {"a":num,"b":num} -> {"sum":num}. It proves
// the tier-1 path (a result computed locally, zero engine calls).
func calcSum(args []byte) ([]byte, bool) {
	var in struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, false
	}
	return []byte(`{"sum":` + strconv.FormatFloat(in.A+in.B, 'g', -1, 64) + `}`), true
}

// ----------------------------------------------------------------------------
// Tier-4 content dedup cache (issue #1101)
// ----------------------------------------------------------------------------

// contentHash computes the content-addressed hash of a content block.
// It uses the same SHA256-based approach as argHash for consistency.
func contentHash(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])[:24]
}

// LookupContent checks if the given content is already in the content cache.
// If found, it returns the cached Ref (which may be inline or blob-backed).
// If not found, it returns (zero Ref, false).
// This is the fast path for deduplicating inbound content blocks.
func (v *VDSO) LookupContent(b []byte) (abi.Ref, bool) {
	if len(b) == 0 {
		return abi.Ref{}, false
	}
	hash := contentHash(b)
	v.mu.Lock()
	ref, ok := v.contentCache[hash]
	if ok {
		// Move to front of LRU
		if el, exists := v.contentCacheIndex[hash]; exists {
			v.contentCacheLRU.MoveToFront(el)
		}
	}
	v.mu.Unlock()
	if ok {
		atomic.AddInt64(&v.contentHits, 1)
	}
	return ref, ok
}

// FillContent adds a content block to the dedup cache, returning its Ref.
// If the content is already cached, the existing Ref is returned.
// Otherwise, the content is stored (as a Ref via the resolver) and cached.
// The caller must supply a context for the Put operation.
func (v *VDSO) FillContent(ctx context.Context, b []byte) abi.Ref {
	if len(b) == 0 {
		return abi.Ref{Kind: abi.RefInline, Inline: []byte{}}
	}

	hash := contentHash(b)

	// Check if already cached
	v.mu.Lock()
	if ref, ok := v.contentCache[hash]; ok {
		if el, exists := v.contentCacheIndex[hash]; exists {
			v.contentCacheLRU.MoveToFront(el)
		}
		v.mu.Unlock()
		return ref
	}
	v.mu.Unlock()

	// Not cached: store via resolver
	ref := abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), b...)}
	if res := abi.ActiveResolver(); res != nil {
		if r, err := res.Put(ctx, b); err == nil {
			ref = r
		}
	}

	// Add to cache with LRU eviction
	v.mu.Lock()
	defer v.mu.Unlock()

	// Double-check in case of race
	if existing, ok := v.contentCache[hash]; ok {
		return existing
	}

	v.contentCache[hash] = ref
	el := v.contentCacheLRU.PushFront(hash)
	v.contentCacheIndex[hash] = el
	atomic.AddInt64(&v.contentFills, 1)

	// Evict if over capacity
	for v.contentCacheLRU.Len() > v.contentCacheSize {
		back := v.contentCacheLRU.Back()
		if back == nil {
			break
		}
		oldHash := back.Value.(string)
		v.contentCacheLRU.Remove(back)
		delete(v.contentCacheIndex, oldHash)
		delete(v.contentCache, oldHash)
	}

	return ref
}

// ContentStats returns the content dedup metrics.
func (v *VDSO) ContentStats() (hits, fills int64) {
	return atomic.LoadInt64(&v.contentHits), atomic.LoadInt64(&v.contentFills)
}
