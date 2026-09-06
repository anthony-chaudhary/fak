package ctxresidency

import (
	"context"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/capindex"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// ---------------------------------------------------------------------------
// C3: per-capability residency + eviction — the MemTool working set with fak's
// WITNESSED eviction and MEASURED blast radius (issue #1106).
//
// A faulted capability body (a skill body, an MCP tool schema, an A2A method
// spec) is resident. Under context pressure the loader evicts the COLDEST
// evictable capability — never a held (CAS-pinned / in-flight) one — through
// ctxmmu's page-out + witness-clear gate. The blast radius is MEASURED from the
// tracked dependents (what dropping the capability invalidates), not a heuristic
// relevance score. A re-query for an evicted capability is a FAULT that re-pages
// it (PageInBody round-trips the witness gate); an identical re-invocation is a
// HIT from contextq's procedural cache (out of this leaf's lane — this leaf
// tracks the residency STATE, contextq owns the cache).
//
// This is the MemTool per-turn working set (arXiv 2507.21428) over the MMU fak
// already operates: query-driven admission, pressure-driven eviction, witnessed
// page-out. It reuses the SAME State / BlastRadius vocabulary the span-level
// Query already exposes (so a capability and a KV span speak one residency
// language) and drives the SAME ctxmmu held-ledger + CAS-pin + witness-clear
// gate (so a capability eviction is as witnessed as a quarantine).
// ---------------------------------------------------------------------------

// CapKey is the capability identity the residency tracker keys on. It is an
// alias for capindex.CapRef now that the protocol-blind capindex core is tier-2
// and legal for this tier-3 leaf to import; the local mirror that used string
// Kind fields is gone, while the public ctxresidency spelling remains stable.
type CapKey = capindex.CapRef

// StateEvicting marks an in-flight eviction candidate while page-out is prepared.
// It prevents multi-evictor collision by taking the candidate out of the evictable
// pool while the lock is temporarily released.
const StateEvicting State = "evicting"

// capState is one resident capability's tracked residency. It is the
// per-capability analogue of a kvmmu Segment row: its residency class, its
// coldness clock (for the LRU eviction pick), its CAS-pin (in-flight) flag, the
// measured cost of dropping it, and — once evicted — the ctxmmu held id its body
// pages back in through.
type capState struct {
	key        CapKey
	digest     string
	tokens     int      // the body's resident cost — the measured blast-radius base.
	dependents []CapKey // capabilities this one invalidates if dropped (MEASURED, not guessed).
	pinned     bool     // CAS-pinned: an in-flight invocation holds it; NEVER evicted.
	tier       Tier     // base-context layout tier (Rung 4, #1262); spine/policy are NEVER paged.
	lastUse    int64    // monotonic access seq; the smallest is the COLDEST.
	state      State    // resident (has dependents) | evictable (none) | evicting (in-flight) | held (evicted).
	pageID     string   // ctxmmu held id once paged out (re-fault pages in through it).
}

// CapResidency is the per-capability residency + eviction tracker. It holds the
// working set of faulted capabilities, ranks them by coldness, and evicts the
// coldest EVICTABLE one under pressure through the witnessed ctxmmu gate. It is
// concurrency-safe (a guard/serve loop faults and evicts from multiple turns).
//
// Construct with NewCapResidency(mmu): the mmu is the SAME ctxmmu.MMU the rest
// of the context runs through, so a capability page-out shares the held ledger,
// the CAS pin, the FIFO bound, and the witness-clear gate with quarantine. A nil
// mmu is allowed (the tracker still tracks state and measures blast radius) but
// an Evict then cannot witness a page-out and reports it.
type CapResidency struct {
	mmu *ctxmmu.MMU

	mu   sync.Mutex
	caps map[CapKey]*capState
	seq  int64 // monotonic access clock; bumped on every Fault/Touch.
	// maxHeld bounds the retained HELD (evicted) tombstones — see
	// defaultMaxHeldCaps and pruneHeldLocked. 0 = the default cap; <0 = unbounded
	// (the original behavior); >0 = a custom cap.
	maxHeld int

	unlockHook func() // for testing concurrent eviction interleavings
}

// SetEvictUnlockHookForTest installs a callback invoked while cr.mu is released
// during EvictColdest, before the lock is re-acquired. For testing concurrent
// interleavings only.
func (cr *CapResidency) SetEvictUnlockHookForTest(hook func()) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	cr.unlockHook = hook
}

// NewCapResidency builds a tracker over the shared ctxmmu gate.
func NewCapResidency(mmu *ctxmmu.MMU) *CapResidency {
	return &CapResidency{mmu: mmu, caps: make(map[CapKey]*capState)}
}

// Fault records that a capability body has been paged in (faulted) and is now
// resident. body is the materialized body (its length is the measured residency
// cost); dependents are the capabilities that dropping this one would invalidate
// (e.g. a version-bound child, a composed sub-capability) — the MEASURED blast
// radius, supplied by the caller from the real dependency graph, never guessed
// here. A re-Fault of an already-resident capability refreshes its coldness and
// dependents (a re-query is a fresh access) and clears any prior held id, since
// its body is resident again. Fault is the admission half of the working set.
func (cr *CapResidency) Fault(key CapKey, digest string, body []byte, dependents []CapKey) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	cr.seq++
	st, ok := cr.caps[key]
	if !ok {
		st = &capState{key: key}
		cr.caps[key] = st
	}
	st.digest = digest
	st.tokens = len(body)
	st.dependents = append([]CapKey(nil), dependents...)
	st.lastUse = cr.seq
	st.pageID = "" // resident again: no outstanding page-out.
	st.state = classify(st)
}

// Touch bumps a resident capability's coldness clock (an access that is not a
// fresh fault — e.g. a procedural-cache HIT that re-uses the body without
// re-rendering it). It keeps the working set's LRU honest: a frequently-used
// capability stays warm and is not the eviction pick. Touching an unknown or
// evicted capability is a no-op (nothing resident to warm).
func (cr *CapResidency) Touch(key CapKey) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	if st, ok := cr.caps[key]; ok && st.state != StateHeld {
		cr.seq++
		st.lastUse = cr.seq
	}
}

// Pin marks a capability as CAS-pinned: an in-flight invocation holds its frame,
// so it must survive eviction (the CAS pin in ctxmmu already guarantees a pinned
// PAGE survives; this pins the capability at the residency tier so the eviction
// PICK never even selects it). Pin/Unpin bracket an in-flight invocation. Pin of
// an unknown capability is a no-op.
func (cr *CapResidency) Pin(key CapKey) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	if st, ok := cr.caps[key]; ok {
		st.pinned = true
		st.state = classify(st)
	}
}

// Unpin releases an in-flight pin: the invocation completed, so the capability
// is once again an eviction candidate (subject to its dependents). Unpin of an
// unknown capability is a no-op.
func (cr *CapResidency) Unpin(key CapKey) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	if st, ok := cr.caps[key]; ok {
		st.pinned = false
		st.state = classify(st)
	}
}

// classify is the per-capability residency rule, the capability analogue of the
// span-level resident/evictable/held split in Query. A paged-out (evicted) body
// is HELD; a block in a never-paged layout tier (spine/policy — Rung 4, #1262), a
// pinned (in-flight) capability, or one with live dependents is RESIDENT (held in
// residence — an eviction would invalidate something, or the tier forbids paging
// it); a clean, unpinned, dependent-free OVERLAY capability is EVICTABLE.
//
// The spine/policy guard is the by-construction exclusion invariant 3 asks for: a
// safety/identity-load-bearing block resolves (via ClassifyTier) to a never-paged
// tier, so it can never reach StateEvictable regardless of how cold it gets —
// silent under-retrieval cannot drop it because no eviction path selects it. The
// caller holds the lock.
func classify(st *capState) State {
	if st.pageID != "" {
		return StateHeld
	}
	if st.tier.AlwaysResident() || st.pinned || len(st.dependents) > 0 {
		return StateResident
	}
	return StateEvictable
}

// MeasureBlastRadius returns what evicting key would cost — its resident tokens
// and the count of capabilities it would invalidate (its tracked dependents) —
// WITHOUT evicting anything. This is the "cost is measured, not guessed"
// contract: the number reported before an evict is exactly the dependents an
// evict would drop, read from the live dependency graph the loader supplied at
// Fault time. An unknown or already-held capability reports a zero radius.
func (cr *CapResidency) MeasureBlastRadius(key CapKey) BlastRadius {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	st, ok := cr.caps[key]
	if !ok || st.state == StateHeld {
		return BlastRadius{}
	}
	return blastOf(st)
}

// blastOf is the measured eviction cost of one capability: its own resident
// token count plus the number of live dependents an eviction would invalidate.
// It is a pure read of the tracked graph — never a heuristic. The caller holds
// the lock.
func blastOf(st *capState) BlastRadius {
	return BlastRadius{Tokens: st.tokens, DependentEntries: len(st.dependents)}
}

// EvictColdest evicts the single coldest EVICTABLE capability under context
// pressure and returns which one it dropped, the MEASURED blast radius it
// incurred, and whether anything was evicted. The pick is deterministic:
//
//   - HELD capabilities are already evicted (skipped).
//   - PINNED (in-flight) and RESIDENT (has live dependents) capabilities are
//     NEVER evicted — held in residence.
//   - EVICTING capabilities are currently in-flight by another evictor (skipped).
//   - Among the remaining EVICTABLE capabilities, the COLDEST (smallest lastUse)
//     is chosen; ties break on the capability key for determinism.
//
// The blast radius is measured BEFORE the page-out (so the returned cost is the
// real cost, read from the dependency graph). The eviction is WITNESSED: the
// body pages out through ctxmmu.PageOutBody (CAS-pinned, gated), so a re-fault
// must page it back in through a witness Clear() — it cannot silently reappear.
// If no capability is evictable (all pinned/resident/held, or the tracker is
// empty) it returns ok=false with a zero radius and evicts nothing.
//
// A nil mmu (or a page-out the CAS refuses) still updates the residency STATE to
// held and returns the measured radius and ok=true, but pageID stays empty so a
// later re-fault re-resolves the body from its source rather than the CAS — the
// fail-closed degradation, never a claim that an unwitnessed body is recoverable.
func (cr *CapResidency) EvictColdest(ctx context.Context) (evicted CapKey, radius BlastRadius, ok bool) {
	cr.mu.Lock()
	victim := cr.coldestEvictableLocked()
	if victim == nil {
		cr.mu.Unlock()
		return CapKey{}, BlastRadius{}, false
	}
	// Reserve candidate in StateEvicting so concurrent evictors select different candidates.
	victim.state = StateEvicting

	// MEASURE before the drop: this is the cost an evict actually incurs.
	radius = blastOf(victim)
	key := victim.key
	victimLastUse := victim.lastUse
	mmu := cr.mmu
	hook := cr.unlockHook
	cr.mu.Unlock()

	if hook != nil {
		hook()
	}

	cr.mu.Lock()
	defer cr.mu.Unlock()
	// Re-read under the lock: a concurrent Fault or Touch may have re-admitted it,
	// advanced its lastUse, or transitioned its state away from evicting.
	// If the capability was re-faulted, pinned, or altered while the lock was
	// released, abort the eviction and restore proper state.
	st, still := cr.caps[key]
	if !still || st.lastUse > victimLastUse || st.state != StateEvicting {
		if still && st.state == StateEvicting {
			st.state = classify(st)
		}
		return CapKey{}, BlastRadius{}, false
	}

	// WITNESS the eviction through the shared ctxmmu gate only after confirming
	// the candidate was not invalidated while unlocked. This prevents orphaned
	// CAS page-out handles and skewed MMUCapPaged counters if eviction aborts.
	var pageID string
	if mmu != nil && st.tokens > 0 {
		// Page out a body-sized marker through the real gate. The capability
		// body itself is owned by the resolver; what the gate holds is the
		// witnessed, CAS-pinned page-out the re-fault round-trips. A
		// body-length placeholder keeps the held-ledger accounting honest
		// (a >0 length the gate will pin and a Clear() must re-admit).
		if id, paged := mmu.PageOutBody(ctx, make([]byte, st.tokens)); paged {
			pageID = id
		}
	}

	st.pageID = pageID
	st.tokens = 0 // body no longer resident.
	st.state = StateHeld
	cr.pruneHeldLocked() // bound the accumulated held tombstones (the "delete on evict" axis).
	return key, radius, true
}

// defaultMaxHeldCaps bounds how many HELD (evicted) tombstones the tracker
// retains. A held entry survives eviction ONLY so Snapshot can report it (the
// Held count and the paged-out row) — its body is already gone, and a re-Fault
// re-resolves the body from source whether or not the tombstone is still present
// (Fault recreates a missing entry). Without a bound the tombstones accumulate
// one per distinct eviction for the life of the tracker: a long guard/serve loop
// that faults and evicts thousands of distinct capabilities leaks a map row for
// each. The bound keeps caps O(live working set + cap) — the live
// resident/evictable set is the caller's to bound (via eviction), and only the
// disposable held tombstones are pruned here.
const defaultMaxHeldCaps = 1024

// SetMaxHeld overrides the retained-held-tombstone cap: n>0 sets a custom cap,
// n<0 disables the bound (unbounded — the original behavior), n==0 restores the
// default. It re-applies the bound immediately, so lowering the cap trims the
// existing tombstones. Concurrency-safe.
func (cr *CapResidency) SetMaxHeld(n int) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	cr.maxHeld = n
	cr.pruneHeldLocked()
}

// heldCap resolves the effective held-tombstone cap: (cap, true) when bounded,
// (_, false) when the caller disabled the bound (maxHeld<0).
func (cr *CapResidency) heldCap() (max int, bounded bool) {
	switch {
	case cr.maxHeld < 0:
		return 0, false
	case cr.maxHeld == 0:
		return defaultMaxHeldCaps, true
	default:
		return cr.maxHeld, true
	}
}

// pruneHeldLocked bounds the retained HELD tombstones. The caller holds the lock;
// it runs after an eviction transitions an entry to StateHeld (and after a cap
// change). When the held set exceeds the cap, the COLDEST held tombstones
// (smallest lastUse, then key — the same determinism the eviction pick uses) are
// deleted until the held set is back at the cap. Below the cap it is a no-op, so
// residency reporting is unchanged for any realistic working set. Live
// (resident/evictable) entries are never touched: dropping one would silently
// evict a still-usable capability.
func (cr *CapResidency) pruneHeldLocked() {
	max, bounded := cr.heldCap()
	if !bounded {
		return
	}
	held := 0
	for _, st := range cr.caps {
		if st.state == StateHeld {
			held++
		}
	}
	if held <= max {
		return
	}
	victims := make([]*capState, 0, held)
	for _, st := range cr.caps {
		if st.state == StateHeld {
			victims = append(victims, st)
		}
	}
	sort.Slice(victims, func(i, j int) bool {
		if victims[i].lastUse != victims[j].lastUse {
			return victims[i].lastUse < victims[j].lastUse
		}
		return keyLess(victims[i].key, victims[j].key)
	})
	for _, st := range victims[:held-max] {
		delete(cr.caps, st.key)
	}
}

// coldestEvictableLocked returns the coldest evictable capability, or nil if
// none is evictable. The caller holds the lock. Determinism: the smallest
// lastUse wins; equal coldness breaks on the {Kind, Name, Version} key so the
// pick is reproducible across runs (a property the witnessing tests rely on).
func (cr *CapResidency) coldestEvictableLocked() *capState {
	var best *capState
	for _, st := range cr.caps {
		if st.state != StateEvictable {
			continue
		}
		if best == nil || st.lastUse < best.lastUse || (st.lastUse == best.lastUse && keyLess(st.key, best.key)) {
			best = st
		}
	}
	return best
}

// keyLess is the deterministic tiebreak over capability keys.
func keyLess(a, b CapKey) bool {
	if a.Kind != b.Kind {
		return a.Kind < b.Kind
	}
	if a.Name != b.Name {
		return a.Name < b.Name
	}
	return a.Version < b.Version
}

// CapSnapshot is the residency working-set read: one row per tracked capability
// plus the reconciled counts. It is the per-capability peer of the span-level
// Snapshot and the input the C6 audit surface reconciles against the gate's own
// counters. A pure read — it mutates nothing.
type CapSnapshot struct {
	Caps []CapRow

	Resident  int // capabilities held in residence (pinned or with dependents).
	Evictable int // clean eviction candidates.
	Held      int // evicted (paged-out) capabilities.

	// MMUCapPaged is the gate's own lifetime capability page-out counter
	// (ctxmmu.MMU.CapPaged). A witness test asserts it is >= the tracker's Held
	// transitions that actually paged out — the loader's view reconciles with
	// the kernel's ledger, the same trust floor LoaderJournal enforces.
	MMUCapPaged int64
}

// CapRow is one capability's residency snapshot row.
type CapRow struct {
	Key              CapKey
	Digest           string
	State            State
	Tier             Tier // base-context layout tier (spine/policy = never paged; overlay = pageable).
	Pinned           bool
	LastUse          int64
	EvictBlastRadius BlastRadius // measured cost of dropping it (zero once held).
	PageID           string      // ctxmmu held id while held; empty while resident.
}

// Snapshot returns a consistent residency read over the tracked capabilities.
// Rows are sorted by key for a deterministic, diffable view. It is a pure read
// (no eviction, no page-out), so it can never launder an evicted body back into
// residence — a re-fault still pages it in through the witness gate.
func (cr *CapResidency) Snapshot() CapSnapshot {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	out := CapSnapshot{Caps: make([]CapRow, 0, len(cr.caps))}
	for _, st := range cr.caps {
		row := CapRow{
			Key: st.key, Digest: st.digest, State: st.state, Tier: st.tier,
			Pinned: st.pinned, LastUse: st.lastUse, PageID: st.pageID,
		}
		if st.state != StateHeld {
			row.EvictBlastRadius = blastOf(st)
		}
		switch st.state {
		case StateResident:
			out.Resident++
		case StateEvictable, StateEvicting:
			out.Evictable++
		case StateHeld:
			out.Held++
		}
		out.Caps = append(out.Caps, row)
	}
	sort.Slice(out.Caps, func(i, j int) bool { return keyLess(out.Caps[i].Key, out.Caps[j].Key) })
	if cr.mmu != nil {
		out.MMUCapPaged = cr.mmu.CapPaged()
	}
	return out
}
