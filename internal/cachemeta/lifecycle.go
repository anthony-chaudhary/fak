package cachemeta

// lifecycle.go gives a cache entry a multi-state, per-tier-TTL lifecycle — the
// "states flexible, TTL configurable down to the lowest levels" half of the
// hardware-aware design.
//
// A single global Validity.TTLMillis answers only one question: is this entry stale
// yet? A tiered cache needs more. The SAME span is cheap to keep in CXL for a long
// time and expensive to keep in HBM for even a short one, so freshness must be
// expressible PER TIER, and the entry needs explicit states so the policy can act on
// "expired in HBM, demote it" without conflating that with "expired everywhere, drop
// it." This file provides:
//
//   - EntryState: the flexible state set (filling -> resident -> expiring ->
//     expired/spilled -> evicted), each a deterministic function of the clock, not an
//     opaque flag.
//   - TierTTL: a per-tier TTL map — the TTL "down to the lowest levels". Each tier
//     has its own expiry clock, measured from when the entry ENTERED that tier, so a
//     demote resets the freshness window for the colder tier.
//   - Lifecycle + Advance: a pure, wall-clock-FREE transition (the caller injects
//     nowMillis, the same testable posture as tools/bench_plan.py's injected --now),
//     so a workload can be replayed deterministically.
//
// Lifecycle answers WHEN a state changes (time/TTL); placement.go answers WHERE the
// payload should then go (capacity/cost). Keeping the two separate is what lets the
// time policy be tested without a hardware profile and the placement policy without a
// clock. The state transitions speak the existing KVTransfer vocabulary (KVOffload to
// demote, KVRestore to promote) so a consumer can feed a transition straight into
// engine.CacheEvent.

// EntryState is the lifecycle state of a cache entry. The states are deliberately
// richer than present/absent: a tiered cache demotes, spills, and expires-with-grace,
// and each of those is a distinct, observable state a policy and a metric can act on.
type EntryState string

const (
	// StateFilling — the payload is being produced (prefilled/written) and is NOT yet
	// serveable for reuse. Advance never auto-leaves Filling; the producer calls
	// MarkResident when the fill completes.
	StateFilling EntryState = "filling"
	// StateResident — serveable, living in Lifecycle.Tier.
	StateResident EntryState = "resident"
	// StateExpiring — the per-tier TTL for the current tier elapsed; the entry is a
	// demote/evict CANDIDATE but still serveable during the grace window. An access
	// (Touch) during this window REVIVES it to Resident.
	StateExpiring EntryState = "expiring"
	// StateExpired — the grace window elapsed with no access; no longer serveable from
	// this tier. The placement policy will demote or evict it.
	StateExpired EntryState = "expired"
	// StateSpilled — resident in the coldest, non-attendable-in-place tier (disk):
	// still recoverable, but a read must stage it back to a hotter tier first.
	StateSpilled EntryState = "spilled"
	// StateEvicted — removed entirely; the only way back is recompute.
	StateEvicted EntryState = "evicted"
)

// (There is deliberately no "moving"/"demoting" in-flight state: this plane models a
// relocation as ATOMIC — MoveTo lands the entry in its destination tier and emits the
// KVTransfer directive, and the engine adapter that performs the physical transfer
// owns any in-flight tracking. Keeping the move atomic here is what lets the state set
// stay free of an unreachable member.)

// Serveable reports whether an entry in this state can satisfy a reuse lookup. Filling
// is not yet ready; Expired/Evicted are gone. Resident, Expiring (grace), and Spilled
// (after a stage-back) are serveable.
func (s EntryState) Serveable() bool {
	switch s {
	case StateResident, StateExpiring, StateSpilled:
		return true
	default:
		return false
	}
}

// TierTTL is a per-tier freshness budget in milliseconds. A tier with no entry (or a
// zero/negative value) has NO TTL at that tier — the entry lives there until capacity
// pressure moves it. This is the "TTL configurable down to the lowest levels" knob:
// e.g. {HBM: 2s, DRAM: 60s, CXL: 0 (forever)} keeps the scarce hot tier turning over
// fast while letting a span rest indefinitely in cheap, roomy far memory.
type TierTTL map[ResidencyTier]int64

// LifecyclePolicy parameterizes the time-driven transitions. It carries the per-tier
// TTL, a default TTL for tiers the map does not name (mirrors the existing single
// Validity.TTLMillis), the expiring->expired grace window, and whether an expired
// entry should DEMOTE to a colder tier (the relocate-don't-drop default) or go
// straight to evict.
type LifecyclePolicy struct {
	TierTTL          TierTTL
	DefaultTTLMillis int64 // applied to tiers absent from TierTTL; 0 = no TTL
	GraceMillis      int64 // Expiring -> Expired window; 0 = expire immediately
	DemoteOnExpiry   bool  // expired entry relocates to a colder tier (vs evict)
}

// ttlFor returns the freshness budget for a tier: its explicit TierTTL entry, else the
// DefaultTTLMillis. A non-positive result means "no expiry at this tier".
func (p LifecyclePolicy) ttlFor(t ResidencyTier) int64 {
	if p.TierTTL != nil {
		if ms, ok := p.TierTTL[t]; ok {
			return ms
		}
	}
	return p.DefaultTTLMillis
}

// Lifecycle is the mutable per-entry lifecycle record. It is a small value type the
// caller stores alongside (or projects from) a cachemeta.Entry; Advance/Touch/MoveTo
// return a NEW Lifecycle rather than mutating in place, so a replay is reproducible.
//
// EnteredTierMillis is the clock the per-tier TTL is measured from, and is reset on
// every tier change — so freshness is genuinely per-tier, not a single global age.
type Lifecycle struct {
	State             EntryState
	Tier              ResidencyTier
	AdmittedAtMillis  int64  // when the entry first became resident anywhere
	EnteredTierMillis int64  // when it entered its CURRENT tier (per-tier TTL clock)
	StateSinceMillis  int64  // when it entered its CURRENT state (grace clock)
	LastAccessMillis  int64  // most recent Touch
	Accesses          uint64 // total reuse hits
}

// NewLifecycle starts an entry in Filling at the given tier and clock.
func NewLifecycle(tier ResidencyTier, nowMillis int64) Lifecycle {
	return Lifecycle{
		State:             StateFilling,
		Tier:              tier,
		AdmittedAtMillis:  nowMillis,
		EnteredTierMillis: nowMillis,
		StateSinceMillis:  nowMillis,
		LastAccessMillis:  nowMillis,
	}
}

// MarkResident promotes a Filling entry to Resident (or Spilled, if it filled directly
// into a non-attendable-in-place tier). A no-op for an entry already past Filling.
func (lc Lifecycle) MarkResident(profiles map[ResidencyTier]TierProfile, nowMillis int64) Lifecycle {
	if lc.State != StateFilling {
		return lc
	}
	lc.State = residentStateFor(lc.Tier, profiles)
	lc.StateSinceMillis = nowMillis
	return lc
}

// Touch records a reuse hit at nowMillis. A hit during the Expiring grace window
// REVIVES the entry to Resident (the access proves it is still hot), and a hit on a
// Spilled entry leaves it Spilled (the caller stages it back via MoveTo). Accesses and
// LastAccessMillis always advance, feeding the promote heuristic in placement.go.
func (lc Lifecycle) Touch(nowMillis int64) Lifecycle {
	lc.Accesses++
	lc.LastAccessMillis = nowMillis
	if lc.State == StateExpiring {
		lc.State = StateResident
		lc.StateSinceMillis = nowMillis
	}
	return lc
}

// MoveTo relocates the entry to a new tier at nowMillis, resetting the per-tier TTL
// clock, and returns the new Lifecycle plus the KVTransfer directive that describes
// the move (KVOffload to a colder tier, KVRestore to a hotter one). Moving to
// TierRecompute is modeled as eviction (Evict is the dedicated helper); MoveTo to a
// real tier lands the entry Resident, or Spilled if the destination is not attendable
// in place.
func (lc Lifecycle) MoveTo(tier ResidencyTier, profiles map[ResidencyTier]TierProfile, nowMillis int64) (Lifecycle, KVTransferDirection) {
	if tier == TierRecompute || tier == TierUnknown {
		return lc.Evict(nowMillis), KVOffload
	}
	dir := KVRestore
	if TierRank(tier) > TierRank(lc.Tier) {
		dir = KVOffload
	}
	lc.Tier = tier
	lc.EnteredTierMillis = nowMillis
	lc.State = residentStateFor(tier, profiles)
	lc.StateSinceMillis = nowMillis
	return lc, dir
}

// Evict marks the entry Evicted (removed; recompute required to recover it).
func (lc Lifecycle) Evict(nowMillis int64) Lifecycle {
	lc.State = StateEvicted
	lc.StateSinceMillis = nowMillis
	return lc
}

// Advance applies the TIME-driven transitions at nowMillis and reports whether the
// state changed. It moves Resident -> Expiring when the current tier's TTL has
// elapsed, and Expiring -> Expired after the grace window. It never moves bytes or
// chooses a destination tier — that is placement.go's job; Advance only decides WHEN
// an entry stops being fresh in its current tier. A non-empty returned direction is
// advisory (a demote is the natural follow-up to expiry under DemoteOnExpiry); the
// returned Lifecycle's tier is unchanged, so the caller still routes the move through
// placement + MoveTo.
func (lc Lifecycle) Advance(policy LifecyclePolicy, nowMillis int64) (Lifecycle, bool) {
	switch lc.State {
	case StateResident:
		ttl := policy.ttlFor(lc.Tier)
		if ttl > 0 && nowMillis-lc.EnteredTierMillis >= ttl {
			lc.State = StateExpiring
			lc.StateSinceMillis = nowMillis
			return lc, true
		}
	case StateExpiring:
		if nowMillis-lc.StateSinceMillis >= policy.GraceMillis {
			lc.State = StateExpired
			lc.StateSinceMillis = nowMillis
			return lc, true
		}
	}
	return lc, false
}

// AccessRatePerSec is the entry's LIFETIME reuse intensity since admission, in
// hits/second (a simple average, no recency decay). placement.go's promote gate uses
// it together with a recency guard (LastAccessMillis must be at/after EnteredTierMillis,
// i.e. the entry was actually touched since arriving in its current tier), so a long-
// idle entry with a high lifetime count is NOT mistaken for currently hot. A zero
// elapsed window yields 0 (no rate can be inferred from a single instant).
func (lc Lifecycle) AccessRatePerSec(nowMillis int64) float64 {
	elapsed := nowMillis - lc.AdmittedAtMillis
	if elapsed <= 0 {
		return 0
	}
	return float64(lc.Accesses) * 1000.0 / float64(elapsed)
}

// residentStateFor returns the resident state appropriate to a tier: Spilled for a
// tier that is not attendable in place (disk and unknown/off-ladder), Resident
// otherwise. A missing profile is treated conservatively as not-attendable.
func residentStateFor(tier ResidencyTier, profiles map[ResidencyTier]TierProfile) EntryState {
	if p, ok := profiles[tier]; ok && p.AttendableInPlace() {
		return StateResident
	}
	return StateSpilled
}
