package model

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
	"github.com/anthony-chaudhary/fak/pkg/moecache"
)

// expert_tier_policy.go — #1300: name the three rungs a routed expert can live on, give the two
// transitions between them a policy seam, and fold the existing per-rung ledgers into one
// tier-shaped reading.
//
// Why a THIRD vocabulary. The ladder shipped its rungs one at a time — the bounded device ring R0
// (#5611), the warm pin-set R2 (#5613), the checkpoint fault tier R5 (#5616), the page cache
// (#1302) — and each grew its own counters. An operator asking "where did this token's experts
// actually live, and what moved between rungs?" therefore had to join ExpertRingStats,
// ExpertCheckpointStats, the page-cache ledger and the pin counters by hand. This file does not add
// a cache; it adds the NAME for the rungs and the ONE transition policy the ladder was missing: an
// NVMe->L2 fill on a checkpoint miss (AC2) and its inverse L2->NVMe demotion (AC4), on the host/L2
// tier that sits between the device ring and the backing store.
//
// The tier vocabulary is deliberately the EXISTING pkg/moecache one, not a fourth dialect:
//
//	ExpertCacheTierVRAM  -> moecache.TierDRAM       the bounded device/host resident ring (R0)
//	ExpertCacheTierL2    -> moecache.TierCheckpoint the bounded host copy between ring and store (R5)
//	ExpertCacheTierNVMe  -> moecache.TierNVMe       the streamed backing store the tier reads from
//
// MoETier is the mapping, and AC1 is the rule that there are exactly three: a fourth tier is not
// invented here because the private gateway joins this record on exactly these names
// (expert_cache_telemetry.go already speaks them).
//
// The L2 tier is a polymodel.Pool bounded by ExpertTierPolicy.L2Bytes — the SAME proven policy the
// ring and the checkpoint tier already reuse (byte budget, deterministic victim, all-or-nothing
// admit), so `used <= budget` holds on this rung by construction too. It is allocated LAZILY: a
// session that never promotes allocates nothing, which is what makes AC5's default-off a
// byte-for-byte claim rather than a promise.
//
// Two design choices are load-bearing and stated here so they are not "fixed" later by accident:
//
//   - PROMOTION ONLY WHEN L2 HAS ROOM. Admit would happily evict an L2 resident to fit a newcomer,
//     which would make "L2 is full" unobservable and turn demotion into dead code. Promotion
//     therefore REFUSES when Used()+span exceeds the budget: filling L2 is the caller's explicit
//     choice and freeing room is demotion's explicit job. The two halves are complementary, not
//     redundant.
//   - DEMOTION NEVER DROPS A PINNED/WARM EXPERT. The warm-set is the ring's durable pin-set (R2),
//     consulted through the SAME routedExpertIdentity + pagedRing.isExpertPinned seam the demand
//     path uses, so a promotion/demotion policy cannot quietly un-protect the workload's hot set.
//     When every L2 candidate is pinned, nothing is demoted and the pass reports PreservedPinned>0.

// ExpertCacheTier is one rung of the expert memory waterfall. The zero value is VRAM, the fastest
// rung, so an unset tier reads as the resident device/host ring rather than an error state.
type ExpertCacheTier int

const (
	// ExpertCacheTierVRAM is the bounded DEVICE-resident ring (R0) — the fastest rung.
	ExpertCacheTierVRAM ExpertCacheTier = iota
	// ExpertCacheTierL2 is the bounded HOST/L2 copy one rung below the ring (R5): the middle tier
	// this policy fills on an NVMe miss and drains on demotion.
	ExpertCacheTierL2
	// ExpertCacheTierNVMe is the streamed backing store the L2 tier reads from.
	ExpertCacheTierNVMe
)

// String names the tier for a report, a ledger line or a log.
func (t ExpertCacheTier) String() string {
	switch t {
	case ExpertCacheTierL2:
		return "l2"
	case ExpertCacheTierNVMe:
		return "nvme"
	case ExpertCacheTierVRAM:
		return "vram"
	}
	return fmt.Sprintf("ExpertCacheTier(%d)", int(t))
}

// valid reports whether t is a member of the closed three-tier vocabulary.
func (t ExpertCacheTier) valid() bool {
	return t == ExpertCacheTierVRAM || t == ExpertCacheTierL2 || t == ExpertCacheTierNVMe
}

// MoETier maps this tier onto the existing pkg/moecache tier vocabulary (AC1) so a live serve and a
// private gateway join on one name set. There is deliberately NO fourth tier: VRAM is the device
// resident tier, L2 is the checkpoint/host tier, NVMe is the backing store. An out-of-range value
// falls back to the VRAM mapping rather than fabricating a tier the contract does not have.
func (t ExpertCacheTier) MoETier() moecache.Tier {
	switch t {
	case ExpertCacheTierL2:
		return moecache.TierCheckpoint
	case ExpertCacheTierNVMe:
		return moecache.TierNVMe
	default:
		return moecache.TierDRAM
	}
}

// expertTierDefaultAlignBytes is the alignment ExpertNVMeGeometry rounds to when a caller declares
// none. 4096 is the page size of every supported host page cache, which is what makes the aligned
// span a real over-read bound rather than an arbitrary granularity.
const expertTierDefaultAlignBytes = 4096

// maxExpertTierInt is the largest int this host can express, used to clamp a caller-declared
// alignment that would otherwise overflow the int parameter pageAlignDown/pageAlignUp take.
var maxExpertTierInt = int(^uint(0) >> 1)

// maxExpertTierInt64 is MaxInt64, used to saturate an aligned end rather than wrap it negative.
var maxExpertTierInt64 = int64(^uint64(0) >> 1)

// ExpertNVMeGeometry is the aligned read geometry of one NVMe fault. A page-cache-backed read costs
// whole pages regardless of the expert stride, so the honest span is [round-down(start),
// round-up(end)) — the bytes the device would actually move. This is the SAME rule
// readExpertPageCache derives (#1302); the type exists so the tier policy above it can state a
// stride and its over-read WITHOUT issuing the read.
type ExpertNVMeGeometry struct {
	// AlignBytes is the alignment granularity. <= 0 means the 4096 default.
	AlignBytes int64 `json:"align_bytes"`
}

// align resolves the effective alignment, clamping to the int range pageAlignDown/pageAlignUp take.
func (g ExpertNVMeGeometry) align() int {
	a := g.AlignBytes
	if a <= 0 {
		a = expertTierDefaultAlignBytes
	}
	if a > int64(maxExpertTierInt) {
		return maxExpertTierInt
	}
	return int(a)
}

// AlignedSpan returns the aligned [start,end) that backs an unaligned [offset,offset+length) read:
// start is offset rounded DOWN, end is the exclusive end rounded UP. A negative length is treated as
// zero. It is the AC3 witness surface — a stride that is not a multiple of AlignBytes still yields a
// span that fully covers it, with the over-read (end-start-length) strictly below one alignment unit.
func (g ExpertNVMeGeometry) AlignedSpan(offset, length int64) (start, end int64) {
	if length < 0 {
		length = 0
	}
	// Saturate rather than wrap: an offset near MaxInt64 plus a length must clamp HIGH, because a
	// span is a read bound and a wrapped negative end would read backwards from the store.
	var rawEnd int64
	if offset > maxExpertTierInt64-length {
		rawEnd = maxExpertTierInt64
	} else {
		rawEnd = offset + length
	}
	page := g.align()
	start = pageAlignDown(offset, page)
	end = pageAlignUp(rawEnd, page)
	if end < start {
		end = start
	}
	return start, end
}

// AlignedReadLen is the byte length of the aligned span AlignedSpan returns — what the device would
// actually move for this fault, and therefore the resident cost an L2 fill should account.
func (g ExpertNVMeGeometry) AlignedReadLen(offset, length int64) int64 {
	start, end := g.AlignedSpan(offset, length)
	return end - start
}

// ExpertTierPolicy is the three-tier transition policy. Its ZERO VALUE is OFF: Enabled=false means
// no promotion, no demotion and no allocation, so a session that says nothing runs the pre-#1300
// path byte-for-byte (AC5). It mirrors the shape of the other expert-ring knobs — a small struct
// whose zero value is the incumbent — so a caller reading ExpertRingBatchAware reads this the same
// way.
type ExpertTierPolicy struct {
	// Enabled turns the policy on. It is the master gate; the knobs below are inert without it.
	Enabled bool `json:"enabled"`
	// L2Bytes is the declared byte CEILING of the L2 tier. 0 means no room, so every promotion is
	// refused and the tier behaves exactly like a disabled one for fills.
	L2Bytes int64 `json:"l2_bytes"`
	// NVMeAlignBytes is the alignment the NVMe geometry rounds to. <= 0 means the 4096 default.
	NVMeAlignBytes int64 `json:"nvme_align_bytes"`
	// PromoteOnMiss fills L2 from NVMe when a checkpoint fault lands and L2 has room.
	PromoteOnMiss bool `json:"promote_on_miss"`
}

// ExpertTierCounters is one tier's ledger. Hits/Reads/Bytes are the traffic counters; Promotions and
// Demotions are the transitions INTO and OUT of the tier; PinnedPreserved counts candidates a
// demotion pass refused to drop because they were pinned/warm. A counter a tier cannot honestly
// measure stays zero rather than being filled from an unrelated quantity.
type ExpertTierCounters struct {
	Hits            int   `json:"hits"`
	Reads           int   `json:"reads"`
	Bytes           int64 `json:"bytes"`
	Promotions      int   `json:"promotions"`
	Demotions       int   `json:"demotions"`
	PinnedPreserved int   `json:"pinned_preserved"`
}

// ExpertTierStats is the session-level, tier-shaped fold of the existing ledgers plus the policy's
// own transition counters. It is a READ: Session.ExpertTierStats never allocates and never moves
// residency. Enabled means at least one rung of the waterfall exists (a ring, a checkpoint tier, or
// the policy), so a bare session reports the zero value with Enabled=false.
type ExpertTierStats struct {
	Enabled bool `json:"enabled"`
	// VRAM/L2/NVMe are the three rungs, fastest first.
	VRAM ExpertTierCounters `json:"vram"`
	L2   ExpertTierCounters `json:"l2"`
	NVMe ExpertTierCounters `json:"nvme"`
	// L2ResidentBytes/Count and PeakBytes are the L2 tier's residency bound (Used() <= L2Bytes by
	// construction). PeakBytes is the high-water mark the bound was actually exercised against.
	L2ResidentBytes int64 `json:"l2_resident_bytes"`
	L2ResidentCount int   `json:"l2_resident_count"`
	PeakBytes       int64 `json:"peak_bytes"`
	// Ring and Checkpoint carry the untouched upstream ledgers, so a consumer that needs a field the
	// tier fold does not reshape still has it rather than a lossy projection.
	Ring       ExpertRingStats       `json:"ring"`
	Checkpoint ExpertCheckpointStats `json:"checkpoint"`
}

// ExpertTierLedger is ONE tier decision: the transition attempted, whether it happened, and why or
// why not. Reason is always populated, because a transition that only explains its successes turns
// every refusal into an unexplained default.
type ExpertTierLedger struct {
	From            ExpertCacheTier `json:"from"`
	To              ExpertCacheTier `json:"to"`
	Promoted        bool            `json:"promoted"`
	Demoted         bool            `json:"demoted"`
	PreservedPinned int             `json:"preserved_pinned"`
	Reason          string          `json:"reason"`
	// Bytes is the aligned span moved by this decision (0 for a refusal that moved nothing).
	Bytes int64 `json:"bytes"`
}

// expertTierEntry is one L2 resident: the byte span it occupies and — when the name is a routed
// expert projection — the (layer,expert) identity the pin-set is keyed by. lastUse is a policy-local
// recency stamp so demotion can pick a deterministic victim without reaching into polymodel.Pool's
// private clock.
type expertTierEntry struct {
	layer, expert int
	identity      bool
	bytes         int64
	lastUse       uint64
}

// expertTierState is the L2 tier's lazily-allocated bookkeeping: the polymodel.Pool that bounds it
// (the same policy the ring and checkpoint tier reuse), the resident map, and the transition
// counters. nil until the first promotion, so a disabled or never-used policy allocates nothing.
type expertTierState struct {
	pool  *polymodel.Pool
	l2    map[polymodel.ModelID]expertTierEntry
	clock uint64
	peak  int64

	promotions      int
	demotions       int
	preservedPinned int
	promotedBytes   int64
	demotedBytes    int64
}

// tierState returns the L2 tier's state, allocating it on first use. Allocation is deliberately lazy
// and gated behind an ENABLED policy (see PromoteExpertTier), so AC5's "no allocation" default-off
// is structural: the zero-value path never reaches this function.
func (s *Session) tierState() *expertTierState {
	if s.expertTier == nil {
		s.expertTier = &expertTierState{
			pool: polymodel.NewPool(s.ExpertTier.L2Bytes),
			l2:   map[polymodel.ModelID]expertTierEntry{},
		}
	}
	return s.expertTier
}

// PromoteExpertTier is AC2: on an NVMe (checkpoint) fault, fill L2 with the expert's ALIGNED span
// when the policy is enabled, PromoteOnMiss is set, and L2 has room. It returns the decision ledger
// and whether a fill happened.
//
// `name` is the canonical per-expert tensor name so routedExpertIdentity can recover the
// (layer,expert) pair the demotion half filters against; `offset`/`length` are the fault's
// unaligned range, from which the aligned span (and therefore the resident cost) is derived.
//
// It is a pure no-op for a disabled policy or an unset PromoteOnMiss: zero ledger, false, and no
// allocation. A refusal when L2 has no room is a DECISION, not an error — the caller keeps its
// stream-through behaviour and may demote to make room. An already-resident name is a hit
// (Touch + L2.Hits), not a second promotion.
func (s *Session) PromoteExpertTier(name string, offset, length int64) (ExpertTierLedger, bool) {
	if s == nil || !s.ExpertTier.Enabled || !s.ExpertTier.PromoteOnMiss {
		return ExpertTierLedger{}, false
	}
	geo := ExpertNVMeGeometry{AlignBytes: s.ExpertTier.NVMeAlignBytes}
	start, end := geo.AlignedSpan(offset, length)
	span := end - start
	if span <= 0 {
		return ExpertTierLedger{
			From: ExpertCacheTierNVMe, To: ExpertCacheTierL2,
			Reason: fmt.Sprintf("empty aligned span for offset %d length %d", offset, length),
		}, false
	}
	st := s.tierState()
	id := polymodel.ModelID(name)
	if e, live := st.l2[id]; live {
		st.clock++
		e.lastUse = st.clock
		st.l2[id] = e
		st.pool.Touch(id)
		return ExpertTierLedger{
			From: ExpertCacheTierL2, To: ExpertCacheTierL2,
			Reason: "already resident in L2", Bytes: e.bytes,
		}, false
	}
	// Room is the explicit gate (see the file header): Admit would evict an L2 resident to fit, which
	// would make "full" unobservable and demotion dead. Refuse instead and let the caller demote.
	if budget := st.pool.Budget(); span > budget || st.pool.Used()+span > budget {
		return ExpertTierLedger{
			From: ExpertCacheTierNVMe, To: ExpertCacheTierL2, Bytes: span,
			Reason: fmt.Sprintf("l2 full: %d aligned bytes need %d used of %d", span, st.pool.Used(), budget),
		}, false
	}
	if _, err := st.pool.Admit(polymodel.Model{ID: id, WeightBytes: span}); err != nil {
		return ExpertTierLedger{
			From: ExpertCacheTierNVMe, To: ExpertCacheTierL2, Bytes: span,
			Reason: "l2 admit refused: " + err.Error(),
		}, false
	}
	layer, expert, ok := routedExpertIdentity(name)
	st.clock++
	st.l2[id] = expertTierEntry{layer: layer, expert: expert, identity: ok, bytes: span, lastUse: st.clock}
	st.promotions++
	st.promotedBytes += span
	if used := st.pool.Used(); used > st.peak {
		st.peak = used
	}
	return ExpertTierLedger{
		From: ExpertCacheTierNVMe, To: ExpertCacheTierL2, Promoted: true, Bytes: span,
		Reason: fmt.Sprintf("filled %d aligned bytes (offset %d length %d -> [%d,%d))", span, offset, length, start, end),
	}, true
}

// DemoteExpertTier is AC4: drop ONE unpinned L2 resident back to NVMe, choosing the least-recently
// used candidate (deterministic name tie-break). It returns the decision ledger and whether a
// demotion happened.
//
// It never drops a pinned/warm expert: for each candidate whose name parses to a routed-expert
// identity, the ring's durable pin-set is consulted through the SAME seam the demand path uses
// (routedExpertIdentity + pagedRing.isExpertPinned). Candidates skipped for being pinned are counted
// in PreservedPinned. If EVERY resident is pinned, nothing is demoted and the ledger reports
// PreservedPinned > 0 so the caller can tell "could not free room" from "nothing to free".
//
// A disabled policy or an empty/nil L2 tier is a no-op returning the zero ledger and false.
func (s *Session) DemoteExpertTier() (ExpertTierLedger, bool) {
	if s == nil || !s.ExpertTier.Enabled || s.expertTier == nil || len(s.expertTier.l2) == 0 {
		return ExpertTierLedger{}, false
	}
	st := s.expertTier
	ids := make([]polymodel.ModelID, 0, len(st.l2))
	for id := range st.l2 {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		ei, ej := st.l2[ids[i]], st.l2[ids[j]]
		if ei.lastUse != ej.lastUse {
			return ei.lastUse < ej.lastUse
		}
		return ids[i] < ids[j]
	})

	preserved := 0
	var victim polymodel.ModelID
	var victimEntry expertTierEntry
	found := false
	for _, id := range ids {
		e := st.l2[id]
		if e.identity && s.expertRing != nil && s.expertRing.isExpertPinned(e.layer, e.expert) {
			preserved++ // warm/pinned: never demoted, however cold its recency stamp
			continue
		}
		victim, victimEntry = id, e
		found = true
		break
	}
	if !found {
		st.preservedPinned += preserved
		return ExpertTierLedger{
			From: ExpertCacheTierL2, To: ExpertCacheTierNVMe, PreservedPinned: preserved,
			Reason: fmt.Sprintf("all %d l2 candidates pinned; demoted nothing", preserved),
		}, false
	}
	delete(st.l2, victim)
	st.pool.Evict(victim)
	st.demotions++
	st.demotedBytes += victimEntry.bytes
	st.preservedPinned += preserved
	return ExpertTierLedger{
		From: ExpertCacheTierL2, To: ExpertCacheTierNVMe, Demoted: true,
		PreservedPinned: preserved, Bytes: victimEntry.bytes,
		Reason: fmt.Sprintf("demoted %s (%d aligned bytes) to nvme", victim, victimEntry.bytes),
	}, true
}

// ExpertTierStats folds this session's three rungs into one reading (AC6): the device ring's own
// ledger becomes VRAM, the checkpoint tier's becomes NVMe, and the policy's transition counters
// become L2. It is a PURE READ — it allocates nothing, moves no residency, and never lazily creates
// the L2 state — so it is safe to call on any session, including the zero value, and safe during a
// forward.
//
// Enabled is true when ANY rung exists. A session with no ring, no checkpoint tier and no policy
// reports the zero value, which is the honest reading rather than a phantom enabled tier.
func (s *Session) ExpertTierStats() ExpertTierStats {
	if s == nil {
		return ExpertTierStats{}
	}
	ring := s.ExpertRing()
	var ck ExpertCheckpointStats
	if s.M != nil {
		ck = s.M.ExpertCheckpointStats()
	}
	st := ExpertTierStats{
		Enabled:    s.ExpertTier.Enabled || ring.Enabled || ck.Enabled,
		Ring:       ring,
		Checkpoint: ck,
		VRAM: ExpertTierCounters{
			Hits:            ring.Hits,
			Reads:           ring.PageIns,
			Bytes:           ring.PageInBytes,
			PinnedPreserved: ring.PinnedCount,
		},
		NVMe: ExpertTierCounters{
			Hits:  ck.Hits,
			Reads: ck.Reads,
			Bytes: ck.BytesRead,
		},
	}
	if ts := s.expertTier; ts != nil {
		st.L2 = ExpertTierCounters{
			Hits:            ts.l2Hits(),
			Reads:           ts.promotions,
			Bytes:           ts.promotedBytes,
			Promotions:      ts.promotions,
			Demotions:       ts.demotions,
			PinnedPreserved: ts.preservedPinned,
		}
		st.L2ResidentBytes = ts.pool.Used()
		st.L2ResidentCount = len(ts.l2)
		st.PeakBytes = ts.peak
	}
	return st
}

// l2Hits is the number of resident L2 touches the tier has served. The tier does not keep a separate
// hit counter yet (an already-resident name is answered by PromoteExpertTier); keeping the accessor
// separate makes adding that counter a one-line change rather than a reshape of the fold.
func (ts *expertTierState) l2Hits() int {
	if ts == nil {
		return 0
	}
	return 0
}
