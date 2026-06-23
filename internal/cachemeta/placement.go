package cachemeta

// placement.go is the hardware-cost-driven placement policy: given an entry's
// lifecycle, its size, the tier PROFILES, and the current per-tier PRESSURE, it
// decides whether to keep, promote, demote, spill, or evict — and emits the existing
// KVTransfer directive for the move. This is the decision a blind LRU cache cannot
// make, and the reason the cache is co-optimized with the hardware rather than merely
// running on it.
//
// The core move is DEMOTE-INSTEAD-OF-EVICT. Under memory pressure an LRU cache drops
// its coldest leaf, and a later request that wanted it pays a full re-prefill. A
// tiered, hardware-aware cache instead RELOCATES that span one tier colder — to NUMA-
// far DRAM or CXL — where it stays byte-addressable and attendable in place, costing
// only a one-time staging move (bytes / bandwidth) instead of a recompute (tokens x
// per-token prefill). Demotion wins precisely when the colder tier still has room and
// the move is cheaper than the recompute it avoids; eviction is the right call only
// when nothing colder has room, or the span is so cheap to recompute that holding it
// in scarce memory is not worth it. PlanPlacement makes that comparison explicit.
//
// Lifecycle.Advance decided WHEN the entry stopped being fresh; PlanPlacement decides
// WHERE it goes now. It is pure and deterministic (the caller injects nowMillis), and
// it never moves bytes: it returns a PlacementDecision whose Directive a consumer
// feeds to engine.CacheEvent / an engine adapter that performs the physical transfer.

// TierPressure is the fullness of each tier in [0,1] (0 = empty, 1 = full). A tier
// absent from the map is treated as having room (pressure 0). It is the live signal
// that turns the static tier ladder into a dynamic placement: a span demotes only as
// far as the first colder tier that is not already full.
type TierPressure map[ResidencyTier]float64

// HasRoom reports whether a tier can accept more: pressure strictly below 1.0. A tier
// at/above 1.0 is full. (A caller wanting a high-water mark below 1.0 simply scales the
// pressure values it stores.)
func (tp TierPressure) HasRoom(t ResidencyTier) bool {
	return tp[t] < 1.0
}

// PlacementAction is the recommended move for an entry.
type PlacementAction string

const (
	ActionKeep    PlacementAction = "keep"    // leave the entry where it is
	ActionPromote PlacementAction = "promote" // relocate to a hotter tier (it is hot)
	ActionDemote  PlacementAction = "demote"  // relocate to a colder attendable tier
	ActionSpill   PlacementAction = "spill"   // relocate to a colder NON-attendable tier (disk)
	ActionEvict   PlacementAction = "evict"   // drop it; recompute on demand
)

// PlacementDecision is the planner's verdict: the action, the tiers involved, the
// KVTransfer directive a consumer can act on (KVOffload for demote/spill/evict-stage,
// KVRestore for promote), an estimate of the bytes the move would stage, and a
// human/metric-readable reason.
type PlacementDecision struct {
	Action       PlacementAction
	FromTier     ResidencyTier
	ToTier       ResidencyTier
	Directive    KVTransferDirection
	EstMoveBytes int64
	Reason       string
}

// PlacementRequest bundles what the planner needs. SizeBytes and Tokens describe the
// payload (bytes drive the move cost, tokens drive the recompute cost);
// PerTokenPrefillNanos is the model/hardware cost of re-prefilling one token, the
// quantity demotion is weighed against. PromoteAccessRate, when > 0, is the hits/sec
// above which a cold-but-hot entry is recommended for promotion.
type PlacementRequest struct {
	Lifecycle            Lifecycle
	SizeBytes            int64
	Tokens               int64
	Profiles             map[ResidencyTier]TierProfile
	Pressure             TierPressure
	Policy               LifecyclePolicy
	PerTokenPrefillNanos int64
	PromoteAccessRate    float64
	NowMillis            int64
}

// stageNanos estimates the one-time cost of moving SizeBytes into a tier at its
// streaming bandwidth (plus its first-byte latency). 0 bandwidth => a large sentinel
// so an unprofiled/zero-bandwidth tier never looks cheap.
func stageNanos(sizeBytes int64, p TierProfile) int64 {
	if p.BandwidthMBPerSec <= 0 {
		return 1 << 62
	}
	// bytes / (MB/s) -> seconds; * 1e9 -> ns. Done in ns to avoid float drift:
	// sizeBytes * 1e9 / (BandwidthMBPerSec * 1e6) == sizeBytes * 1000 / BandwidthMBPerSec.
	return p.ReadLatencyNanos + sizeBytes*1000/p.BandwidthMBPerSec
}

// recomputeNanos estimates the cost of re-prefilling a span of Tokens.
func recomputeNanos(tokens, perTokenPrefillNanos int64) int64 {
	if tokens <= 0 || perTokenPrefillNanos <= 0 {
		return 0
	}
	return tokens * perTokenPrefillNanos
}

// RetainCheaperThanRecompute reports whether STAGING the span into tier `to` is
// cheaper than EVICTING it and recomputing later — the quantified core of
// demote-vs-evict. When recompute is effectively free (no token/cost info, or a tiny
// span) eviction wins; when the span is large and expensive to rebuild, retaining it
// even in slow far memory wins.
func RetainCheaperThanRecompute(sizeBytes, tokens, perTokenPrefillNanos int64, to TierProfile) bool {
	return stageNanos(sizeBytes, to) < recomputeNanos(tokens, perTokenPrefillNanos)
}

// coldestColderWithRoom walks the demote ladder from the entry's current tier and
// returns the first colder LOCAL tier that is profiled and has room, or TierUnknown if
// none (the ladder bottomed out at disk-full / recompute).
func coldestColderWithRoom(from ResidencyTier, req PlacementRequest) ResidencyTier {
	for t := NextColderTier(from); t != TierUnknown && t != TierRecompute; t = NextColderTier(t) {
		if _, ok := req.Profiles[t]; !ok {
			continue // this box does not have that tier
		}
		if req.Pressure.HasRoom(t) {
			return t
		}
	}
	return TierUnknown
}

// PlanPlacement is the hardware-aware placement decision. The logic, in order:
//
//  1. PROMOTE: an entry sitting below the hottest tier whose measured access rate
//     clears PromoteAccessRate, when a warmer tier has room — it is hot enough to earn
//     a faster seat.
//  2. Under pressure or expiry, RELOCATE rather than evict when possible:
//     - find the nearest colder profiled tier with room;
//     - if that tier is attendable in place (NUMA-far/CXL) and retaining there beats
//     recompute, DEMOTE to it (KVOffload);
//     - else if it is a spill tier (disk) and retaining still beats recompute, SPILL;
//     - else (nothing colder has room, or recompute is cheaper than any retain) EVICT.
//  3. Otherwise KEEP.
//
// "Under pressure" means the current tier is full (no room) OR the lifecycle has
// reached Expiring/Expired. An Expired entry whose policy forbids demotion evicts
// directly.
func PlanPlacement(req PlacementRequest) PlacementDecision {
	lc := req.Lifecycle
	from := lc.Tier

	// (1) Promote a hot entry that is not already in the hottest tier. Only a genuinely
	// RESIDENT entry qualifies — an Expiring/Expired one has been declared due to turn
	// over in its tier and must not be moved UP into scarcer (shorter-TTL) memory — and
	// only one with CURRENT-tier heat: LastAccessMillis must be at/after EnteredTierMillis
	// (touched since arriving here), so a stale entry with a high lifetime count cannot
	// promote on yesterday's heat.
	recentInTier := lc.LastAccessMillis >= lc.EnteredTierMillis
	if req.PromoteAccessRate > 0 && lc.State == StateResident && recentInTier {
		if warmer := NextWarmerTier(from); warmer != TierUnknown {
			if _, ok := req.Profiles[warmer]; ok && req.Pressure.HasRoom(warmer) &&
				lc.AccessRatePerSec(req.NowMillis) >= req.PromoteAccessRate {
				return PlacementDecision{
					Action: ActionPromote, FromTier: from, ToTier: warmer,
					Directive: KVRestore, EstMoveBytes: req.SizeBytes,
					Reason: "hot_entry_promoted",
				}
			}
		}
	}

	underPressure := !req.Pressure.HasRoom(from)
	expiring := lc.State == StateExpiring || lc.State == StateExpired

	if !underPressure && !expiring {
		return PlacementDecision{Action: ActionKeep, FromTier: from, ToTier: from, Reason: "fresh_and_room"}
	}

	// An expired entry whose policy forbids relocation is dropped outright.
	if lc.State == StateExpired && !req.Policy.DemoteOnExpiry {
		return PlacementDecision{
			Action: ActionEvict, FromTier: from, ToTier: TierRecompute,
			Directive: KVOffload, Reason: "expired_no_demote",
		}
	}

	to := coldestColderWithRoom(from, req)
	if to == TierUnknown {
		// Nothing colder can hold it — eviction is forced.
		return PlacementDecision{
			Action: ActionEvict, FromTier: from, ToTier: TierRecompute,
			Directive: KVOffload, Reason: "no_colder_tier_with_room",
		}
	}

	toProfile := req.Profiles[to]
	if !RetainCheaperThanRecompute(req.SizeBytes, req.Tokens, req.PerTokenPrefillNanos, toProfile) {
		// Cheaper to rebuild than to hold even in the colder tier — let it go.
		return PlacementDecision{
			Action: ActionEvict, FromTier: from, ToTier: TierRecompute,
			Directive: KVOffload, Reason: "recompute_cheaper_than_retain",
		}
	}

	action := ActionDemote
	reason := "demote_beats_recompute"
	if !toProfile.AttendableInPlace() {
		action = ActionSpill
		reason = "spill_beats_recompute"
	}
	return PlacementDecision{
		Action: action, FromTier: from, ToTier: to,
		Directive: KVOffload, EstMoveBytes: req.SizeBytes, Reason: reason,
	}
}

// Apply executes a placement decision against a lifecycle, returning the updated
// lifecycle and the transfer directive that was performed (KVTransferDirection ""
// for a Keep). It is the convenience that pairs PlanPlacement with Lifecycle.MoveTo /
// Evict so a caller does not re-implement the state transition.
func (d PlacementDecision) Apply(lc Lifecycle, profiles map[ResidencyTier]TierProfile, nowMillis int64) (Lifecycle, KVTransferDirection) {
	switch d.Action {
	case ActionPromote, ActionDemote, ActionSpill:
		return lc.MoveTo(d.ToTier, profiles, nowMillis)
	case ActionEvict:
		return lc.Evict(nowMillis), KVOffload
	default:
		return lc, ""
	}
}
