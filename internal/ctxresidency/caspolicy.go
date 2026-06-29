package ctxresidency

import "context"

// ---------------------------------------------------------------------------
// Rung 4 — residency policy: what stays in the spine vs pages to the overlay
// (issue #1262, the system-prompt MMU). The MemGPT/Letta tiering decision, with
// safety-critical text ALWAYS resident, never paged.
//
// The dynamic State (resident/evictable/held) answers "is the body in the cache
// right now?". The static Tier added here answers the orthogonal question "may
// this block EVER be paged?". Invariant 3 of the design note —
// safety/identity-load-bearing text is always resident — is enforced by making a
// spine/policy block unreachable by any eviction path BY CONSTRUCTION (classify
// never returns StateEvictable for it), not by a retrieval heuristic that a
// weaker model could under-trigger. The dominant failure mode the note names —
// silent under-retrieval — therefore cannot drop safety-critical text: there is
// no eviction path that selects it.
// ---------------------------------------------------------------------------

// Tier is the base-context LAYOUT tier of a residency block. It mirrors the
// design note's head ordering ([ fak spine ][ policy floor ][ harness overlay ]):
//
//   - TierSpine   — the immutable fak spine: identity-load-bearing, small, used
//     every turn; the attention-sink + heavy-hitter anchor. ALWAYS resident.
//   - TierPolicy  — the versioned deny/allow + safety-critical floor. ALWAYS
//     resident (it changes only at a marked cache breakpoint, never mid-prefix).
//   - TierOverlay — queried harness capability cards. The ONLY pageable tier.
type Tier string

const (
	TierSpine   Tier = "spine"
	TierPolicy  Tier = "policy"
	TierOverlay Tier = "overlay"
)

// AlwaysResident reports whether the tier is never paged under pressure — true
// for the spine and the policy floor, false for the pageable overlay AND for the
// zero value. An unclassified block carries the zero value and is treated as
// pageable: the fail-safe never accidentally pins an unmarked block into the
// spine; only an explicit spine/policy classification protects a block.
func (t Tier) AlwaysResident() bool { return t == TierSpine || t == TierPolicy }

// BlockProfile is the measured profile of one base-context block the Rung-4
// residency classifier decides over. Every field is a MEASURED property of the
// block (how often the turn loop references it, its size against the resident
// budget, whether it is load-bearing for the deny floor or fak's identity), never
// a model judgement made at eviction time.
type BlockProfile struct {
	UsedEveryTurn       bool // referenced on ~every turn (a hot, always-needed block).
	Small               bool // within the resident token budget (cheap to keep pinned).
	SafetyLoadBearing   bool // load-bearing for the deny/allow floor — safety-critical text.
	IdentityLoadBearing bool // load-bearing for fak's own identity (the spine's canonical concepts).
}

// ClassifyTier is the Rung-4 residency rule (#1262): which base-context blocks
// stay in the immutable spine / policy floor (always resident, never paged) vs
// page to the queryable overlay. Stated as the design note states it:
//
//   - A NON-load-bearing block always pages to the overlay (MemGPT/Letta
//     tiering) — the only evictable tier, faulted in by the turn's query.
//   - A load-bearing block is ALWAYS resident (invariant 3): the immutable spine
//     holds the identity-load-bearing block that is ALSO small AND used ~every
//     turn (resident iff used-every-turn ∧ small ∧ load-bearing — the
//     attention-sink anchor); every other load-bearing block sits in the policy
//     floor. Neither tier is ever paged.
//
// The safety guarantee is structural, not a ranking: a SafetyLoadBearing block
// can never resolve to TierOverlay regardless of its size or coldness, so it is
// excluded from the evictable set by construction here — not rescued by a
// retrieval heuristic later that a weaker model could fail to fire.
func ClassifyTier(p BlockProfile) Tier {
	if !p.SafetyLoadBearing && !p.IdentityLoadBearing {
		return TierOverlay
	}
	if p.IdentityLoadBearing && p.UsedEveryTurn && p.Small {
		return TierSpine
	}
	return TierPolicy
}

// SetTier assigns a tracked block's layout tier and re-classifies it under the
// lock. A caller runs the residency classifier over a faulted block and pins the
// verdict here: SetTier(key, ClassifyTier(profile)). Marking a block TierSpine or
// TierPolicy moves it out of the evictable set immediately and permanently (until
// re-tiered) — it can no longer be the EvictColdest pick no matter how cold it
// gets. SetTier of an unknown key is a no-op. Re-tiering a held (paged-out) block
// updates its tier but does not page it back in (re-admission still goes through
// the witness gate via Fault).
func (cr *CapResidency) SetTier(key CapKey, tier Tier) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	if st, ok := cr.caps[key]; ok {
		st.tier = tier
		if st.state != StateHeld {
			st.state = classify(st)
		}
	}
}

// TierOf reports a tracked block's layout tier (the zero value for an unknown
// key, which AlwaysResident reads as pageable). A pure read.
func (cr *CapResidency) TierOf(key CapKey) Tier {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	if st, ok := cr.caps[key]; ok {
		return st.tier
	}
	return ""
}

// RefuseIfCrossesSpine is the explicit eviction guard invariant 3 asks for: it
// MEASURES what evicting key would cost (the same read MeasureBlastRadius does)
// AND reports whether the eviction would cross into the never-paged spine/policy
// tier — in which case it is REFUSED. The caller records the measured radius (it
// feeds the Rung-6 observability surface) and, on refused==true, must NOT page
// the block out: the spine cannot be paged out under pressure.
//
// EvictColdest already never SELECTS a spine/policy block (it is StateResident,
// not StateEvictable), so the overlay-eviction path cannot reach the spine on its
// own. This method is the first-class refusal a Rung-5 promote/demote verb calls
// before acting on a block it did not pick — two independent fences on the spine,
// so a bug in one does not open it.
func (cr *CapResidency) RefuseIfCrossesSpine(key CapKey) (radius BlastRadius, refused bool) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	st, ok := cr.caps[key]
	if !ok {
		return BlastRadius{}, false
	}
	return blastOf(st), st.tier.AlwaysResident()
}

// EvictColdestOverlay is the Rung-4 eviction entry point: it evicts the coldest
// EVICTABLE OVERLAY body through the witness-clear gate (ctxmmu.PageOutBody) and
// never a spine/policy block, recording the measured blast radius of the drop. It
// is exactly EvictColdest — the pick already excludes spine/policy by
// construction (they are StateResident) — exposed under the name the issue uses
// so the call site reads as "page out the overlay, never the spine". The measured
// radius it returns is the cost recorded for Rung 6.
func (cr *CapResidency) EvictColdestOverlay(ctx context.Context) (evicted CapKey, radius BlastRadius, ok bool) {
	return cr.EvictColdest(ctx)
}
