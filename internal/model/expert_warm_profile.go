package model

import (
	"fmt"
	"sort"
)

// expert_warm_profile.go — CW-10 of the agent-startup cache-warming chain (#13349, parent
// #12640, epic #12952): the DETERMINISTIC SELECTION leaf.
//
// What this leaf is. A recorded demand-access profile names the (layer, expert) pairs a previous
// workload actually routed. This file turns that profile — plus the checkpoint tier's own index,
// which is the authority on WHICH projections exist and how many bytes each one costs — into a
// deduplicated ExpertWarmPlan whose selected bytes fit a caller's budget. It reads no payload and
// faults nothing: every name and byte count comes from the tier's index, so a warm plan is a pure
// function of (profile, tier index, budget).
//
// What this leaf is NOT. Executing the plan (issuing the reads and populating the retained host
// cache) is #13343, deliberately a separate leaf. This file selects; it does not warm. Keeping
// selection and execution apart is what lets the plan be unit-tested without a real checkpoint and
// what keeps the "warm touches never feed the demand histogram" invariant structural: selection
// observes only the profile it was handed and never the live router.
//
// Identity. A plan is only meaningful against the checkpoint, quantization and layout the profile
// was recorded from; an expert index reused across a re-quantization or a re-layout names different
// bytes. ExpertWarmProfileIdentity is the caller-supplied witness of that agreement, and a stale
// profile (mismatched identity) produces an empty plan — never a warm against the wrong bytes.
//
// Fail-closed posture. An absent, empty or stale profile is an empty plan with a named reason. A
// profile entry naming a projection the tier does not index is dropped, not guessed at. An entry
// larger than the whole budget is skipped rather than allowed to overrun it. Nothing here invents a
// projection, a byte count or a cache hit.

// ExpertWarmProfileIdentity is the (checkpoint, quantization, layout) identity a demand profile was
// recorded against. A profile whose identity does not match the identity the caller presents is
// stale: its (layer, expert) ordinals name different resident bytes after a re-quantization or a
// re-layout, so selecting from it would warm the wrong experts. Every field is a caller-supplied
// string/content witness; this file does not derive it from a file path or a clock.
type ExpertWarmProfileIdentity struct {
	// Checkpoint is the checkpoint artifact identity (e.g. a revision or content digest).
	Checkpoint string
	// Quantization is the expert-slab representation identity (e.g. "Q2_K").
	Quantization string
	// Layout is the tensor-layout identity (e.g. the fused-tensor naming scheme).
	Layout string
}

// Equal reports whether two identities agree on every field. An empty identity equals only another
// empty identity, so a profile that never recorded an identity can never be treated as matching a
// caller that presents one.
func (id ExpertWarmProfileIdentity) Equal(other ExpertWarmProfileIdentity) bool {
	return id.Checkpoint == other.Checkpoint &&
		id.Quantization == other.Quantization &&
		id.Layout == other.Layout
}

// ExpertWarmDemandEntry is one row of a recorded demand profile: how many times a workload routed
// the (layer, expert) pair. Count is a float so the online-learning heat model in
// expert_warmpins.go (which both sums across turns and decays inside its actuator) can be the
// producer without a lossy conversion; a non-positive count is demand-free and is ignored.
type ExpertWarmDemandEntry struct {
	Layer  int
	Expert int
	Count  float64
}

// ExpertWarmProfile is a recorded demand-access profile: the identity it was recorded against plus
// its rows. It is intentionally a plain value — the durable histogram in expert_warmpins.go is one
// producer, but the selection leaf depends only on this shape, so a test or a different recorder
// can supply one without touching the on-disk histogram format.
type ExpertWarmProfile struct {
	Identity ExpertWarmProfileIdentity
	Entries  []ExpertWarmDemandEntry
}

// ExpertWarmReason is the closed verdict vocabulary for a warm plan. It is the machine-checkable
// "why" a caller (and the execution leaf #13343) branches on, so an empty plan is never mistaken
// for "nothing to do" when the real cause was a stale profile or a zero budget.
type ExpertWarmReason string

const (
	// ExpertWarmReasonPlanned: at least one expert was selected within budget.
	ExpertWarmReasonPlanned ExpertWarmReason = "planned"
	// ExpertWarmReasonNoProfile: the profile carried no entries (absent or empty).
	ExpertWarmReasonNoProfile ExpertWarmReason = "no_profile"
	// ExpertWarmReasonStaleIdentity: the profile's identity did not match the caller's.
	ExpertWarmReasonStaleIdentity ExpertWarmReason = "stale_identity"
	// ExpertWarmReasonBudgetZero: the tier declares no positive residency budget, so nothing can be
	// retained; warming would stream bytes through and drop them. Selection declines rather than
	// issuing reads that cannot become residency.
	ExpertWarmReasonBudgetZero ExpertWarmReason = "budget_zero"
	// ExpertWarmReasonNoIndexedExpert: the profile named experts, but none is indexed by the tier.
	ExpertWarmReasonNoIndexedExpert ExpertWarmReason = "no_indexed_expert"
	// ExpertWarmReasonNoFit: indexed experts existed, but none fit the remaining budget.
	ExpertWarmReasonNoFit ExpertWarmReason = "no_fit"
)

// ExpertWarmSelection is one selected projection in a plan: the canonical per-expert weight name
// (the key the tier faults and the ring stages), the (layer, expert) identity it belongs to, the
// demand that ranked it, and the resident byte cost the tier reports for it.
type ExpertWarmSelection struct {
	Name    string
	Layer   int
	Expert  int
	Demand  float64
	Bytes   int64
	Project string
}

// ExpertWarmPlan is the deterministic result of selecting warm candidates. Selected is ordered by
// (demand desc, layer asc, expert asc, name asc) — a total order, so two plans from the same inputs
// are byte-identical. The *Bytes fields are the accounting the execution leaf and its receipt cite.
type ExpertWarmPlan struct {
	Reason     ExpertWarmReason
	Identity   ExpertWarmProfileIdentity
	Selected   []ExpertWarmSelection
	Bytes      int64
	Budget     int64
	Skipped    int
	NoFit      int
	NotIndexed int
}

// Empty reports whether the plan selects nothing to warm. It is true for every non-planned reason,
// so a caller can treat "no plan" uniformly while still reading Reason for the cause.
func (p ExpertWarmPlan) Empty() bool { return len(p.Selected) == 0 }

// expertWarmPlanCandidate is one indexed projection a profile row implies. The byte cost comes from
// the tier's index (entry.stride), never from the profile, so a profile cannot understate the bytes
// a warm would move.
type expertWarmPlanCandidate struct {
	name    string
	layer   int
	expert  int
	demand  float64
	bytes   int64
	project string
}

// ExpertWarmPlanFor selects the warm candidates a profile implies for this tier under budgetBytes.
//
// Identity first, then budget, then demand. The checks are ordered so the reason is the LOUDEST
// applicable cause: a stale profile is reported stale even if the budget is also zero, because a
// caller holding a stale profile must fix the profile before any budget discussion matters.
//
// budgetBytes <= 0 declines with budget_zero unconditionally. A zero-budget tier retains nothing
// (see NewExpertCheckpointTier), so selecting a warm set against it would issue reads nothing keeps.
//
// The vanilla path (no tier, no profile): returns no_profile. It never panics and never touches the
// index, so the default-off binary is unchanged.
func (t *ExpertCheckpointTier) ExpertWarmPlanFor(profile ExpertWarmProfile, identity ExpertWarmProfileIdentity, budgetBytes int64) ExpertWarmPlan {
	plan := ExpertWarmPlan{Identity: identity, Budget: budgetBytes}
	if budgetBytes <= 0 {
		plan.Reason = ExpertWarmReasonBudgetZero
		return plan
	}
	if len(profile.Entries) == 0 {
		plan.Reason = ExpertWarmReasonNoProfile
		return plan
	}
	if !profile.Identity.Equal(identity) {
		plan.Reason = ExpertWarmReasonStaleIdentity
		return plan
	}
	if t == nil {
		// A profile exists but there is no checkpoint tier to warm from; every row is unindexed.
		plan.Reason = ExpertWarmReasonNoIndexedExpert
		plan.NotIndexed = len(profile.Entries)
		return plan
	}

	candidates := t.warmPlanCandidates(profile)
	if len(candidates) == 0 {
		plan.Reason = ExpertWarmReasonNoIndexedExpert
		plan.NotIndexed = len(profile.Entries)
		return plan
	}

	// Deterministic rank: demand desc, layer asc, expert asc, name asc. A stable total order is what
	// makes the plan reproducible; map iteration order must never leak into the selected set.
	sort.Slice(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		if a.demand != b.demand {
			return a.demand > b.demand
		}
		if a.layer != b.layer {
			return a.layer < b.layer
		}
		if a.expert != b.expert {
			return a.expert < b.expert
		}
		return a.name < b.name
	})

	var used int64
	for _, c := range candidates {
		if c.bytes > budgetBytes-used {
			// An expert larger than the whole remaining budget is skipped, never allowed to overrun
			// it: a warm set that does not fit is worse than a smaller one that does.
			plan.NoFit++
			continue
		}
		used += c.bytes
		plan.Selected = append(plan.Selected, ExpertWarmSelection{
			Name: c.name, Layer: c.layer, Expert: c.expert,
			Demand: c.demand, Bytes: c.bytes, Project: c.project,
		})
	}
	plan.Bytes = used
	plan.Skipped = plan.NotIndexed + plan.NoFit
	if len(plan.Selected) == 0 {
		plan.Reason = ExpertWarmReasonNoFit
		return plan
	}
	plan.Reason = ExpertWarmReasonPlanned
	return plan
}

// warmPlanCandidates resolves each profile row to the projections the tier indexes, deduplicating
// by canonical name. The tier indexes three projections per routed expert (w1/w3/w2); a demand row
// for (layer, expert) therefore implies all three, because a warm set that kept only one projection
// would still fault the other two on the first real routing.
//
// A row whose projection is not indexed contributes nothing — the tier is the authority on what
// exists, and a profile recorded against a different (larger) checkpoint must not make this tier
// warm names it does not carry. Cacheability (a ring hit) is NOT considered here: this leaf plans
// what a warm would retain, and the execution leaf decides against the live ring.
func (t *ExpertCheckpointTier) warmPlanCandidates(profile ExpertWarmProfile) []expertWarmPlanCandidate {
	t.mu.Lock()
	index := t.index
	t.mu.Unlock()

	seen := make(map[string]struct{}, len(profile.Entries)*3)
	out := make([]expertWarmPlanCandidate, 0, len(profile.Entries)*3)
	for _, row := range profile.Entries {
		if row.Layer < 0 || row.Expert < 0 || row.Count <= 0 {
			continue
		}
		for proj, leaf := range v41ExpertLeafMap {
			name := layerName(row.Layer, "ffn.experts."+itoa(row.Expert)+"."+leaf+".weight")
			if _, dup := seen[name]; dup {
				continue
			}
			entry, ok := index[name]
			if !ok {
				// Try the historical spelling too, so a non-V4.1 profile selects correctly.
				name = expertName(row.Layer, row.Expert, proj+".weight")
				if _, dup := seen[name]; dup {
					continue
				}
				entry, ok = index[name]
				if !ok {
					continue
				}
			}
			seen[name] = struct{}{}
			out = append(out, expertWarmPlanCandidate{
				name: name, layer: row.Layer, expert: row.Expert,
				demand: row.Count, bytes: entry.stride, project: proj,
			})
		}
	}
	return out
}

// WarmPlanString renders a plan for a receipt or an error. It names the reason and the byte
// accounting, never a fabricated hit: an execution leaf may only claim residency it later witnesses.
func (p ExpertWarmPlan) String() string {
	return fmt.Sprintf("v41 expert warm plan reason=%s selected=%d bytes=%d/%d not_indexed=%d no_fit=%d",
		p.Reason, len(p.Selected), p.Bytes, p.Budget, p.NotIndexed, p.NoFit)
}
