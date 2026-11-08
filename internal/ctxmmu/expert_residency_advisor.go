package ctxmmu

// expert_residency_advisor.go — the seam between an ONLINE hot-set learner and this
// residency policy (cluster-bringup P2-4; parent plan
// fak-private#cluster-bringup-rtx6000-b300-readiness-20261104).
//
// The learner lives in internal/model (ExpertHotSetLearner, expert_hot_set.go): it accumulates
// EPLB-style activation statistics and commits a hot-set under a hysteresis band, so a lukewarm
// challenger cannot displace a hot incumbent. The shared expertaccess descriptor keeps
// the model-side learner independent of this higher-tier policy. The policy declares
// the decisions it delegates (ExpertResidencyAdvisor), and the model side ships an adapter
// (model.ExpertHotSetAdvisor) that implements it from the learner. The serve-path hookup of that
// adapter is the documented one-line follow-up; this file lands the loop fully unit-tested.
//
// Two residency decisions are delegated, both at the quiescent boundary the learner already owns:
//
//  1. Decode hot-set admission/eviction. A cold decode touch that would evict a resident is
//     admitted only when its heat beats the victim's heat through the advisor's displacement
//     band (the learner's own margin); inside the band the touch is BYPASSED — served from
//     backing without becoming resident — exactly the offline policy's bypass decision
//     (expert_residency_lfu.go: `hot > victim + victim/4 + 4`).
//  2. Prefill transient-ring sizing. The advisor reports how many distinct units it has observed
//     and how many it committed as hot; SizeTransientRing turns that breadth into a byte budget
//     for the separate prefill ring, which by construction can only churn itself, never the hot
//     set the learner protects.
//
// Membership stays honest in both directions: every hot-set insert/eviction the policy performs
// is reported back through MarkResident/ClearResident, so the learner's resident-first ranking
// partition tracks the ring's actual contents rather than a stale copy.

// ExpertResidencyAdvisor is the online-learning seam ExpertResidencyPolicy consults. The public
// engine's ExpertHotSetLearner implements it via model.ExpertHotSetAdvisor; a nil advisor on the
// policy restores the plain-LRU behavior byte-for-byte.
type ExpertResidencyAdvisor interface {
	// Heat returns the accumulated activation heat for (layer, expert); 0 for a unit the
	// advisor has never observed. Both sides of a gate read their heat from here, so the
	// policy never carries a second, drifting heat model.
	Heat(layer, expert int) float64
	// Admit reports whether a challenger carrying chHeat may displace a resident carrying
	// vHeat under the advisor's displacement band. The band is the advisor's own (the
	// learner's hysteresis margin plus its constant addend); the policy only asks.
	Admit(chHeat, vHeat float64) bool
	// HotSet returns the committed hot-set, hottest first, each with its current heat.
	// WeightBytes is 0: the learner tracks activations, not bytes, and the policy loads
	// each descriptor from backing when it seeds the ring.
	HotSet() []ExpertAccess
	// ObservedCount is how many distinct (layer, expert) units the advisor has ever
	// observed — the working-set breadth SizeTransientRing budgets against.
	ObservedCount() int
	// MarkResident declares (layer, expert) currently resident in the hot set; the
	// learner's total order ranks residents first. Idempotent.
	MarkResident(layer, expert int)
	// ClearResident removes (layer, expert) from the resident partition. No-op for a unit
	// that is not marked.
	ClearResident(layer, expert int)
}

// NewExpertResidencyPolicyWithAdvisor builds a policy whose cold-miss admission is gated by
// advisor's displacement band and which can warm-start from advisor's committed hot-set. A nil
// advisor is equivalent to NewExpertResidencyPolicy: plain-LRU admission, no warm-start source,
// and DecodeBypasses stays 0 for the policy's whole life.
func NewExpertResidencyPolicyWithAdvisor(opts ExpertResidencyOptions, backing ResidencyBacking, advisor ExpertResidencyAdvisor) (*ExpertResidencyPolicy, error) {
	p, err := NewExpertResidencyPolicy(opts, backing)
	if err != nil {
		return nil, err
	}
	p.advisor = advisor
	return p, nil
}

// WarmStartFromAdvisor seeds the resident hot set from the advisor's committed hot-set — the
// learned-pins twin of WarmStart's trace-frequency seeding: the units the learner committed
// (hottest first, its total order) are loaded from backing once and promoted into the ring,
// stopping at the budget rather than churning the set being built. Units already resident are
// skipped; a descriptor backing cannot supply is skipped, not an error. It returns the number
// of experts seeded. A policy with no advisor seeds nothing.
func (p *ExpertResidencyPolicy) WarmStartFromAdvisor() int {
	if p == nil || p.advisor == nil {
		return 0
	}
	seeded := 0
	for _, u := range p.advisor.HotSet() {
		if u.Layer < 0 || u.Expert < 0 {
			continue // identity guard, the same one WarmStart applies
		}
		key := expertResidencyKey{u.Layer, u.Expert}
		if p.resident.has(key) {
			continue
		}
		d, ok := p.load(key.layer, key.expert)
		if !ok {
			continue
		}
		// Seed the budget hottest that FIT: warm start is one-shot seeding, not a replay, so it
		// stops at the budget instead of evicting an already-seeded hotter unit.
		if p.resident.used+d.Bytes > p.resident.budget {
			break
		}
		p.promote(key, d)
		seeded++
	}
	return seeded
}

// SizeTransientRing sizes the prefill transient ring from the advisor's observed working-set
// breadth: every observed unit OUTSIDE the committed hot-set is a candidate cold touch a prefill
// sweep may stage, so the ring budgets (observed - hotSet) entries at avgExpertBytes each.
//
// It is advisory sizing input, not a claim: the operator still picks the final TransientBytes,
// and the result is clamped to capBytes. Non-positive inputs yield 0 (nothing to size). The
// overflow guard is exact: n > capBytes/avgExpertBytes proves n*avgExpertBytes > capBytes, so the
// clamped arm is returned without forming the product; otherwise n*avgExpertBytes <= capBytes and
// the multiplication cannot overflow.
func SizeTransientRing(observed, hotSet int, avgExpertBytes, capBytes int64) int64 {
	if observed <= 0 || avgExpertBytes <= 0 || capBytes <= 0 {
		return 0
	}
	cold := observed - hotSet
	if cold <= 0 {
		return 0
	}
	n := int64(cold)
	if n > capBytes/avgExpertBytes {
		return capBytes
	}
	return n * avgExpertBytes
}
