// Package computeadmit is the ONE shared admission kernel over the compute
// partitioners (#3269, parent epic #3259) — the compute-plane twin of
// regionadmit.Decide / laneadmit.Decide.
//
// The control plane already holds two pure admission twins (laneadmit,
// regionadmit) that apply one contract strongest-rule-first and refuse with a
// closed vocabulary. The compute plane's placers — NativePDCluster.routeDecode,
// ExpertParallelPlan, TPPlan, the kvmmu/cachemeta tier placer — each decide
// privately and cannot emit that vocabulary or RepartitionAdvice, so a
// compute-region overflow looks nothing like a lane collision even though it is
// one. This package gives every placer the shared *answer* additively: state in
// (Request, live Leases, Taxonomy), verdict out (Decision) — no clock, no I/O,
// no scheduling. Contention reuses the C1 collision machinery
// (dispatchorder.ComputeClaimsContend) so a placer prices a region exactly as
// the dispatch fan-out price does; an out-of-taxonomy claim (ExpertParallelPlan's
// "ranks outside [1,N]" fail-closed) refuses POLICY_BLOCK with the
// out_of_taxonomy rung instead of a private error shape.
//
// Decision algebra ONLY — not a preemptive scheduler; every existing placer's
// mechanism stays byte-identical (borrow-the-term / disclaim-the-scope).
package computeadmit

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// The closed refusal vocabulary a compute-admission Decision cites — the same
// dos.toml [reasons] tokens the dispatch loop, the dos arbiter, and the two
// control-plane admission twins already speak. Never free text.
const (
	// ReasonCollisionRisk: the claimed region contends with a live compute lease
	// (dispatchorder's token re-exported, exactly as regionadmit re-exports it).
	ReasonCollisionRisk = dispatchorder.ReasonCollisionRisk
	// ReasonPolicyBlock: the claim addresses a region outside the class's
	// declared taxonomy — structurally invalid, refused before any contention
	// pricing. Mirrors abi.ReasonPolicyBlock's stable name.
	ReasonPolicyBlock = "POLICY_BLOCK"
)

// The rung names a Decision carries so a refusal says WHICH admission rule
// fired, not just that one did. Closed set; evidence, not prose.
const (
	// RungOutOfTaxonomy: the claim's range parses outside the class's declared
	// address space (or does not parse at all while a space is declared) — the
	// compute twin of requesting an undeclared lane, and the shared-vocabulary
	// form of ExpertParallelPlan's "ranks outside [1,N]" fail-closed.
	RungOutOfTaxonomy = "out_of_taxonomy"
	// RungRegionCollision: the claimed region contends with a live lease under
	// the same class/mode/range fold the dispatch fan-out price uses.
	RungRegionCollision = "compute_region_collision"
)

// The well-known compute resource classes the placers claim. Free-form class
// labels are accepted everywhere; these constants keep the evidence vocabulary
// stable across placers (they mirror dispatchorder.ComputeClaim's doc).
const (
	ClassDevice    = "device"
	ClassPhase     = "phase-pool"
	ClassExpertSet = "expert-set"
	ClassKVTier    = "kv-tier"
)

// Space is one class's declared inclusive address space [Lo, Hi] — the compute
// twin of a lane's canonical tree in the dos.toml taxonomy.
type Space struct {
	Lo int64
	Hi int64
}

// Taxonomy is the compute-plane lane data the decision honors: each resource
// class's declared valid address space. A class absent from Space is
// unconstrained (greenfield) — its claims skip the taxonomy rung and are priced
// on contention only, which keeps every existing placer admissible with a zero
// Taxonomy.
type Taxonomy struct {
	Space map[string]Space
}

// Request is one placer asking to admit work onto a compute region: Actor names
// the would-be holder (an ensemble member id, a decode session, host:pid), Claim
// is the region being claimed (class + integer range + lock mode), and SelfID
// names the caller's own lease id so a re-admission (renew/re-route of the same
// holder) is not refused against itself.
type Request struct {
	Actor  string
	Claim  dispatchorder.ComputeClaim
	SelfID string
}

// Lease is the projection of one live compute lease the decision folds — any
// source (a Submit-seam gate row, a scheduler table, a dos arbiter row)
// projects to this shape.
type Lease struct {
	ID     string
	Holder string
	Claim  dispatchorder.ComputeClaim
}

// Decision is the admission verdict. A refusal carries the closed reason
// (COLLISION_RISK / POLICY_BLOCK), the rung that fired, a human-readable detail
// naming the conflict, the conflicting lease as evidence, and — for a collision
// — the same RepartitionAdvice row the dispatch price emits, so "narrow the
// claimed range and re-ask" is machine-actionable on the compute plane too.
type Decision struct {
	Admit    bool
	Reason   string
	Rung     string
	Detail   string
	Conflict *Lease
	Advice   *dispatchorder.RepartitionAdvice
}

// Decide answers "may this actor run on this compute region right now?" against
// the live compute-lease set and the class taxonomy. It is pure: state in,
// verdict out. The rules, strongest first, mirror the control-plane twins:
//
//  1. out_of_taxonomy — a claim whose range falls outside its class's declared
//     address space (or fails to parse while a space is declared) is refused
//     POLICY_BLOCK before any contention pricing (fail-closed, the
//     ExpertParallelPlan discipline in the shared vocabulary);
//  2. compute_region_collision — a claim contending with a live lease under
//     dispatchorder.ComputeClaimsContend (same class, not shared/shared,
//     overlapping ranges; an empty range collides conservatively) is refused
//     COLLISION_RISK with RepartitionAdvice.
//
// A lease whose ID equals req.SelfID is the caller's own and never conflicts.
func Decide(req Request, live []Lease, tax Taxonomy) Decision {
	class := strings.ToLower(strings.TrimSpace(req.Claim.Class))
	if sp, ok := tax.Space[class]; ok {
		rng := strings.TrimSpace(req.Claim.Range)
		if rng != "" {
			within, known := dispatchorder.RangeWithin(rng, sp.Lo, sp.Hi)
			if !known {
				return Decision{
					Admit:  false,
					Reason: ReasonPolicyBlock,
					Rung:   RungOutOfTaxonomy,
					Detail: fmt.Sprintf(
						"claim %s does not parse; class %q declares address space [%d,%d] and an unprovable membership fails closed",
						dispatchorder.ComputeRegionLabel(req.Claim), class, sp.Lo, sp.Hi),
				}
			}
			if !within {
				return Decision{
					Admit:  false,
					Reason: ReasonPolicyBlock,
					Rung:   RungOutOfTaxonomy,
					Detail: fmt.Sprintf(
						"claim %s is outside class %q declared address space [%d,%d]",
						dispatchorder.ComputeRegionLabel(req.Claim), class, sp.Lo, sp.Hi),
				}
			}
		}
	}
	for i := range live {
		l := live[i]
		if req.SelfID != "" && l.ID == req.SelfID {
			continue
		}
		if !dispatchorder.ComputeClaimsContend(req.Claim, l.Claim) {
			continue
		}
		adv := repartition(req, l)
		return Decision{
			Admit:  false,
			Reason: ReasonCollisionRisk,
			Rung:   RungRegionCollision,
			Detail: fmt.Sprintf("claim %s contends with live compute lease %s on %s",
				dispatchorder.ComputeRegionLabel(req.Claim), leaseLabel(l),
				dispatchorder.ComputeRegionLabel(l.Claim)),
			Conflict: &l,
			Advice:   &adv,
		}
	}
	return Decision{Admit: true}
}

// repartition builds the compute-plane advice row for a refused claim, using
// the SAME action vocabulary the dispatch price's computeRepartition emits:
// narrow a declared range to a disjoint sub-region, or declare a range at all
// when the claim left it unknown.
func repartition(req Request, l Lease) dispatchorder.RepartitionAdvice {
	var cur []string
	action := "narrow_compute_range"
	detail := "narrow the claimed compute range to a disjoint sub-region of its class, then re-ask"
	if strings.TrimSpace(req.Claim.Range) != "" {
		cur = []string{dispatchorder.ComputeRegionLabel(req.Claim)}
	} else {
		action = "declare_compute_range"
		detail = "declare this claim's compute region (class:range) before admission; an unknown range collides conservatively"
	}
	return dispatchorder.RepartitionAdvice{
		Candidate:     req.Actor,
		CollidesWith:  []string{l.ID},
		CurrentRegion: cur,
		OverlapRegion: []string{dispatchorder.ComputeRegionLabel(l.Claim)},
		Action:        action,
		Reason:        ReasonCollisionRisk,
		Detail:        detail,
	}
}

func leaseLabel(l Lease) string {
	if l.Holder == "" || l.Holder == l.ID {
		return l.ID
	}
	return fmt.Sprintf("%s (holder %s)", l.ID, l.Holder)
}

// ExpertTaxonomy declares the expert-parallel planner's address space: rank
// counts in [1, numExperts], the SAME closed interval ExpertParallelPlan and
// NewTPPlan fail closed on ("ranks must be in [1,numExperts]; a rank with no
// experts is a hole"). numExperts <= 0 declares an empty space every claim
// refuses against — matching ExpertParallelPlan(0, r) failing.
func ExpertTaxonomy(numExperts int) Taxonomy {
	return Taxonomy{Space: map[string]Space{ClassExpertSet: {Lo: 1, Hi: int64(numExperts)}}}
}

// ExpertRanksRequest expresses ExpertParallelPlan's rank-band claim as an
// admission request: `ranks` contiguous bands over the expert axis, i.e. the
// claim "expert-set:1-ranks" priced against ExpertTaxonomy(numExperts). An
// out-of-range ranks (0, negative, > numExperts) becomes a regionadmit-style
// out-of-taxonomy POLICY_BLOCK instead of a private error — the planner's
// mechanism is untouched; this is the shared answer it can additionally ask.
func ExpertRanksRequest(actor string, ranks int) Request {
	return Request{
		Actor: actor,
		Claim: dispatchorder.ComputeClaim{Class: ClassExpertSet, Range: fmt.Sprintf("1-%d", ranks)},
	}
}

// DecodeTaxonomy declares the P/D decode phase-pool address space: worker
// indices in [0, workers). workers <= 0 declares an empty space every claim
// refuses against.
func DecodeTaxonomy(workers int) Taxonomy {
	return Taxonomy{Space: map[string]Space{ClassPhase: {Lo: 0, Hi: int64(workers) - 1}}}
}

// DecodeWorkerRequest expresses a P/D decode placement (the worker
// NativePDCluster.routeDecode picks) as an admission request over the decode
// phase-pool, so a decode-worker over-subscription refuses as a COLLISION_RISK
// with RepartitionAdvice instead of being silently co-scheduled.
func DecodeWorkerRequest(actor string, worker int) Request {
	return Request{
		Actor: actor,
		Claim: dispatchorder.ComputeClaim{Class: ClassPhase, Range: strconv.Itoa(worker)},
	}
}
