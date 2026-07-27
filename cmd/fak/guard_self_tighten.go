// guard_self_tighten.go implements the self-amendment admission gate (#5181,
// epic #5170, Track F): a self-authored (agent-proposed) policy overlay may be
// applied WITHOUT operator gating only when it does not loosen the floor. It is
// the enforcement companion to the propose-only channel (#5182) and the
// monotone-tighten envelope design (#5185): the agent's own authority may
// ratchet the floor tighter, never wider. The decision routes through the
// canonical policy.DiffAmendment engine, so "tighten-only" means exactly what
// the amendment-class registry says it means — a widened Allow/AllowPrefix,
// a removed Deny/SelfModifyGlob, a loosened Posture, or any FROZEN-floor
// movement all refuse; only a tighten-only or no-op delta is self-admissible.
//
// AND TODAY THERE IS NO CALLER. This ships as the DECISION half only: nothing
// outside its test calls admitSelfTightenOverlay, so no live overlay is admitted
// or refused by it yet. Said outright because "enforcement companion" above is a
// DESIGN relationship and not a wiring claim — the propose-only channel does not
// route through this function today, and naming a call site that does not exist
// is the failure mode this repo treats as an unwitnessed claim. Promotion
// evidence is a self-authored overlay that actually reaches this gate and is
// journaled with its class; the arming rung is filed rather than half-done here.
//
// ONE BRANCH IS UNREACHABLE BY CONSTRUCTION, AND THAT IS THE POINT.
// AmendmentFrozenViolation cannot be produced by DiffAmendment as it stands:
// DiffAmendment emits changes under exactly eight Field names (Allow,
// AllowPrefix, Deny, SelfModifyGlobs, Posture, ArgPredicates, SecretPatterns,
// EgressBlockHosts), every one of them is registered, and none is AmendFrozen —
// the FROZEN knobs are declared with Field:"" and are compiled-in only. The
// branch arms itself the moment someone extends DiffAmendment with a field they
// forget to register, because route() sends an unknown field to Frozen,
// fail-closed. So it is insurance against a future edit rather than
// currently-exercised logic, and the tests below do not pretend to cover it.
package main

import (
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// admitSelfTightenOverlay classifies the delta from the current effective floor
// to a proposed self-authored one and returns whether it may be admitted on the
// agent's own authority, the amendment class, and a one-line reason. Admit only
// AmendmentNone (no-op) or AmendmentTighten; refuse AmendmentWiden and
// AmendmentFrozenViolation (and, fail-closed, anything unclassified).
func admitSelfTightenOverlay(current, proposed adjudicator.Policy) (admit bool, class, reason string) {
	class = policy.DiffAmendment(current, proposed).Class()
	switch class {
	case policy.AmendmentNone:
		return true, class, "no-op: proposed overlay is byte-equal to the current floor"
	case policy.AmendmentTighten:
		return true, class, "tighten-only: proposed overlay only narrows the floor"
	case policy.AmendmentWiden:
		return false, class, "refused: proposed overlay would WIDEN the floor; a self-authored channel may only tighten"
	case policy.AmendmentFrozenViolation:
		return false, class, "refused: proposed overlay moves a FROZEN floor element"
	default:
		return false, class, "refused: unclassified amendment delta (fail-closed)"
	}
}
