package modelroute

import "fmt"

// SPAWN PLACEMENT: a delegated turn gets its own rung (epic #5416, track E).
//
// A sub-agent sweep, a background turn, a scouted lookup — these are delegated
// work: one turn spawns another. In a 100-500 engineer shop they are the
// high-volume, low-stakes majority of traffic, and they are exactly the work a
// device- or fleet-class model can serve. The epic's thesis ("the bulk of token
// usage gets covered by these self-hosted items") lives or dies here, because
// this is the traffic that moves in VOLUME rather than one expensive request at
// a time.
//
// The status quo is inheritance: a spawned turn runs on whatever the parent was
// running on. placement.go has no notion of a parent at all — Place takes a
// class and candidates, and nothing else — so a child placed by inheriting is a
// child that was never placed. This file is the seam that says a delegated turn
// gets its OWN walk down the ladder.
//
// INHERITANCE IS NOT MERELY EXPENSIVE. IT IS A FLOOR BYPASS.
//
// That is the part worth saying first, because the cost argument is the obvious
// one and the safety argument is the load-bearing one. TierPolicy fixes the
// floor by the WORK: security/release/destructive work requires T1 no matter how
// small the request looks. Place enforces that floor on every candidate it
// considers. A child that inherits its parent's model is never handed to Place,
// so no floor is ever applied to it — and the parent's floor was computed for
// the PARENT's class. A routine turn legitimately placed on a small T2 device
// model that spawns a security-release child hands destructive work to a T2
// model, and every gate in this package was satisfied, because the child never
// went through one. Re-placing the child is what closes that, and it closes the
// bill at the same time. Both directions of the bug have the same fix.
//
// THE PARENT'S ZONE IS AN INPUT, NEVER A FLOOR.
//
// A sub-agent spawned from a vendor turn must still be able to land on the
// engineer's own laptop. If the parent's rung were a floor, delegation would
// ratchet: one frontier turn near the root would pin its entire subtree to the
// vendor rung, which is precisely today's behavior wearing a ladder costume.
//
// That rule is STRUCTURAL here, not a promise. PlaceSpawnWithServing delegates
// the decision to PlaceWithServing and does not pass the parent to it. The
// parent is not consulted because it is not in scope at the point of decision,
// so no future edit can quietly make it a floor without deleting the delegation
// and being seen doing it. Everything this file adds is recorded ALONGSIDE the
// placement, never folded into it.
//
// WHICH SETTLES #5420'S OPEN QUESTION. That issue asks whether "background" is a
// property of the turn, the session, or the tool call, and guesses at a new
// Aspect. It is none of the three, and it is not an Aspect. Aspect is the
// granularity of the ROUTING question — which model serves a request, a tool
// call, a step (see Subject). Placement is a separate axis: which RUNG that
// model runs on and who pays, deliberately not entangled with routing (see the
// placement.go header). Delegation is not a granularity; a spawned turn can be a
// whole request or a single tool call and is no different in size for having
// been delegated. What is actually distinct about it is its RELATIONSHIP TO
// ANOTHER PLACEMENT — it has a parent — and a relationship between two
// placements belongs on the placement axis. So: no new Aspect, no new orthogonal
// flag, and no change to Subject. The cheapest correct answer the issue asked
// for turns out to be a function signature that takes the parent as an argument.

// Closed reason vocabulary for spawn placement. These describe the RELATIONSHIP
// between a child's placement and its parent's; the child's own ladder walk
// keeps reporting in the placement.go vocabulary, unchanged.
const (
	// ReasonSpawnDescended: the child landed on a strictly cheaper rung than its
	// parent. This is the token the epic's headline fraction counts.
	ReasonSpawnDescended = "spawn-descended-from-parent-zone"
	// ReasonSpawnHeld: the child landed on the same rung as its parent. NOT the
	// same as inheriting — it was placed there on its own evidence, and would
	// have landed there with no parent at all.
	ReasonSpawnHeld = "spawn-held-parent-zone"
	// ReasonSpawnRose: the child landed on a MORE expensive rung than its parent.
	// Legitimate and worth recording: a routine turn on the laptop may spawn work
	// that genuinely needs a frontier model, and an operator reading a vendor bill
	// should be able to find it.
	ReasonSpawnRose = "spawn-rose-above-parent-zone"
	// ReasonSpawnRootTurn: no parent — this turn was not spawned by anything, so
	// there is no relationship to report.
	ReasonSpawnRootTurn = "spawn-has-no-parent"

	// ReasonSpawnInheritUnderTier: had this child INHERITED its parent's model,
	// that model would not have cleared the child's own floor. The status quo was
	// unsafe for this spawn, not merely expensive.
	ReasonSpawnInheritUnderTier = "spawn-inherit-would-under-tier"
	// ReasonSpawnInheritAdmitted: the parent's model would have cleared the
	// child's floor. Inheriting would have been survivable here — the cost
	// argument applies, the safety one does not.
	ReasonSpawnInheritAdmitted = "spawn-inherit-would-admit"
	// ReasonSpawnInheritUnmeasured: the parent's capability was never measured, so
	// what inheriting would have done CANNOT be answered. See SpawnPlacement.
	ReasonSpawnInheritUnmeasured = "spawn-inherit-unmeasured"
)

// SpawnPlacement is a delegated turn's own placement, plus the record of how it
// relates to the turn that spawned it.
//
// The relationship fields are an AUDIT surface and gate nothing. That is what
// lets them be honest about ignorance: InheritWouldUnderTier answers a
// counterfactual about the status quo ("what would inheriting have done?"), and
// when the parent's capability is unmeasured the truthful answer is that nobody
// knows. Because no decision rides on it, reporting the unknown as a verdict
// would buy no safety and would put a number nobody computed onto an operator's
// screen. Placement.Measured carries the same warning for the same reason: an
// unmeasured candidate enters Admit at the zero-value tier — the MOST demanding
// one — so an ungraded parent would otherwise appear to clear every floor in the
// system.
type SpawnPlacement struct {
	// Placement is the child's own ladder walk — byte-identical to what
	// Place(class, candidates) returns, because that is literally what produced
	// it.
	Placement Placement `json:"placement"`

	// ParentZone and ParentModel are the spawning turn's placement, recorded as
	// provenance. Empty for a root turn.
	ParentZone  PlacementZone `json:"parent_zone,omitempty"`
	ParentModel string        `json:"parent_model,omitempty"`

	// Descended reports that the child landed strictly cheaper than its parent —
	// the event the epic exists to cause, counted per spawn rather than per token
	// so that a fleet of cheap sub-agents is visible even when the parent turn
	// dominates the bill.
	Descended bool `json:"descended,omitempty"`

	// InheritWouldUnderTier reports that the parent's model would NOT have
	// cleared this child's floor. False also means "not established" — see
	// ReasonSpawnInheritUnmeasured in Reasons, and the type doc for why this
	// field declines to guess.
	InheritWouldUnderTier bool `json:"inherit_would_under_tier,omitempty"`

	// Reasons are closed tokens from this file's vocabulary, in a fixed order:
	// the zone relationship first, then the inheritance counterfactual.
	Reasons []string `json:"reasons"`
}

// SelfHostedDescent reports the event epic #5416 counts: a delegated turn that
// both landed on hardware the organization operates AND did so on a cheaper rung
// than its parent. It is deliberately narrower than SelfHosted — a child that
// held an already-self-hosted rung did not move any tokens off a vendor, and
// counting it would inflate the only number that says whether this works.
func (s SpawnPlacement) SelfHostedDescent() bool {
	return s.Descended && s.Placement.SelfHosted()
}

// PlaceSpawn places a delegated turn: the child gets its own walk down the
// ladder for its OWN work class, and the parent is recorded rather than obeyed.
//
// Pass the zero Placement for a turn with no parent. A parent that is neither
// zero nor a complete placement (a valid zone AND a bound model) is a fail-loud
// error: a half-filled parent means the caller lost track of where the spawning
// turn landed, and inventing a relationship from it would put a provenance claim
// into the audit trail that nobody can stand behind.
//
// The child's class is REQUIRED and an empty one is refused. Place tolerates an
// empty class because PolicyFor already answers conservatively for it (the T0
// floor, ReasonUnknownClass), and that contract is public and stays as it is.
// This path holds itself tighter, because "unclassified" arriving at a spawn is
// exactly the shape a floor bypass would take: if delegation ever became a way
// to reach a placement without stating what the work IS, the classifier is the
// thing that got skipped, and a conservative default would hide that by placing
// the work expensively and looking correct. A caller that cannot say what a
// sub-agent is doing has a bug worth surfacing at the call site.
func (r Roster) PlaceSpawn(parent Placement, class WorkClass, candidates []Candidate) (SpawnPlacement, error) {
	return r.PlaceSpawnWithServing(parent, class, candidates, ServingReport{})
}

// PlaceSpawnWithServing is PlaceSpawn against a liveness snapshot (serving.go),
// mirroring the Place / PlaceWithServing pair so a spawned turn is subject to
// exactly the same failover rules as any other placement — a delegated turn
// must not be the one path that keeps dispatching to a host that stopped
// answering.
func (r Roster) PlaceSpawnWithServing(parent Placement, class WorkClass, candidates []Candidate, serving ServingReport) (SpawnPlacement, error) {
	if class == "" {
		return SpawnPlacement{}, fmt.Errorf("modelroute: spawn placement requires a work class (an unclassified spawn is a missing classification, not a routine one)")
	}
	rooted, err := parentIsRoot(parent)
	if err != nil {
		return SpawnPlacement{}, err
	}

	// The decision. The parent is NOT passed here, and that is the rule about
	// the parent's zone never being a floor — expressed as a scope rather than
	// as a promise.
	placed, err := r.PlaceWithServing(class, candidates, serving)
	if err != nil {
		return SpawnPlacement{}, err
	}

	out := SpawnPlacement{Placement: placed}
	if rooted {
		out.Reasons = []string{ReasonSpawnRootTurn}
		return out, nil
	}
	out.ParentZone, out.ParentModel = parent.Zone, parent.Model

	switch {
	case placed.Zone.Rank() < parent.Zone.Rank():
		out.Descended = true
		out.Reasons = append(out.Reasons, ReasonSpawnDescended)
	case placed.Zone.Rank() == parent.Zone.Rank():
		out.Reasons = append(out.Reasons, ReasonSpawnHeld)
	default:
		out.Reasons = append(out.Reasons, ReasonSpawnRose)
	}

	// The counterfactual: what the status quo would have done with this child.
	if !parent.Measured {
		out.Reasons = append(out.Reasons, ReasonSpawnInheritUnmeasured)
		return out, nil
	}
	if PolicyFor(class).Admit(parent.Model, parent.Choice.Capability).Admitted {
		out.Reasons = append(out.Reasons, ReasonSpawnInheritAdmitted)
	} else {
		out.InheritWouldUnderTier = true
		out.Reasons = append(out.Reasons, ReasonSpawnInheritUnderTier)
	}
	return out, nil
}

// parentIsRoot reports whether parent names no spawning turn, and refuses a
// parent that is neither absent nor complete. "Absent" is the zero Placement:
// no zone and no model. Anything with one of the two is a caller that half-knows
// where its parent landed, which is not a thing this file will guess about.
func parentIsRoot(parent Placement) (bool, error) {
	switch {
	case parent.Zone == "" && parent.Model == "":
		return true, nil
	case parent.Model == "":
		return false, fmt.Errorf("modelroute: spawn parent is in zone %q with no bound model", parent.Zone)
	case !parent.Zone.Valid():
		return false, fmt.Errorf("modelroute: spawn parent model %q is in unknown zone %q", parent.Model, parent.Zone)
	}
	return false, nil
}
