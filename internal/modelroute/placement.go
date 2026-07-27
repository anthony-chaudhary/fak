package modelroute

import (
	"fmt"
	"sort"
)

// PLACEMENT: walking the zone ladder (epic #5416, track A).
//
// PlacementZone names the three rungs a self-hosting organization runs models on
// (device -> fleet -> vendor, cheapest first). This file is the part that WALKS that
// ladder: given a class of work and the models an operator has bound, choose the
// cheapest rung that can actually do the job.
//
// It is deliberately separate from Route(). Route() answers "WHICH MODEL serves this
// aspect" (epic #595); this answers "WHICH RUNG does that model run on, and who
// pays". Composing two small decisions beats entangling them into one: an operator
// can change their hardware story without touching routing policy, and vice versa.
//
// Every tier comparison here routes through the existing WorkTier vocabulary —
// TierPolicy.Admit and WorkTier.MeetsRequirement — and never through a raw `<`. That
// matters more than usual on this path because the tier numbers are INVERTED
// (T0 < T1 < T2 numerically, but T0 is the MOST demanding). A raw comparison reads
// backwards and would silently place the most demanding work on the cheapest rung —
// the one bug that would make an automatic placer worse than a hand-written binding.

// Closed reason vocabulary for placement. A status surface renders these verbatim,
// so an operator can see WHY their expensive vendor bill did not move to the fleet
// without anyone writing free text.
const (
	ReasonPlacedInZone      = "placed-in-zone"              // this rung took the work
	ReasonZoneNoCandidate   = "zone-no-candidate"           // nothing bound in this rung
	ReasonZoneUnderTier     = "zone-under-tier"             // bound, but below the work's floor
	ReasonZoneUnmeasured    = "zone-capability-unmeasured"  // bound, but capability is asserted, not measured
	ReasonEscalatedPast     = "escalated-past-cheaper-zone" // a cheaper rung existed and lost
	ReasonNoZoneCanServe    = "no-zone-can-serve"           // the ladder ran out
	ReasonTopRungUnmeasured = "top-rung-unmeasured-allowed" // the vendor fallback, taken on an asserted capability
	ReasonZoneNotReached    = "zone-not-reached"            // a cheaper rung took the work first
)

// Candidate is one placeable option: a routed model id that the roster binds, plus
// the work tier the operator has evidence it can serve.
//
// Measured is load-bearing, not decoration. It records whether Capability is a
// MEASUREMENT or an ASSERTION. Today every production capability grade is an
// assertion: internal/ablate's StubTierScorer measures nothing, so real runs grade
// UNMEASURED (epic #5416 track F replaces it). Without this flag the placer would
// send every task to the cheapest rung on the strength of a number nobody computed,
// and would look like it was working.
type Candidate struct {
	Model      string   `json:"model"`      // a routed id the Roster can Resolve
	Capability WorkTier `json:"capability"` // the most demanding tier this model can serve
	Measured   bool     `json:"measured"`   // is Capability backed by a measurement?
}

// ZoneVerdict is the record of what happened at one rung of the ladder — including
// the rungs that were passed over. The rejections are the interesting half: "why is
// this still going to a vendor?" is the question an operator actually asks.
type ZoneVerdict struct {
	Zone    PlacementZone `json:"zone"`
	Model   string        `json:"model,omitempty"`
	Reasons []string      `json:"reasons"`
}

// Placement is the chosen rung plus the full ladder walk that produced it.
type Placement struct {
	Model     string        `json:"model"`
	Target    Target        `json:"target"`
	Zone      PlacementZone `json:"zone"`
	Choice    TierChoice    `json:"choice"`
	Escalated bool          `json:"escalated"` // a cheaper rung had a candidate and lost
	Ladder    []ZoneVerdict `json:"ladder"`    // every rung, in ladder order

	// Measured reports whether the winning candidate's capability was MEASURED or
	// merely the unmeasured default.
	//
	// It is carried explicitly because Choice cannot be read honestly without it. An
	// unmeasured candidate enters Admit at the zero-value tier — the most demanding one
	// — so the top rung's fallback is admitted with Choice.Capability == TierT0 and,
	// against a routine floor, Choice.OverTier == true. Both are artifacts of the
	// default, not findings: reporting "over-tier waste" about a model nobody graded
	// asserts a capability fak does not have. A surface rendering Choice MUST consult
	// this first. Recovering it by scanning Ladder for ReasonTopRungUnmeasured would
	// work today and break the moment the reason vocabulary grows.
	Measured bool `json:"measured"`
}

// SelfHosted reports whether the placement landed on hardware the engineer or their
// organization operates — the predicate epic #5416's headline token fraction counts.
func (p Placement) SelfHosted() bool { return p.Zone.SelfHosted() }

// Place chooses the cheapest zone that can serve the given class of work.
//
// The walk is: for each zone in ladder order (device, fleet, vendor), consider the
// candidates that resolve into that zone, in the order given; take the first the
// class's TierPolicy ADMITS. Same roster, same candidates, same class => same
// placement; there is no scoring, randomness, or map iteration in the decision.
//
// Two rules keep it from being wishful:
//
//  1. The floor is fixed by the WORK, not the model. TierPolicy.Admit refuses a
//     candidate below the class's required tier no matter how cheap its rung is, so
//     security/release/destructive work cannot fall to a small local model just
//     because one is available. An unknown class stays conservative at the highest
//     floor rather than inferring a cheap tier.
//
//  2. UNMEASURED CAPABILITY MAY NOT DESCEND THE LADDER. A candidate whose capability
//     is asserted rather than measured is skipped on every rung below the most
//     expensive one. It can still serve as the vendor-rung fallback, because
//     declining to place work anywhere is worse than placing it on the rung that was
//     already the status quo. This is what stops the placer from "saving" money on
//     evidence that does not exist — and it degrades honestly: until capability
//     measurement lands, an operator opts a cheap rung in explicitly by marking a
//     candidate Measured, which is a claim they can be held to.
//
// A candidate the roster cannot resolve is a fail-loud error, never a silent skip: a
// typo in a placement config should surface as a misconfiguration, not as traffic
// quietly continuing to go to a vendor.
func (r Roster) Place(class WorkClass, candidates []Candidate) (Placement, error) {
	if len(candidates) == 0 {
		return Placement{}, fmt.Errorf("modelroute: placement for class %q has no candidates", class)
	}
	policy := PolicyFor(class)

	// Resolve every candidate up front so a misconfiguration fails before any
	// placement decision is reported.
	type resolved struct {
		cand   Candidate
		target Target
		zone   PlacementZone
	}
	byZone := map[PlacementZone][]resolved{}
	for _, c := range candidates {
		if c.Model == "" {
			return Placement{}, fmt.Errorf("modelroute: placement candidate has an empty model id")
		}
		t, err := r.Resolve(c.Model)
		if err != nil {
			return Placement{}, fmt.Errorf("modelroute: placement candidate %q: %w", c.Model, err)
		}
		z := t.Zone()
		byZone[z] = append(byZone[z], resolved{cand: c, target: t, zone: z})
	}

	ladder := Zones()
	topRung := ladder[len(ladder)-1]

	var (
		verdicts []ZoneVerdict
		// cheaperRungLost records that some STRICTLY cheaper rung had a candidate
		// bound and failed to take the work. It is only set after a zone finishes
		// without placing, so a rung that rejects its first candidate and admits its
		// second is not mislabeled as an escalation.
		cheaperRungLost bool
		placed          *Placement
	)

	for _, zone := range ladder {
		here := byZone[zone]
		if len(here) == 0 {
			verdicts = append(verdicts, ZoneVerdict{Zone: zone, Reasons: []string{ReasonZoneNoCandidate}})
			continue
		}
		var zoneReasons []string
		var chosen *resolved
		var choice TierChoice
		for i := range here {
			cand := here[i]
			// Rule 2: an asserted capability may not win a cheaper rung.
			if !cand.cand.Measured && zone != topRung {
				zoneReasons = append(zoneReasons, ReasonZoneUnmeasured)
				continue
			}
			c := policy.Admit(cand.cand.Model, cand.cand.Capability)
			if !c.Admitted {
				zoneReasons = append(zoneReasons, ReasonZoneUnderTier)
				continue
			}
			chosen, choice = &here[i], c
			if !cand.cand.Measured {
				zoneReasons = append(zoneReasons, ReasonTopRungUnmeasured)
			}
			break
		}
		if chosen == nil {
			verdicts = append(verdicts, ZoneVerdict{Zone: zone, Reasons: zoneReasons})
			cheaperRungLost = true // this rung was available and did not take the work
			continue
		}
		zoneReasons = append(zoneReasons, ReasonPlacedInZone)
		if cheaperRungLost {
			zoneReasons = append(zoneReasons, ReasonEscalatedPast)
		}
		verdicts = append(verdicts, ZoneVerdict{Zone: zone, Model: chosen.cand.Model, Reasons: zoneReasons})
		placed = &Placement{
			Model:     chosen.cand.Model,
			Target:    chosen.target,
			Zone:      zone,
			Choice:    choice,
			Escalated: cheaperRungLost,
			Measured:  chosen.cand.Measured,
		}
		break
	}

	if placed == nil {
		return Placement{}, fmt.Errorf("modelroute: no zone can serve class %q (%s): %s",
			class, ReasonNoZoneCanServe, describeLadder(verdicts))
	}
	// Record the rungs BELOW the placement (already walked) plus the untouched ones
	// above it, so the artifact always shows the whole ladder. A rung above the
	// placement reads NOT REACHED rather than "no candidate": it may well have had
	// one, and claiming otherwise would misreport the operator's own roster.
	for _, zone := range ladder {
		if zone.Rank() > placed.Zone.Rank() {
			verdicts = append(verdicts, ZoneVerdict{Zone: zone, Reasons: []string{ReasonZoneNotReached}})
		}
	}
	sort.SliceStable(verdicts, func(i, j int) bool { return verdicts[i].Zone.Rank() < verdicts[j].Zone.Rank() })
	placed.Ladder = verdicts
	return *placed, nil
}

// describeLadder renders a ladder walk as a compact, deterministic string for an
// error message — closed reason tokens only, no free text.
func describeLadder(verdicts []ZoneVerdict) string {
	out := ""
	for i, v := range verdicts {
		if i > 0 {
			out += "; "
		}
		out += string(v.Zone) + "="
		if len(v.Reasons) == 0 {
			out += ReasonZoneNoCandidate
			continue
		}
		for j, rsn := range v.Reasons {
			if j > 0 {
				out += ","
			}
			out += rsn
		}
	}
	return out
}
