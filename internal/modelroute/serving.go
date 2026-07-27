package modelroute

import "fmt"

// SERVING LIVENESS ON THE PLACEMENT LADDER (epic #5416, track H).
//
// placement.go walks device -> fleet -> vendor and takes the cheapest rung that
// can serve the work. It decides on two facts about a candidate: the tier its
// capability was graded at, and whether that grade was measured. Both are
// STATIC — they describe what a model can do, not whether anything is currently
// answering on the other end.
//
// That is the gap this file closes. The fleet rung is a machine somebody has to
// keep running: a GPU host reboots, a vLLM process OOMs, a rack loses power on a
// Sunday. Without a liveness signal the placer keeps handing that host work,
// because from the ladder's point of view nothing has changed — the model is
// still bound, still graded, still the cheapest thing that clears the floor. The
// operator sees a wall of dispatch failures and no explanation, and the ladder
// artifact, which exists to answer "why did this land here?", says the placement
// was fine.
//
// A ServingReport is an injected, credential-free snapshot of what is answering.
// It is INJECTED and carries its own AsOfUnix rather than reading a clock,
// because placement must stay pure: same roster, same candidates, same report =>
// same placement, on any machine, at any time. Probing is somebody else's job,
// above this tier.
//
// THREE RULES DECIDE WHAT A SNAPSHOT IS ALLOWED TO DO.
//
//  1. SILENCE IS ONLY MEANINGFUL WHERE THE SNAPSHOT CLAIMS COVERAGE. A report
//     declares the rungs it speaks for. Inside that coverage, a candidate with no
//     observation is UNKNOWN and is passed over: the producer said it was watching
//     this rung and then said nothing about this model, and treating that as
//     healthy is how a crashed prober turns into a fail-open. Outside it, an
//     absent observation gates nothing at all. That is what makes the zero report
//     a structural no-op — Place is literally PlaceWithServing with an empty
//     report, so a deployment with no prober behaves today exactly as it did
//     before this file existed, not merely by default.
//
//  2. AN OBSERVATION THAT EXISTS IS ALWAYS HONORED, COVERAGE OR NOT. Coverage
//     answers "is silence meaningful?", and nothing else. A report that names
//     corp-mid as DOWN but forgets to list the fleet rung in Covers must still
//     keep work off corp-mid; the alternative is the one failure this file exists
//     to prevent, arrived at through a config typo.
//
//  3. FRESHNESS IS FAIL-CLOSED, AND SO IS EVERY WAY OF NOT HAVING IT. Under a
//     declared MaxAgeSeconds, only an observation that can be SHOWN fresh keeps a
//     candidate eligible. An observation older than the bound, one with no
//     timestamp, one in a report with no as-of stamp to measure against, and one
//     stamped AFTER the report that contains it all read the same way: the
//     freshness claim cannot be checked, so it is not granted. The last case is
//     the one worth stating out loud — a producer whose clock runs ahead would
//     otherwise pin a rung open forever, and a broken producer must never be able
//     to do that.
//
// The cost of rule 3 is real and worth naming: if the prober dies, every covered
// rung goes stale and work escalates to a vendor that costs money. That is the
// right side to fail on, and it is the side the ladder can EXPLAIN. The other
// choice — let a stale "up" keep the rung — spends the operator's afternoon on
// dispatch failures against a host that has been dead since morning, with the
// artifact insisting everything is fine. A bill that goes up with a
// zone-serving-stale token in the ladder is a diagnosis. Requests that fail
// against a rung the artifact calls healthy are not.
//
// DEGRADED IS RECORDED, NOT ACTED ON. A host under load is still serving, and
// shedding its work to a vendor at the first sign of queueing would invert the
// entire point of self-hosting: the busy hour is exactly when the bulk of the
// tokens are, and it is exactly when a load-shedding placer would send them off
// the fleet. Whether saturation should cost money is an operator's policy call,
// not this file's, so the state is carried into the ladder verdict and the
// placement still lands on the rung.

// ServingReportSchema names the wire shape of a liveness snapshot. A producer
// that grows a v2 must say so; see ServingReport.Validate for why an unversioned
// report carrying observations is refused rather than guessed at.
const ServingReportSchema = "fak.modelroute.serving.v1"

// ServingState is the closed set of things a probe can conclude about one bound
// model. It is deliberately NOT a bool: "nobody could tell" is a distinct answer
// from "it answered" and from "it refused", and collapsing the three is how a
// gate ends up trusting a probe that never ran.
type ServingState string

const (
	// ServingUp means the probe reached this model and it answered.
	ServingUp ServingState = "up"
	// ServingDegraded means it is answering but under strain — queued, throttled,
	// slow. It is still serving, so it still takes work; see the file header.
	ServingDegraded ServingState = "degraded"
	// ServingDown means the probe reached a verdict that it cannot serve.
	ServingDown ServingState = "down"
	// ServingUnknown means the probe ran and could not tell. It is an explicit
	// admission of ignorance and is treated exactly like an absent observation
	// inside declared coverage — passed over, never assumed well.
	ServingUnknown ServingState = "unknown"
)

// Valid reports whether s is one of the four defined states. The set is CLOSED —
// a new state is an added constant plus validation, never manifest free text —
// mirroring PlacementZone and ProviderKind.
func (s ServingState) Valid() bool {
	switch s {
	case ServingUp, ServingDegraded, ServingDown, ServingUnknown:
		return true
	}
	return false
}

// ServingObservation is one probe result for one bound model.
//
// ObservedUnix is not decoration. A liveness state without the time it was taken
// is an assertion, not an observation, and the whole reason this type exists is
// to stop the placer from acting on assertions.
type ServingObservation struct {
	State        ServingState `json:"state"`
	ObservedUnix int64        `json:"observed_unix,omitempty"`
}

// ServingReport is an injected, credential-free snapshot of what is currently
// answering, keyed by the ROUTED MODEL ID the roster binds — not by zone.
//
// Per-model is the load-bearing choice. A fleet rung is usually several hosts,
// and a zone-keyed signal would take the whole rung down because one GPU box
// rebooted, sending every token to a vendor to route around a machine whose
// neighbor was idle. Keyed per model, one dead host costs exactly the candidates
// bound to it: the ladder tries the next candidate ON THE SAME RUNG first and
// only leaves the rung when nothing there can serve. Failing over within the
// rung before failing over up it is the difference between a resilient fleet and
// an expensive one.
//
// There is deliberately no credential, token, or base-URL field. A liveness
// snapshot is published, logged, and pasted into issues; it must be safe to hand
// to anyone who can already see the roster's model names.
type ServingReport struct {
	Schema string `json:"schema,omitempty"`

	// AsOfUnix is when this snapshot was assembled — the clock every freshness
	// check is measured against, so that placement itself never reads one.
	AsOfUnix int64 `json:"as_of_unix,omitempty"`

	// MaxAgeSeconds is the freshness bound the operator declares. Zero means no
	// bound: observations are honored at any age, which is the honest reading of
	// declining to state one. See rule 3 in the file header for what a declared
	// bound refuses.
	MaxAgeSeconds int64 `json:"max_age_seconds,omitempty"`

	// Covers names the rungs this snapshot claims to speak for — the rungs where
	// SILENCE about a candidate is meaningful. It is not a filter on the
	// observations below; see rule 2.
	Covers []PlacementZone `json:"covers,omitempty"`

	// Models maps a routed model id to its observation.
	Models map[string]ServingObservation `json:"models,omitempty"`
}

// Validate reports the first way a report is unfit to place work against.
//
// An invalid snapshot is an ERROR, never an ignored one. That follows the rule
// Place already holds itself to for an unresolvable candidate: a typo in a
// placement config must surface as a misconfiguration, not as traffic quietly
// continuing to a vendor — or, here, quietly continuing to a host the operator
// believed they had gated. Every failure mode below is one where ignoring the
// report silently disables the gate.
//
// The zero report is valid, because Place delegates with one.
func (s ServingReport) Validate() error {
	empty := len(s.Covers) == 0 && len(s.Models) == 0
	if s.Schema != "" && s.Schema != ServingReportSchema {
		return fmt.Errorf("modelroute: serving report schema %q is not %q", s.Schema, ServingReportSchema)
	}
	// An unversioned report carrying observations is refused rather than guessed
	// at: the day a v2 changes what a state MEANS, reading it as v1 gates the
	// wrong rungs and nothing says so. The zero report is exempt because it
	// gates nothing at all, so there is no meaning to get wrong.
	if s.Schema == "" && !empty {
		return fmt.Errorf("modelroute: serving report carries observations but declares no schema (want %q)", ServingReportSchema)
	}
	if s.MaxAgeSeconds < 0 {
		return fmt.Errorf("modelroute: serving report max_age_seconds %d is negative", s.MaxAgeSeconds)
	}
	for _, z := range s.Covers {
		if !z.Valid() {
			return fmt.Errorf("modelroute: serving report covers unknown zone %q", z)
		}
	}
	for model, obs := range s.Models {
		if model == "" {
			return fmt.Errorf("modelroute: serving report has an observation under an empty model id")
		}
		if !obs.State.Valid() {
			return fmt.Errorf("modelroute: serving report model %q has unknown state %q", model, obs.State)
		}
	}
	return nil
}

// covers reports whether the snapshot claims to speak for rung z — i.e. whether
// SILENCE about a candidate in that rung means anything.
func (s ServingReport) covers(z PlacementZone) bool {
	for _, c := range s.Covers {
		if c == z {
			return true
		}
	}
	return false
}

// fresh reports whether obs can be SHOWN fresh under the report's declared
// bound. With no bound declared, everything passes. With one declared, each of
// the four ways of not having a checkable age fails: no observation stamp, no
// snapshot stamp to measure against, an age past the bound, and a negative age
// (an observation stamped after the report containing it — a producer whose
// clock runs ahead, which must not be able to pin a rung open forever).
//
// The two missing-stamp guards look redundant against the arithmetic below and
// are not. Either one alone can be removed without changing an answer, because a
// zero stamp against a real one yields an age of about fifty-five years, which
// fails any bound. Remove BOTH and a report where NEITHER side is stamped has an
// age of exactly zero and reads as perfectly fresh — a snapshot that says nothing
// about when it was taken becomes the freshest one in the system. The guards are
// there to say that an unset stamp is a missing measurement rather than an
// instant in 1970, so the right answer does not depend on how far away that
// instant happens to be.
func (s ServingReport) fresh(obs ServingObservation) bool {
	if s.MaxAgeSeconds <= 0 {
		return true
	}
	if obs.ObservedUnix <= 0 || s.AsOfUnix <= 0 {
		return false
	}
	age := s.AsOfUnix - obs.ObservedUnix
	return age >= 0 && age <= s.MaxAgeSeconds
}

// verdict decides what the snapshot says about one candidate.
//
// skip is whether the candidate is passed over; reason is the closed token to
// record, which is non-empty for a DEGRADED candidate that is placed anyway —
// the caller must not infer "recorded" from "skipped".
func (s ServingReport) verdict(zone PlacementZone, model string) (skip bool, reason string) {
	obs, observed := s.Models[model]
	if !observed {
		// Rule 1: silence only means something where the snapshot claims coverage.
		if s.covers(zone) {
			return true, ReasonZoneServingUnknown
		}
		return false, ""
	}
	// Rule 2: from here down, coverage is not consulted again — an observation
	// that exists is honored whether or not its rung was declared.
	if !s.fresh(obs) {
		return true, ReasonZoneServingStale
	}
	switch obs.State {
	case ServingUp:
		return false, ""
	case ServingDegraded:
		return false, ReasonZoneServingDegraded
	case ServingDown:
		return true, ReasonZoneServingDown
	}
	// ServingUnknown, and any state Validate would have refused: an unreadable
	// verdict is not a passing one.
	return true, ReasonZoneServingUnknown
}
