package modelroute

// ESCALATION: what to do once the cheap rung's attempt comes back (epic #5416, track D).
//
// Place() walks DOWN the ladder to the cheapest rung that can serve a class of work. This
// file is the other half of that loop: the attempt has now run, and something has to say
// whether the result stands, needs checking, or has earned a retry one rung up. Until this
// existed the ladder was one-way. An operator could place work on a laptop model and had no
// sanctioned answer for "it came back wrong" — which in practice means a human quietly re-runs
// it on a vendor, and the automatic placer keeps the credit for a saving it never made.
//
// The whole file is a pure decision over facts the caller already has. It performs no retry,
// touches no roster, and reads no clock: the caller owns the attempt, this owns the rule.
//
// Four rules, and each one exists because the obvious alternative is worse:
//
//  1. SUCCESS NEVER ESCALATES — an unverified success is a reason to CHECK, not to buy a
//     bigger model. The intuitive design escalates a success nobody witnessed, and on a fleet
//     whose witness wiring is incomplete that escalates EVERY local success: the local-first
//     thesis silently inverts into vendor-by-default, which is the precise outcome this epic
//     exists to prevent. Verification here is local, cheap and repeatable (a DOS git witness);
//     escalation is remote and metered. So an untrusted success returns ActionVerifyFirst, and
//     no bound can turn it into a spend.
//
//  2. ONLY A CAPABILITY FAILURE EARNS A RUNG. A transport failure earns a retry on the SAME
//     rung — a larger model does not fix an unreachable one, and escalating on it converts
//     every network blip into vendor spend. A broken work item earns nothing anywhere: a
//     frontier model cannot make a repository compile, and a ladder that tries will burn its
//     entire budget discovering that. An unclassified failure earns nothing either, because
//     the default for an AUTOMATIC SPEND must be "don't" — the opposite of the conservatism
//     Place() uses when picking a floor, and for the opposite reason.
//
//  3. A REFUSAL IS NEVER RETRIED UPWARD. If a guard, a residency floor or a provider refused
//     the call, re-issuing it on a different rung is a guard bypass by retry: the same request
//     aimed at a rung the refusing guard may not cover. This is checked before every bound and
//     there is deliberately no knob that enables it. Escalation must never be a laundering
//     path for a refusal.
//
//  4. ESCALATION IS BOUNDED BY AN EXPLICITLY DECLARED CEILING. The zero-value EscalationBounds
//     grants nothing: an escalator with no declared ceiling and no attempt budget holds an
//     unbounded spend authority nobody granted. A ceiling of ZoneDevice is how work that may
//     not leave the box says so — the escalator then has nowhere to go, by construction rather
//     than by a residency check bolted on afterwards.

// FailureKind names WHY an attempt failed. It is supplied by the caller and never inferred
// here: only the caller can tell a model that tried and could not from a socket that never
// opened, and guessing between them is exactly how a network blip becomes a vendor bill.
type FailureKind string

const (
	// FailNone is the zero value: the attempt did not fail. A FAILED attempt carrying it is
	// read as FailUnclassified rather than as "no failure" — an unnamed cause must not read
	// as an absent one.
	FailNone FailureKind = ""
	// FailUnderpowered: the model tried the work and could not do it. The ONLY kind that
	// earns a rung, because it is the only one a more capable model would change.
	FailUnderpowered FailureKind = "underpowered"
	// FailTransport: the rung was unreachable, timed out, or errored before the model tried.
	// Says nothing about capability, so it earns a retry where it stood.
	FailTransport FailureKind = "transport"
	// FailRefused: a guard, a residency floor or the provider refused the call. Never
	// escalates — see rule 3.
	FailRefused FailureKind = "refused-upstream"
	// FailWorkItem: the task itself cannot succeed on any rung (it will not build, the spec
	// contradicts itself, the target does not exist). No amount of capability fixes it.
	FailWorkItem FailureKind = "work-item-broken"
	// FailUnclassified: it failed and nothing named why. Distinct from every other kind
	// because the fix is instrumenting the caller, not choosing a rung.
	FailUnclassified FailureKind = "unclassified"
)

// AttemptResult is what came back from running a Placement.
//
// Verify is load-bearing on the SUCCESS path and ignored on the failure path: a claim of
// failure is self-limiting (nobody forges a failure to buy themselves a cheaper rung), while
// a claim of success is exactly what a cheap rung has an incentive to overstate.
type AttemptResult struct {
	Succeeded bool         `json:"succeeded"`
	Verify    Verification `json:"verify"`         // provenance of a success claim
	Fail      FailureKind  `json:"fail,omitempty"` // why it failed, when it failed
}

// EscalationBounds is the operator's explicit authority to spend one rung up. Its ZERO VALUE
// grants nothing, which is the point: automatic escalation is an automatic spend, and an
// authority nobody wrote down is one nobody granted.
type EscalationBounds struct {
	// Ceiling is the highest rung escalation may reach. ZoneDevice means the work may not
	// leave the box at all; ZoneVendor means the full ladder is authorised. An invalid or
	// empty ceiling authorises nothing.
	Ceiling PlacementZone `json:"ceiling"`
	// MaxAttempts is how many escalations ONE work item may make. Zero or negative
	// authorises nothing; it is not read as "unlimited".
	MaxAttempts int `json:"max_attempts"`
}

// Granted reports whether these bounds authorise any escalation at all.
func (b EscalationBounds) Granted() bool { return b.Ceiling.Valid() && b.MaxAttempts > 0 }

// EscalationAction is the closed vocabulary of what a caller should do next. A surface renders
// these verbatim; nothing here returns free text.
type EscalationAction string

const (
	// ActionAccept: the result stands as it is.
	ActionAccept EscalationAction = "accept"
	// ActionVerifyFirst: it claims success with no independent check. Witness it, then
	// accept — do NOT re-run it more expensively.
	ActionVerifyFirst EscalationAction = "verify-then-accept"
	// ActionRetrySameRung: retry where it stood; the failure said nothing about capability.
	ActionRetrySameRung EscalationAction = "retry-same-rung"
	// ActionEscalate: retry on the rung named by EscalationVerdict.To.
	ActionEscalate EscalationAction = "retry-one-rung-up"
	// ActionStop: no further attempt is authorised or would help. The reason says which.
	ActionStop EscalationAction = "stop"
)

// Closed reason vocabulary for the escalation decision, parallel to placement.go's. Every
// answer names its own cause: "stop" alone would leave an operator unable to tell work that
// is beyond the ladder from work whose authority nobody declared.
const (
	ReasonAttemptStands       = "attempt-stands"                       // witnessed/judged success
	ReasonSuccessUnverified   = "success-is-self-reported"             // check it, do not buy bigger
	ReasonEarnedByUnderpower  = "underpowered-for-this-work"           // the one thing that earns a rung
	ReasonTransportRetry      = "transport-failure-not-a-capability"   // same rung, again
	ReasonWorkItemBroken      = "work-item-cannot-succeed-on-any-rung" // capability will not fix it
	ReasonRefusalNotRetried   = "refusal-may-not-be-retried-upward"    // rule 3; no knob enables it
	ReasonFailureUnclassified = "failure-unclassified"                 // instrument the caller
	ReasonNoCeiling           = "no-declared-escalation-ceiling"       // nobody granted the authority
	ReasonAtCeiling           = "at-declared-escalation-ceiling"       // authority ends here
	ReasonBudgetSpent         = "escalation-budget-spent"              // this item has had its rungs
	ReasonAtTopRung           = "already-at-top-rung"                  // structural: nowhere above
	ReasonUnplacedAttempt     = "attempt-carries-no-placed-rung"       // nothing ran, or nothing recorded where
)

// EscalationVerdict is the decision plus the rung it moves between, so a log line reads
// "device -> fleet: underpowered-for-this-work" without the caller composing it.
type EscalationVerdict struct {
	Action EscalationAction `json:"action"`
	From   PlacementZone    `json:"from,omitempty"`
	To     PlacementZone    `json:"to,omitempty"`
	Reason string           `json:"reason"`
}

// Escalates reports whether this verdict authorises spending a rung.
func (v EscalationVerdict) Escalates() bool { return v.Action == ActionEscalate }

// AfterAttempt decides what happens next, given the placement that ran, what came back, the
// authority the operator declared, and how many escalations this work item has already made.
//
// It is total: every input produces exactly one action with a named reason, and the default
// on every unrecognised input is to stop rather than to spend.
//
// priorEscalations counts escalations ALREADY made for this work item (0 on the first
// attempt). The caller carries it because only the caller knows what a "work item" is.
func AfterAttempt(p Placement, res AttemptResult, b EscalationBounds, priorEscalations int) EscalationVerdict {
	from := p.Zone
	if !from.Valid() {
		// Nothing ran, or nothing recorded where it ran. Escalating from an unknown rung
		// would pick a "next" rung out of the air, and Rank() deliberately ranks an unknown
		// zone above vendor — which would read as "already at the top" and hide the gap.
		return EscalationVerdict{Action: ActionStop, From: from, Reason: ReasonUnplacedAttempt}
	}

	if res.Succeeded {
		// Rule 1. Success never escalates, whatever the bounds say.
		if res.Verify.Trusted() {
			return EscalationVerdict{Action: ActionAccept, From: from, Reason: ReasonAttemptStands}
		}
		return EscalationVerdict{Action: ActionVerifyFirst, From: from, Reason: ReasonSuccessUnverified}
	}

	switch fail := res.Fail; fail {
	case FailRefused:
		// Rule 3, checked before any bound: a refusal is not a capability finding, and
		// re-aiming it at another rung is a bypass, not a retry.
		return EscalationVerdict{Action: ActionStop, From: from, Reason: ReasonRefusalNotRetried}
	case FailTransport:
		return EscalationVerdict{Action: ActionRetrySameRung, From: from, Reason: ReasonTransportRetry}
	case FailWorkItem:
		return EscalationVerdict{Action: ActionStop, From: from, Reason: ReasonWorkItemBroken}
	case FailUnderpowered:
		// Falls through to the bounded ladder walk below.
	default:
		// FailNone on a failed attempt, and any token this vocabulary does not know.
		// Both mean nobody named the cause, and neither may buy a rung.
		return EscalationVerdict{Action: ActionStop, From: from, Reason: ReasonFailureUnclassified}
	}

	// Rule 2 is satisfied: the model tried and was underpowered. Structural limits first, so
	// an operator is told when granting more authority would not have helped.
	next, ok := nextZoneAbove(from)
	if !ok {
		return EscalationVerdict{Action: ActionStop, From: from, Reason: ReasonAtTopRung}
	}
	// Rule 4.
	if !b.Ceiling.Valid() {
		return EscalationVerdict{Action: ActionStop, From: from, Reason: ReasonNoCeiling}
	}
	if next.Rank() > b.Ceiling.Rank() {
		return EscalationVerdict{Action: ActionStop, From: from, Reason: ReasonAtCeiling}
	}
	if b.MaxAttempts <= 0 || priorEscalations >= b.MaxAttempts {
		return EscalationVerdict{Action: ActionStop, From: from, Reason: ReasonBudgetSpent}
	}
	return EscalationVerdict{Action: ActionEscalate, From: from, To: next, Reason: ReasonEarnedByUnderpower}
}

// nextZoneAbove returns the cheapest rung strictly above z, in canonical ladder order. It
// walks Zones() rather than doing arithmetic on Rank() so that adding a rung to the ladder
// cannot leave this stepping over it.
func nextZoneAbove(z PlacementZone) (PlacementZone, bool) {
	for _, cand := range Zones() {
		if cand.Rank() > z.Rank() {
			return cand, true
		}
	}
	return "", false
}
