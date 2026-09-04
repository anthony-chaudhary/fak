package humanctl

import (
	"fmt"
	"strings"
	"time"
)

// Delivery says when an admitted semantic instruction may reach its addressee.
// It is orthogonal to Verb: redirect can be immediate or queued without becoming
// two different human intents.
//
// Invariant: delivery mode must be one of immediate, next_safe_point, next_turn, queued, or draft.
// Guard: unrecognized delivery modes fail envelope validation.
type Delivery string

// Delivery modes supported by human control envelopes.
const (
	DeliveryImmediate     Delivery = "immediate"
	DeliveryNextSafePoint Delivery = "next_safe_point"
	DeliveryNextTurn      Delivery = "next_turn"
	DeliveryQueued        Delivery = "queued"
	DeliveryDraft         Delivery = "draft"
)

// AddresseeKind identifies the execution object under control.
//
// Invariant: must be one of turn, session, subagent, cohort, or fleet.
// Guard: invalid addressee kinds fail validation.
type AddresseeKind string

// Addressee kinds supported by human control envelopes.
const (
	AddresseeTurn     AddresseeKind = "turn"
	AddresseeSession  AddresseeKind = "session"
	AddresseeSubagent AddresseeKind = "subagent"
	AddresseeCohort   AddresseeKind = "cohort"
	AddresseeFleet    AddresseeKind = "fleet"
)

// Cardinality keeps one, selected-many, and all-target controls distinct.
//
// Invariant: cardinality must match the addressee kind constraint (one, many, all).
// Guard: mismatched cardinality causes envelope validation to fail closed.
type Cardinality string

// Cardinality constraints for target selection.
const (
	CardinalityOne  Cardinality = "one"
	CardinalityMany Cardinality = "many"
	CardinalityAll  Cardinality = "all"
)

// Addressee identifies who is controlled. Fleet/all intentionally has no IDs;
// a selected subset is a cohort so the scope remains visible.
//
// Invariant: single addressees require exactly 1 ID; cohorts require >= 2 IDs; fleet requires 0 IDs.
// Guard: blank IDs or cardinality mismatches fail closed with an error.
type Addressee struct {
	Kind        AddresseeKind `json:"kind"`
	Cardinality Cardinality   `json:"cardinality"`
	IDs         []string      `json:"ids,omitempty"`
}

// Duration defines how long a control remains effective.
//
// Invariant: duration must be once, turn, session, or until_expiry.
// Guard: unknown duration tokens fail validation.
type Duration string

// Duration spans for control lifetime bounds.
const (
	DurationOnce        Duration = "once"
	DurationTurn        Duration = "turn"
	DurationSession     Duration = "session"
	DurationUntilExpiry Duration = "until_expiry"
)

// Lifetime bounds persistence. ExpiresAt is required only for UntilExpiry.
//
// Invariant: UntilExpiry requires a future timestamp; fixed durations must not have an expiry.
// Guard: non-zero expiry on fixed duration or expired UntilExpiry fails validation.
type Lifetime struct {
	Duration  Duration  `json:"duration"`
	ExpiresAt time.Time `json:"expires_at,omitempty"`
}

// ReceiptStatus reports transport acknowledgement only.
//
// Invariant: must be unacknowledged or acknowledged.
// Guard: unrecognized receipt states fail validation.
type ReceiptStatus string

// Transport receipt states.
const (
	ReceiptUnacknowledged ReceiptStatus = "unacknowledged"
	ReceiptAcknowledged   ReceiptStatus = "acknowledged"
)

// AdmissionStatus reports whether the addressee accepted the instruction.
//
// Invariant: must be pending, accepted, or rejected.
// Guard: unacknowledged receipts cannot hold non-pending admission status.
type AdmissionStatus string

// Control admission states.
const (
	AdmissionPending  AdmissionStatus = "pending"
	AdmissionAccepted AdmissionStatus = "accepted"
	AdmissionRejected AdmissionStatus = "rejected"
)

// EffectStatus reports independently observed execution effect. Acceptance is
// never an effect: EffectObserved requires a non-empty witness.
//
// Invariant: observed effect requires non-empty witness; non-observed must have empty witness.
// Guard: invalid effect statuses or rejected controls with effects fail validation.
type EffectStatus string

// Execution effect observation states.
const (
	EffectUnobserved EffectStatus = "unobserved"
	EffectPending    EffectStatus = "pending"
	EffectObserved   EffectStatus = "observed"
	EffectFailed     EffectStatus = "failed"
)

// Outcome keeps receipt, admission, and effect on separate evidence rungs.
//
// Invariant: receipt, admission, and effect transition through fail-closed verification rungs.
// Guard: rejected controls cannot claim effect, and observed effects require an explicit witness string.
type Outcome struct {
	Receipt       ReceiptStatus   `json:"receipt"`
	Admission     AdmissionStatus `json:"admission"`
	Effect        EffectStatus    `json:"effect"`
	EffectWitness string          `json:"effect_witness,omitempty"`
}

// Envelope binds semantic intent to delivery, addressee, lifetime, and status.
//
// Invariant: drafts must not specify addressee or outcome; non-drafts must satisfy all sub-validations.
// Guard: any inconsistency across instruction, delivery, addressee, lifetime, or outcome causes validation failure.
type Envelope struct {
	Instruction Instruction `json:"instruction"`
	Delivery    Delivery    `json:"delivery"`
	Addressee   Addressee   `json:"addressee,omitempty"`
	Lifetime    Lifetime    `json:"lifetime"`
	Outcome     Outcome     `json:"outcome"`
}

// ValidateAt checks cross-axis combinations at a caller-supplied time so expiry
// behavior is deterministic in tests and replay.
//
// Invariant: validation is pure and evaluated against the provided evaluation timestamp.
// Guard: returns a non-nil error if any axis invariant is violated or if lifetime has expired.
func (e Envelope) ValidateAt(now time.Time) error {
	if err := e.Instruction.Validate(); err != nil {
		return err
	}
	if !validDelivery(e.Delivery) {
		return fmt.Errorf("humanctl: invalid delivery %q", e.Delivery)
	}
	if e.Delivery == DeliveryDraft {
		if e.Addressee.Kind != "" || len(e.Addressee.IDs) != 0 {
			return fmt.Errorf("humanctl: draft cannot have an addressee")
		}
		if e.Outcome != (Outcome{}) {
			return fmt.Errorf("humanctl: draft cannot have an outcome")
		}
	} else if err := e.Addressee.validate(); err != nil {
		return err
	}
	if err := e.Lifetime.validate(now); err != nil {
		return err
	}
	if e.Delivery != DeliveryDraft {
		if err := e.Outcome.validate(); err != nil {
			return err
		}
	}
	return nil
}

func (a Addressee) validate() error {
	for _, id := range a.IDs {
		if strings.TrimSpace(id) == "" {
			return fmt.Errorf("humanctl: addressee IDs cannot be blank")
		}
	}
	switch a.Kind {
	case AddresseeTurn, AddresseeSession, AddresseeSubagent:
		if a.Cardinality != CardinalityOne || len(a.IDs) != 1 {
			return fmt.Errorf("humanctl: %s requires cardinality one and exactly one ID", a.Kind)
		}
	case AddresseeCohort:
		if a.Cardinality != CardinalityMany || len(a.IDs) < 2 {
			return fmt.Errorf("humanctl: cohort requires cardinality many and at least two IDs")
		}
	case AddresseeFleet:
		if a.Cardinality != CardinalityAll || len(a.IDs) != 0 {
			return fmt.Errorf("humanctl: fleet requires cardinality all and no selected IDs")
		}
	default:
		return fmt.Errorf("humanctl: invalid addressee kind %q", a.Kind)
	}
	return nil
}

func (l Lifetime) validate(now time.Time) error {
	switch l.Duration {
	case DurationOnce, DurationTurn, DurationSession:
		if !l.ExpiresAt.IsZero() {
			return fmt.Errorf("humanctl: %s lifetime cannot carry expiry", l.Duration)
		}
	case DurationUntilExpiry:
		if l.ExpiresAt.IsZero() || !l.ExpiresAt.After(now) {
			return fmt.Errorf("humanctl: until_expiry requires a future expiry")
		}
	default:
		return fmt.Errorf("humanctl: invalid duration %q", l.Duration)
	}
	return nil
}

func (o Outcome) validate() error {
	if o.Receipt != ReceiptUnacknowledged && o.Receipt != ReceiptAcknowledged {
		return fmt.Errorf("humanctl: invalid receipt status %q", o.Receipt)
	}
	if o.Admission != AdmissionPending && o.Admission != AdmissionAccepted && o.Admission != AdmissionRejected {
		return fmt.Errorf("humanctl: invalid admission status %q", o.Admission)
	}
	if o.Effect != EffectUnobserved && o.Effect != EffectPending && o.Effect != EffectObserved && o.Effect != EffectFailed {
		return fmt.Errorf("humanctl: invalid effect status %q", o.Effect)
	}
	if o.Receipt == ReceiptUnacknowledged && o.Admission != AdmissionPending {
		return fmt.Errorf("humanctl: unacknowledged control cannot have an admission decision")
	}
	if o.Admission == AdmissionRejected && o.Effect != EffectUnobserved {
		return fmt.Errorf("humanctl: rejected control cannot report an effect")
	}
	if o.Effect == EffectObserved && strings.TrimSpace(o.EffectWitness) == "" {
		return fmt.Errorf("humanctl: observed effect requires a witness")
	}
	if o.Effect != EffectObserved && strings.TrimSpace(o.EffectWitness) != "" {
		return fmt.Errorf("humanctl: effect witness requires observed status")
	}
	return nil
}

func validDelivery(d Delivery) bool {
	return d == DeliveryImmediate || d == DeliveryNextSafePoint || d == DeliveryNextTurn || d == DeliveryQueued || d == DeliveryDraft
}

// InstructionFromSessionDecision maps the existing policy wire tokens without
// changing them. END_TURN remains distinct from pause because only pause is a
// resumable session state.
//
// Invariant: token mappings preserve distinct semantic verbs for CONTINUE, END_TURN, PAUSE_SESSION, and STOP_SESSION.
// Guard: unknown tokens fail closed returning ok=false and a zero-value Instruction.
func InstructionFromSessionDecision(token string) (Instruction, bool) {
	var verb Verb
	switch token {
	case "CONTINUE":
		verb = Continue
	case "END_TURN":
		verb = EndTurn
	case "PAUSE_SESSION":
		verb = Pause
	case "STOP_SESSION":
		verb = Stop
	default:
		return Instruction{}, false
	}
	return Instruction{Verb: verb}, true
}
