package supervisoragent

import (
	"errors"
	"fmt"
	"strings"
)

// This file is fence #4 of the supervisor-seat doctrine (epic #4477, leaf
// #4480): the AUTHORIZATION ENVELOPE around the closed action vocabulary. The
// agent may handle unattended only the interrupt classes it has EARNED via the
// #2274 autonomy ratchet; below the earned width an action is proposed and a
// human confirms; a withheld witness collapses the envelope (escalate, never
// act); and the seat itself is an experiment whose keep-bit is the #2269
// babysitting counter — touches per witnessed shipped unit — over a soak.
//
// Fail-closed is structural here, not a convention:
//   - a class missing from the earned-width table reads as width 0;
//   - a malformed soak-switch token resolves to enforce, never to "off";
//   - an out-of-vocabulary action escalates before any admission verb runs;
//   - a witness_refused row or an absent input surface collapses the earned
//     width for every widening verb in BOTH modes — the soak switch cannot
//     soften it;
//   - an unmeasured counter can never testify keep.
// The two narrowing verbs (hold, escalate) demand width 0 and so remain inside
// every envelope, including the collapsed one: the mandated fail-closed
// response IS escalation, so the escape hatch can never deadlock.

// EnvelopeModeEnv is the soak switch's environment variable. The package never
// reads the environment itself (it stays pure); the wake-time caller passes
// the raw value through ModeFromValue. It mirrors #3517's operator soak
// switch: unset/warn soaks (verdicts are recorded, nothing new is refused),
// enforce refuses.
const EnvelopeModeEnv = "FAK_SUPERVISOR_ENVELOPE_MODE"

// EnvelopeMode is the warn->enforce soak switch for the ratchet's sub-envelope
// rule. It scopes ONLY that rule: the fail-closed collapses (refused witness,
// absent surface, out-of-vocabulary) behave identically in both modes.
type EnvelopeMode string

const (
	// ModeWarn is the soak default: a sub-envelope action still executes, but
	// its verdict records FindingWouldConfirm so the soak leaves a typed trail.
	ModeWarn EnvelopeMode = "warn"
	// ModeEnforce is the earned envelope live: a sub-envelope action is
	// proposed (ErrConfirmRequired) and does not reach its admission verb.
	ModeEnforce EnvelopeMode = "enforce"
)

// ModeFromValue resolves the soak switch from a raw environment value. Unset
// and "warn" soak; "enforce" enforces; ANY other token resolves to enforce —
// a malformed switch can only tighten the envelope, never widen it.
func ModeFromValue(v string) EnvelopeMode {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", string(ModeWarn):
		return ModeWarn
	default:
		return ModeEnforce
	}
}

// classDemand is the pre-declared ladder, committed as data (#2274's
// "pre-declared ladder of envelope steps"): the earned width each action class
// demands before it may run unattended. The ordering encodes blast radius —
// hold and escalate are narrowing verbs (width 0, always inside the
// envelope); re-admitting an issue is the mildest widening verb; superseding a
// live run and widening a held lane demand the most earned trust. The ladder
// is pinned by TestEnvelopeLadderCoversVocabulary; #2274 owns the EARNING
// formula, this table only prices the classes.
var classDemand = map[ActionKind]int{
	ActionHold:       0,
	ActionEscalate:   0,
	ActionRedispatch: 1,
	ActionSpawn:      2,
	ActionReplace:    3,
	ActionWiden:      4,
}

// DemandFor reads the ladder: the width class k demands before it may run
// unattended. ok is false for a class outside the closed vocabulary — the
// caller must fail closed on it, never default it.
func DemandFor(k ActionKind) (demand int, ok bool) {
	demand, ok = classDemand[k]
	return demand, ok
}

// Envelope is the earned authorization envelope read from the #2274 ratchet
// surface: the per-class earned widths, plus the fail-closed bit. It is a thin
// typed adapter — the ratchet owns HOW width is earned; this package only
// consumes the result (issue #4480's stated assumption).
type Envelope struct {
	// Widths is the earned width per action class, as the ratchet reports it.
	// A missing class (or a nil map) reads as width 0: an unconfigured
	// envelope never defaults open.
	Widths map[ActionKind]int
	// WitnessRefused reports that a witness_refused row was observed in the
	// ratchet's window. It collapses the envelope for every widening verb:
	// escalate, never act. A green ABSENCE of rows is not this bit's false —
	// the ratchet surface sets it from rows it actually read.
	WitnessRefused bool
}

// EarnedWidth reads the earned width for one class; a missing entry is 0.
func (e Envelope) EarnedWidth(k ActionKind) int { return e.Widths[k] }

// Authorization is the closed three-way verdict on one action.
type Authorization string

const (
	// AuthUnattended: the action is at/above its earned width (or is a
	// narrowing verb) and executes without a human.
	AuthUnattended Authorization = "unattended"
	// AuthConfirm: the action is below its earned width under enforce — it is
	// proposed, and a human confirms before anything runs.
	AuthConfirm Authorization = "confirm"
	// AuthEscalate: the fail-closed verdict — a refused/absent witness or a
	// malformed action. The seat must escalate; the action never runs.
	AuthEscalate Authorization = "escalate"
)

// Closed reason tokens an EnvelopeVerdict carries — tokens, never prose.
const (
	// ReasonNarrowing: hold/escalate — width-0 verbs inside every envelope.
	ReasonNarrowing = "NARROWING_VERB"
	// ReasonSuperEnvelope: earned width at/above the class demand.
	ReasonSuperEnvelope = "SUPER_ENVELOPE"
	// ReasonSubEnvelope: earned width below the class demand.
	ReasonSubEnvelope = "SUB_ENVELOPE"
	// ReasonRefused: a witness_refused row collapsed the envelope.
	ReasonRefused = "WITNESS_REFUSED"
	// ReasonSurfaceAbsent: an input surface's witness could not be obtained.
	ReasonSurfaceAbsent = "SURFACE_ABSENT"
	// ReasonOutOfVocabulary: a nil or out-of-union action.
	ReasonOutOfVocabulary = "OUT_OF_VOCABULARY"
)

// FindingWouldConfirm is the warn-mode soak record: this action executed, but
// under enforce it would have required confirmation. The soak's typed trail is
// these findings — they are what the warn->enforce flip is judged on.
const FindingWouldConfirm = "WOULD_CONFIRM"

// EnvelopeVerdict is the typed, payload-free record of one authorization: the
// class, the mode it was judged under, the three-way verdict, the widths that
// produced it, a closed reason token, and (warn mode only) the soak finding.
type EnvelopeVerdict struct {
	Action  ActionKind
	Mode    EnvelopeMode
	Auth    Authorization
	Demand  int
	Earned  int
	Reason  string
	Finding string
}

// ErrConfirmRequired is the enforce-mode refusal for a sub-envelope action:
// it is proposed for a human to confirm; no admission verb has run.
var ErrConfirmRequired = errors.New("supervisoragent: action below the earned envelope width — operator confirmation required")

// ErrEnvelopeFailClosed is the fail-closed refusal: a refused or absent
// witness (or a malformed action) — the seat must escalate, never act.
var ErrEnvelopeFailClosed = errors.New("supervisoragent: envelope fail-closed — escalate, never act")

// Authorize is the pure per-action verdict: it gates one action class by the
// earned envelope width under the given soak mode. It performs no I/O and
// executes nothing — LowerInEnvelope binds the verdict to execution. The
// check order is fail-closed by construction: vocabulary first, then the
// witness collapses, then the width comparison; only the LAST step is
// soakable.
func Authorize(in SupervisorInput, env Envelope, mode EnvelopeMode, a SupervisorAction) EnvelopeVerdict {
	class, ok := classOf(a)
	if !ok {
		return EnvelopeVerdict{Action: class, Mode: mode, Auth: AuthEscalate, Reason: ReasonOutOfVocabulary}
	}
	demand, ok := classDemand[class]
	if !ok {
		// Unreachable while classOf and the ladder agree; kept so a widened
		// union without a ladder row fails closed instead of defaulting.
		return EnvelopeVerdict{Action: class, Mode: mode, Auth: AuthEscalate, Reason: ReasonOutOfVocabulary}
	}
	if demand == 0 {
		// Narrowing verbs stay executable under every posture below — that is
		// what makes the fail-closed escalation path deadlock-free.
		return EnvelopeVerdict{Action: class, Mode: mode, Auth: AuthUnattended, Demand: demand, Earned: env.EarnedWidth(class), Reason: ReasonNarrowing}
	}
	if env.WitnessRefused {
		return EnvelopeVerdict{Action: class, Mode: mode, Auth: AuthEscalate, Demand: demand, Reason: ReasonRefused}
	}
	if in.AnyAbsent() {
		return EnvelopeVerdict{Action: class, Mode: mode, Auth: AuthEscalate, Demand: demand, Reason: ReasonSurfaceAbsent}
	}
	earned := env.EarnedWidth(class)
	v := EnvelopeVerdict{Action: class, Mode: mode, Demand: demand, Earned: earned}
	if earned >= demand {
		v.Auth, v.Reason = AuthUnattended, ReasonSuperEnvelope
		return v
	}
	v.Reason = ReasonSubEnvelope
	if mode == ModeEnforce {
		v.Auth = AuthConfirm
		return v
	}
	// Warn: allowed through, with the would-be confirmation recorded.
	v.Auth, v.Finding = AuthUnattended, FindingWouldConfirm
	return v
}

// LowerInEnvelope authorizes one action and, only on an unattended verdict,
// lowers it through the closed admission path (Lower). A confirm verdict
// returns ErrConfirmRequired and an escalate verdict ErrEnvelopeFailClosed —
// in both cases NO admission verb runs and no artifact exists. The verdict is
// returned alongside so the caller can ledger it (the warn-mode soak trail).
func LowerInEnvelope(a SupervisorAction, v AdmissionVerbs, in SupervisorInput, env Envelope, mode EnvelopeMode) (EnvelopeVerdict, ActionEffect, error) {
	d := Authorize(in, env, mode, a)
	switch d.Auth {
	case AuthConfirm:
		return d, ActionEffect{}, fmt.Errorf("%w: %s demands width %d, earned %d", ErrConfirmRequired, d.Action, d.Demand, d.Earned)
	case AuthEscalate:
		return d, ActionEffect{}, fmt.Errorf("%w (%s)", ErrEnvelopeFailClosed, d.Reason)
	}
	eff, err := Lower(a, v)
	return d, eff, err
}

// classOf names the class of an action iff it is one of the six closed union
// cases. A nil action or an in-package rogue reports ok == false.
func classOf(a SupervisorAction) (ActionKind, bool) {
	switch a.(type) {
	case SpawnAction, ReplaceAction, RedispatchAction, WidenAction, EscalateAction, HoldAction:
		return a.Kind(), true
	default:
		return "", false
	}
}

// KeepSample is one soak checkpoint of the #2269 babysitting counter: the
// touches-per-witnessed-shipped-unit KPI head, projected payload-free.
// Measured == false mirrors the counter's not_yet posture (no witnessed unit
// in the window — the ratio is undefined, not zero). Value is only meaningful
// when Measured. The wake-time caller folds the counter's report into this
// head; #2269 owns the counter's definition.
type KeepSample struct {
	Measured bool
	Value    float64 // touches per witnessed shipped unit
}

// KeepBit is the seat experiment's verdict token: keep the seat or revert it.
type KeepBit string

const (
	KeepSeat   KeepBit = "keep"
	RevertSeat KeepBit = "revert"
)

// Closed reason tokens a KeepVerdict carries.
const (
	// KeepReasonHeld: every measured checkpoint stayed at/below the baseline.
	KeepReasonHeld = "TOUCHES_HELD"
	// KeepReasonRose: some measured checkpoint rose above the baseline.
	KeepReasonRose = "TOUCHES_ROSE"
	// KeepReasonBaselineUnmeasured: no measured pre-seat baseline exists.
	KeepReasonBaselineUnmeasured = "BASELINE_UNMEASURED"
	// KeepReasonSoakUnmeasured: no soak checkpoint was measured.
	KeepReasonSoakUnmeasured = "SOAK_UNMEASURED"
)

// KeepVerdict is the keep-or-revert verdict with its evidence: the baseline
// and the worst (peak) measured soak value that produced the bit.
type KeepVerdict struct {
	Bit      KeepBit
	Baseline float64
	Peak     float64
	Reason   string
}

// KeepBitVerdict renders the seat experiment's verdict from the babysitting
// counter series: the pre-seat baseline and the soak checkpoints. The rule is
// the doctrine's ratchet: touches per witnessed shipped unit only goes down —
// ANY measured rise above the baseline reverts the seat, even if the tail
// recovered. Fail-closed: an unmeasured baseline, or a soak with no measured
// checkpoint, can never testify keep. The verdict is a pure function of the
// series — deliberately independent of EnvelopeMode, so the warn->enforce
// ratchet cannot move the keep-bit (TestKeepBitPreservedAcrossRatchet).
func KeepBitVerdict(baseline KeepSample, soak []KeepSample) KeepVerdict {
	if !baseline.Measured {
		return KeepVerdict{Bit: RevertSeat, Reason: KeepReasonBaselineUnmeasured}
	}
	peak, seen := 0.0, false
	for _, s := range soak {
		if !s.Measured {
			continue
		}
		if !seen || s.Value > peak {
			peak = s.Value
		}
		seen = true
	}
	if !seen {
		return KeepVerdict{Bit: RevertSeat, Baseline: baseline.Value, Reason: KeepReasonSoakUnmeasured}
	}
	if peak > baseline.Value {
		return KeepVerdict{Bit: RevertSeat, Baseline: baseline.Value, Peak: peak, Reason: KeepReasonRose}
	}
	return KeepVerdict{Bit: KeepSeat, Baseline: baseline.Value, Peak: peak, Reason: KeepReasonHeld}
}
