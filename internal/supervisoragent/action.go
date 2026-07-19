package supervisoragent

import (
	"errors"
	"fmt"
)

// This file is fence #3 of the supervisor-seat doctrine (epic #4477, leaf
// #4479): the closed ACTION VOCABULARY a supervisor agent is restricted to.
// Every action lowers to an EXISTING deterministic admission verb, so the
// agent's blast radius is exactly a human operator's — no wider. No action
// reaches a raw shell, and no path spawns outside the one spawn authority.

// ActionKind is the closed verb token of the supervisor vocabulary: exactly
// spawn / replace / replan / widen / escalate / hold. The set is pinned by
// TestSupervisorActionClosedVocabulary; widening it is a deliberate,
// test-breaking act.
type ActionKind string

const (
	ActionSpawn   ActionKind = "spawn"
	ActionReplace ActionKind = "replace"
	// ActionRedispatch carries the vocabulary token "replan". The Go symbol is
	// deliberately spelled redispatch — outside the planning concept family —
	// while the wire token keeps the issue's closed six-verb spelling.
	ActionRedispatch ActionKind = "replan"
	ActionWiden      ActionKind = "widen"
	ActionEscalate   ActionKind = "escalate"
	ActionHold       ActionKind = "hold"
)

// SupervisorAction is the closed union of the six verbs. It is SEALED: the
// marker method is unexported, so no type outside this package can satisfy it,
// and Lower rejects anything that is not one of the six case types — an
// out-of-vocabulary action cannot be expressed, and an in-package rogue is
// refused (TestOutOfVocabularyActionRejected). Each case carries only the
// typed args its deterministic verb needs; none carries free text.
type SupervisorAction interface {
	Kind() ActionKind
	isSupervisorAction()
}

// SpawnAction requests a NEW worker: it lowers to the dos_arbitrate lane
// admission, whose granted lease row is the witnessed artifact the existing
// dispatch path then launches under. There is no private spawn: without the
// arbiter's grant, nothing runs.
type SpawnAction struct {
	Issue string // the issue the new worker will own
	Lane  string // the lane to arbitrate for it
}

func (SpawnAction) Kind() ActionKind    { return ActionSpawn }
func (SpawnAction) isSupervisorAction() {}

// ReplaceAction supersedes a dead/stale run with a fresh worker for the same
// issue: it lowers to the dispatch admit path, naming the run it supersedes so
// two live workers can never hold one issue. The admit receipt is the witness.
type ReplaceAction struct {
	RunID string // the run being superseded (from its fleetmon verdict)
	Issue string // the issue the replacement will own
	Lane  string // the lane it runs under
}

func (ReplaceAction) Kind() ActionKind    { return ActionReplace }
func (ReplaceAction) isSupervisorAction() {}

// RedispatchAction (vocabulary token "replan") re-admits an issue for a fresh
// dispatch through the SAME dispatch admit path a first dispatch uses —
// replanning is a re-admission, not a new authority. The admit receipt is the
// witness.
type RedispatchAction struct {
	Issue string // the issue to re-dispatch
	Lane  string // the lane to run it under
}

func (RedispatchAction) Kind() ActionKind    { return ActionRedispatch }
func (RedispatchAction) isSupervisorAction() {}

// WidenAction asks for a wider tree on a held lane. It lowers to dos_arbitrate
// as a RE-arbitration — the same admission a first grant goes through, which
// may refuse. It is not a bypass of the lease rule, and there is deliberately
// no force variant.
type WidenAction struct {
	Lane string   // the held lane being re-arbitrated
	Tree []string // the requested (wider) repo-relative globs
}

func (WidenAction) Kind() ActionKind    { return ActionWiden }
func (WidenAction) isSupervisorAction() {}

// EscalateAction hands a situation to the operator: it lowers to the
// fak.escalation.v1 packet emit (#2271), carrying only the typed head fields —
// closed tokens, never prose. The emitted packet (with its assigned id) is the
// witness.
type EscalateAction struct {
	RunID      string // the run it concerns
	Issue      string // the issue it concerns
	Class      string // escalation class token
	Severity   string // severity token: status / operator
	ReasonCode string // closed refusal-reason token (never prose)
}

func (EscalateAction) Kind() ActionKind    { return ActionEscalate }
func (EscalateAction) isSupervisorAction() {}

// HoldAction is the deliberate no-op: do nothing this wake. It lowers to no
// verb at all and leaves no artifact — holding costs nothing and touches
// nothing, which is why it is always a safe default.
type HoldAction struct{}

func (HoldAction) Kind() ActionKind    { return ActionHold }
func (HoldAction) isSupervisorAction() {}

// AdmitReceipt is the payload-free witness the dispatch admit path leaves for
// an admitted worker: the new run's identity, what it owns, and (for a
// replace) the run it superseded. No plan body or transcript rides along.
type AdmitReceipt struct {
	RunID      string // the admitted run
	Issue      string // the issue it was admitted for
	Lane       string // the lane it runs under
	Supersedes string // the run it replaces; empty for a fresh plan
}

// AdmissionVerbs is the CLOSED set of deterministic admission calls an action
// may lower to — the existing verbs, and nothing else: the dos_arbitrate lane
// admission, the dispatch admit path (the one spawn authority), and the
// escalation packet emit. The method set is pinned by
// TestAdmissionVerbsClosedMethodSet; there is no shell method to reach. The
// live seat binds these to the real calls; tests bind a recording double.
// Every method returns the witnessed artifact the underlying verb leaves, or
// its refusal.
type AdmissionVerbs interface {
	// Arbitrate is the dos_arbitrate lane admission. A nil tree asks for the
	// lane's own scope (spawn); a non-nil tree is a re-arbitration for that
	// scope (widen). It returns the granted lease row or the arbiter's refusal.
	Arbitrate(lane string, tree []string) (Lease, error)
	// Admit is the dispatch admit path — the one spawn authority. A non-empty
	// supersedes names the run being replaced. It returns the admit receipt or
	// the admit path's refusal.
	Admit(issue, lane, supersedes string) (AdmitReceipt, error)
	// EmitEscalation appends a fak.escalation.v1 packet (#2271) built from the
	// typed head and returns it with its assigned id, or the emit's refusal.
	EmitEscalation(head Escalation) (Escalation, error)
}

// LoweredVerb names which deterministic admission call an executed action
// lowered to — the audit token an ActionEffect carries.
type LoweredVerb string

const (
	VerbArbitrate      LoweredVerb = "dos_arbitrate"   // lane admission / re-arbitration
	VerbAdmit          LoweredVerb = "dispatch_admit"  // the one spawn authority
	VerbEmitEscalation LoweredVerb = "escalation_emit" // fak.escalation.v1 packet emit
	VerbNone           LoweredVerb = "none"            // hold: no verb ran
)

// ActionEffect is the witnessed record of one executed action: which verb it
// lowered to and the artifact that verb left. Exactly one artifact field is
// set for an artifact-leaving verb; hold sets none. An effect exists only for
// an action that was actually admitted — a refusal returns an error and no
// effect.
type ActionEffect struct {
	Action ActionKind    // the vocabulary verb that was executed
	Verb   LoweredVerb   // the deterministic admission call it lowered to
	Lease  *Lease        // spawn / widen: the arbiter's granted row
	Admit  *AdmitReceipt // replace / replan: the dispatch admit receipt
	Packet *Escalation   // escalate: the emitted packet head (id assigned)
}

// ErrOutOfVocabulary is the rejection for any action that is not one of the six
// closed union cases (including a nil action). Nothing runs for a rejected
// action — no verb is reached, not even partially.
var ErrOutOfVocabulary = errors.New("supervisoragent: action outside the closed vocabulary")

// Lower executes one supervisor action by lowering it to its deterministic
// admission verb and returns the witnessed effect. It is the ONLY execution
// path for the vocabulary: spawn/widen go through Arbitrate, replace/replan
// through Admit, escalate through EmitEscalation, and hold through nothing. A
// verb's refusal propagates unchanged — this layer never retries, forces, or
// widens around an admission. Anything outside the union is rejected with
// ErrOutOfVocabulary before any verb runs.
func Lower(a SupervisorAction, v AdmissionVerbs) (ActionEffect, error) {
	switch act := a.(type) {
	case SpawnAction:
		lease, err := v.Arbitrate(act.Lane, nil)
		if err != nil {
			return ActionEffect{}, err
		}
		return ActionEffect{Action: ActionSpawn, Verb: VerbArbitrate, Lease: &lease}, nil
	case WidenAction:
		lease, err := v.Arbitrate(act.Lane, act.Tree)
		if err != nil {
			return ActionEffect{}, err
		}
		return ActionEffect{Action: ActionWiden, Verb: VerbArbitrate, Lease: &lease}, nil
	case ReplaceAction:
		receipt, err := v.Admit(act.Issue, act.Lane, act.RunID)
		if err != nil {
			return ActionEffect{}, err
		}
		return ActionEffect{Action: ActionReplace, Verb: VerbAdmit, Admit: &receipt}, nil
	case RedispatchAction:
		receipt, err := v.Admit(act.Issue, act.Lane, "")
		if err != nil {
			return ActionEffect{}, err
		}
		return ActionEffect{Action: ActionRedispatch, Verb: VerbAdmit, Admit: &receipt}, nil
	case EscalateAction:
		packet, err := v.EmitEscalation(Escalation{
			RunID:      act.RunID,
			Issue:      act.Issue,
			Class:      act.Class,
			Severity:   act.Severity,
			ReasonCode: act.ReasonCode,
		})
		if err != nil {
			return ActionEffect{}, err
		}
		return ActionEffect{Action: ActionEscalate, Verb: VerbEmitEscalation, Packet: &packet}, nil
	case HoldAction:
		return ActionEffect{Action: ActionHold, Verb: VerbNone}, nil
	default:
		return ActionEffect{}, fmt.Errorf("%w: %T", ErrOutOfVocabulary, a)
	}
}
