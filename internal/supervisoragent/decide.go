package supervisoragent

// This file is build-plan step S2 of the Chief-of-Staff runtime note
// (fak-private docs/notes/CONCEPT-CHIEF-OF-STAFF-AGENT-RUNTIME-2026-09-21.md,
// sections 6.1 and 6.2; issue #13464): the PURE decision seam of the seat
// loop. Decide maps the assembled SupervisorInput, the earned Envelope, and
// the soak mode to exactly one proposed SupervisorAction and its closed
// Reason — deterministic-first, no I/O, no clock reads, no randomness, and
// never a transcript (the input type has none). The lower stage (S3) binds
// the proposal to execution through LowerInEnvelope; this file only decides.

// Reason is the closed reason token Decide returns alongside its proposed
// verb. The vocabulary is pinned by TestDecideReasonVocabularyClosed; adding
// a token is a deliberate, test-breaking act, like widening the verb union.
type Reason string

const (
	// ReasonWitnessAbsent: an input surface's witness could not be obtained —
	// the fail-closed rung fires before any other rung, on every path.
	ReasonWitnessAbsent Reason = "witness_absent"
	// ReasonEnvelopeRefused: the envelope refused the proposed verb — either
	// the earned width is below demand under enforce (proposed for
	// confirmation, nothing runs) or a witness_refused row collapsed the
	// envelope (replaced by the narrowing escalate verb).
	ReasonEnvelopeRefused Reason = "envelope_refused"
	// ReasonHoldGreen: every witness is present and no typed pressure witness
	// is active — the fleet is green; holding costs nothing.
	ReasonHoldGreen Reason = "hold_green"
	// ReasonSpawnAdmitted: reserved for a fresh spawn admitted through the
	// lane arbitration. No rung emits it yet: the input projection carries
	// no open-work witness a spawn could key on, and manufacturing one would
	// turn the seat into a recognizer.
	ReasonSpawnAdmitted Reason = "spawn_admitted"
	// ReasonReplaceStaleLease: a dead/stale worker verdict supersedes its run
	// through the dispatch admit path — the one verb action.go assigns to
	// dead/stale runs.
	ReasonReplaceStaleLease Reason = "replace_stale_lease"
	// ReasonReplanDrift: the liveness witness drifted from the census — the
	// run reports stalled while its witnessed owner is done; the issue is
	// re-admitted for a fresh dispatch.
	ReasonReplanDrift Reason = "replan_drift"
	// ReasonWidenCapacity: reserved for a re-arbitration of a held lane to a
	// wider tree. No rung emits it yet: widening is a capacity request, not
	// a health reaction, and no typed witness in the projection demands one.
	ReasonWidenCapacity Reason = "widen_capacity"
	// ReasonEscalateResidual: the residual rung — a situation the
	// deterministic rungs cannot resolve safely (a blocked worker, a stalled
	// run with no witnessed completion) hands to the operator.
	ReasonEscalateResidual Reason = "escalate_residual"
)

// stalledClass is the liveness-class token that witnesses drift: the run has
// stopped advancing (input.go documents "stalled" as the class token the
// dos_status digest emits opposite "moving").
const stalledClass = "stalled"

// Decide is the seat's pure decision function: it maps the assembled
// SupervisorInput, the earned Envelope, and the soak mode to exactly one
// proposed SupervisorAction and its closed Reason. It performs no I/O, reads
// no clock, and draws no randomness — the DecidedAt a journal record carries
// is the value the wake-time caller passes in, never a read here.
//
// The deterministic-first ladder (design-note 6.1):
//
//  1. ANY absent/unwitnessed surface -> escalate with ReasonWitnessAbsent.
//     Fail-closed on every path, including paths where a verb would otherwise
//     be admitted: a green absence is not a green witness.
//  2. The envelope gates the proposed verb (Authorize). An escalate verdict
//     (a witness_refused row collapsed the earned width) replaces the
//     proposal with the narrowing escalate verb; a confirm verdict (below
//     earned width under enforce) keeps the proposal but tags it
//     ReasonEnvelopeRefused so the lower stage refuses it and the journal
//     records the refusal. Only an unattended verdict lets the proposal stand
//     with its positive reason.
//  3. A green fleet — no typed pressure witness active — holds with
//     ReasonHoldGreen. Holding costs nothing and touches nothing.
//  4. Otherwise the strongest typed witness maps to its verb:
//     - a dead or stale worker verdict -> replace (supersede the run through
//     the dispatch admit path; the one verb action.go assigns to dead/stale
//     runs). First such verdict in census order — deterministic.
//     - a blocked worker verdict -> escalate (its blocker class is folded
//     auth/rate/attention evidence the seat cannot resolve, and re-admitting
//     the issue without superseding the live run risks two live workers on
//     one issue, so the operator judges).
//     - a stalled liveness digest whose run has a witnessed owner verdict in
//     the done state -> replan (the liveness/census drift means finished
//     work was never advanced; re-admit the issue for a fresh dispatch —
//     safe because the owner is done).
//     - a stalled liveness digest with no done owner -> escalate (an alive,
//     silent run the census does not explain is a human-residual finding,
//     never a guess).
//
// Open escalation packets are passive context: they are already routed to the
// operator, and re-emitting them would double-record a handoff that happened.
// They do not change the proposed verb.
func Decide(in SupervisorInput, env Envelope, mode EnvelopeMode) (SupervisorAction, Reason) {
	// Rung 1: fail-closed on any withheld witness, before anything else.
	if in.AnyAbsent() {
		return EscalateAction{Class: "witness_absent", Severity: "operator", ReasonCode: "witness_absent"}, ReasonWitnessAbsent
	}

	proposed, reason := pressureVerb(in)

	// Rung 2: the envelope gates the proposal.
	v := Authorize(in, env, mode, proposed)
	switch v.Auth {
	case AuthEscalate:
		// Collapsed envelope (witness_refused row): the mandated fail-closed
		// response IS escalation, and the narrowing verb stays executable
		// under every posture, so this can never deadlock.
		return EscalateAction{Class: "envelope_refused", Severity: "operator", ReasonCode: string(v.Reason)}, ReasonEnvelopeRefused
	case AuthConfirm:
		// Below earned width under enforce: the proposal stands for a human
		// to confirm; the lower stage refuses it and the journal records the
		// refusal. No admission verb runs.
		return proposed, ReasonEnvelopeRefused
	default: // AuthUnattended
		return proposed, reason
	}
}

// pressureVerb maps the strongest active typed witness to its verb (ladder
// rungs 3-4). It is a pure total function over a fully-witnessed input:
// every call returns exactly one verb and one positive reason.
func pressureVerb(in SupervisorInput) (SupervisorAction, Reason) {
	workers := in.Workers.Value
	live := in.Liveness.Value

	// Strongest: a dead or stale run holds a lease and blocks its issue;
	// supersede it. First such verdict in census order (deterministic).
	for _, w := range workers {
		if w.State == WorkerDead || w.State == WorkerStale {
			return ReplaceAction{RunID: w.RunID, Issue: w.Issue, Lane: w.Lane}, ReasonReplaceStaleLease
		}
	}

	// A blocked worker's blocker is folded auth/rate/attention evidence the
	// seat cannot resolve; handing it to the operator is the only move that
	// cannot create a second live worker on the issue.
	for _, w := range workers {
		if w.State == WorkerBlocked {
			return EscalateAction{RunID: w.RunID, Issue: w.Issue, Class: "worker_blocked", Severity: "operator", ReasonCode: "worker_blocked"}, ReasonEscalateResidual
		}
	}

	// Liveness drift: the run reports stalled.
	if live.Class == stalledClass {
		if issue, lane, state, ok := ownerOf(workers, live.RunID); ok && state == WorkerDone {
			return RedispatchAction{Issue: issue, Lane: lane}, ReasonReplanDrift
		}
		return EscalateAction{RunID: live.RunID, Class: "run_stalled", Severity: "operator", ReasonCode: "run_stalled"}, ReasonEscalateResidual
	}

	// Green: every witness present, no typed pressure active. Hold.
	return HoldAction{}, ReasonHoldGreen
}

// ownerOf finds the census verdict that serves the named run. ok is false
// when the run has no witnessed owner — unwitnessed ownership is treated as a
// residual, never inferred.
func ownerOf(workers []WorkerVerdict, runID string) (issue, lane string, state WorkerState, ok bool) {
	for _, w := range workers {
		if w.RunID == runID {
			return w.Issue, w.Lane, w.State, true
		}
	}
	return "", "", "", false
}
