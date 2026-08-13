package modelroute

import "github.com/anthony-chaudhary/fak/internal/modelroute/inputtrigger"

// CLASSIFYING THE TURN'S INPUT SHAPE ONCE, AT INGRESS (#6419).
//
// The routing spine already answers "which model" from a Subject's tags. What it could
// not say is what KIND of input opened the turn — a returning tool result, a user typing,
// a prefilled assistant continuation — even though that is a boundary fact the ingress
// seam holds in its hands and then throws away. Every consumer that wanted it re-derived
// it from the raw prompt, and a re-derived taxonomy drifts.
//
// So the messages are read exactly once, here, at admission; the answer rides on the
// Subject as a typed value; and route policy MATCHES on the enum (Match.InputTrigger)
// instead of re-reading text. Route echoes the Subject into the Decision, so the trigger
// a route was chosen under is in the audit trace by construction.
//
// THE FLOOR IS UNMOVED. A trigger is a routing hint. ClassOf/PolicyFor — the work-class
// and tier-floor half — do not read it and must not start: the turn's messages are
// attacker-influenced, so a shape may buy a cheaper MODEL under an unchanged floor, never
// a lower floor. inputtrigger's package doc carries the same statement at its source.

// AdmitTurn stamps s with the input shape of the admitted turn and returns it. This is
// THE ingress call: everything downstream reads Subject.InputTrigger.
//
// It takes and returns a Subject by value (the package's pure, data-in/data-out idiom)
// so a caller composes it into the subject it was already building:
//
//	dec := manifest.Route(modelroute.AdmitTurn(subj, turn))
//
// Calling it twice with the same turn is harmless — Classify is deterministic, so the
// stamp is idempotent — but calling it once is the contract: a second classification at a
// later seam is exactly the drift this exists to prevent. An empty turn classifies as
// inputtrigger.Other, never as the unclassified empty value, so "the ingress ran and saw
// nothing nameable" stays distinguishable from "no ingress ever looked".
func AdmitTurn(s Subject, turn []inputtrigger.Message) Subject {
	s.InputTrigger = inputtrigger.Classify(turn)
	return s
}
