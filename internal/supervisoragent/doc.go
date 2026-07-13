// Package supervisoragent defines the closed, payload-free INPUT CONTRACT that a
// supervisor agent consumes — fence #1 of the supervisor-seat doctrine
// (docs/notes/CONCEPT-SUPERVISOR-AGENT-SEAT-2026-07-13.md; epic #4477, leaf
// #4478).
//
// The supervisor seat is the meta-loop actor that reads the fleet's already-
// witnessed state and decides the next move (spawn/replace/replan/widen/escalate/
// hold). The no-babysitting doctrine forbids an agent that MANUFACTURES a health
// signal — a transcript-reading "is it stuck?" recognizer in an arms race. It
// permits an agent that CONSUMES a witnessed signal and picks a deterministic
// action. The line between the two is structural, not a matter of good
// intentions, and this package is where the structure lives.
//
// SupervisorInput is a fixed projection of typed witnesses alone: the dos_status
// liveness/progress digest, the fleetmon per-worker verdict, the open
// fak.escalation.v1 packets, and the live lease table — each a payload-free
// field. There is, by construction, NO transcript, log body, or free-text field:
// that single field is what would turn the agent into a recognizer, so the
// projection's closedness is the whole point. The Witness (input_test.go) pins the
// closed field set so a "just in case" payload field cannot be added silently.
//
// A witness that could not be obtained is marked absent (Witnessed.Present ==
// false). Absence is an explicit signal the decision layer MUST escalate on, never
// infer around: a green absence is not a green witness.
package supervisoragent
