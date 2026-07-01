package sessionreset

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// objective.go — issue #1583: wires ctxplan's OBJECTIVE PIN (ctxplan/objective.go) into the
// reset-carryover contract this package already owns. taskDistill (contributors.go) already
// extracts an objective EXTRACTIVELY (the first durable user line) for the human-readable
// recap; what was missing is a STABLE, ADDRESSABLE identity for that objective that a host
// can carry across the reset boundary and CHECK for preservation — not just re-derive text
// that happens to look similar.
//
// THE CONTRACT. A host that wants the #1583 continuity guarantee calls, in order:
//
//  1. PinObjective(in)               — mint/derive the pin for the session ABOUT to reset
//     (or reuse the PRIOR pin's PinID via RepinObjective so identity carries forward).
//  2. BuildSeed(in)                  — unchanged; ObjectiveContributor (registered alongside
//     the four built-ins) folds a carryover line naming the pin's id+digest into the seed
//     text, so the fresh session's recap is self-describing.
//  3. CarryObjective(before, in)     — AFTER the fresh session is seeded, reconcile the PRIOR
//     pin against the pin re-derived from the same Input the fresh session now holds. This
//     is the runtime check: it calls ctxplan.ReconcileObjective under the hood and returns a
//     typed ObjectiveOutcome the host MUST branch on (surface a refusal/query on anything but
//     Preserved/Established) rather than silently trusting that carryover worked.
//
// This mirrors the model-call task distiller's opt-in shape (model_distill.go): the
// deterministic extraction (taskDistill) always fires; the objective-pin identity/reconcile
// layer is an ADDITIVE contract a host opts into by calling PinObjective/CarryObjective, so a
// caller that never touches this file sees no behavior change.
//
// TIER. This file is the one place in sessionreset (tier 2) that imports ctxplan (tier 1,
// foundation) — a valid downward dependency (mechanism importing a foundation leaf), and it
// does not change sessionreset's own tier: ctxplan itself imports nothing internal.

// ObjectivePinOrder places the objective-pin carryover line right after task_distill (20) —
// the identity/digest line reads as the machine-checkable companion to the human-readable
// "Where we are" recap, before the verbatim tail.
const ObjectivePinOrder = 22

// PinObjective derives a FRESH objective pin (a new PinID) from in — the standing objective
// sessionreset already extracts via firstUserLine, wrapped as a ctxplan.ObjectivePin. Use
// this to mint the FIRST pin of a session; use RepinObjective to derive a SUCCESSOR pin that
// preserves a prior pin's identity across a reset. pinID is caller-supplied (a session/run id
// the host already mints elsewhere) so PinObjective never invents its own id scheme.
func PinObjective(pinID string, in Input) ctxplan.ObjectivePin {
	text := firstUserLine(in.Messages)
	return ctxplan.NewObjectivePin(pinID, text, 0)
}

// RepinObjective derives the objective pin for the FRESH session after a reset, preserving
// prior.PinID so identity carries forward by construction — the caller need not separately
// track "what was the id last time." The Text is re-derived from in the same way
// PinObjective does (firstUserLine), so an honest carryover (the objective did not change)
// naturally reconciles as ctxplan.ObjectivePreserved, while a host bug that re-derives a
// DIFFERENT text under the same id is caught by CarryObjective as ctxplan.ObjectiveDrifted
// rather than silently accepted.
//
// If prior is the zero pin (no prior objective — e.g. this is the very first session),
// RepinObjective mints under pinID exactly like PinObjective; CarryObjective will then
// correctly report ctxplan.ObjectiveEstablished, not a carryover.
func RepinObjective(prior ctxplan.ObjectivePin, pinID string, in Input) ctxplan.ObjectivePin {
	id := pinID
	if !prior.IsZero() {
		id = prior.PinID
	}
	text := firstUserLine(in.Messages)
	return ctxplan.NewObjectivePin(id, text, 0)
}

// CarryObjective is the runtime CHECK a host runs across a reset boundary: given the PRIOR
// session's pin and the Input the FRESH session was (or will be) seeded from, it re-derives
// the fresh pin (preserving prior's identity via RepinObjective) and reconciles the two
// through ctxplan.ReconcileObjective. The returned ctxplan.ObjectiveDecision is the visible,
// typed outcome #1583 requires: a host MUST treat any decision whose Outcome.Refusal() is
// true (Dropped, Drifted, QueryUser) as a case to surface to the user, never as a silent
// pass-through. The re-derived pin is also returned so the host can persist it as the new
// "prior" for the NEXT reset, chaining the identity forward indefinitely.
func CarryObjective(prior ctxplan.ObjectivePin, pinID string, in Input) (ctxplan.ObjectivePin, ctxplan.ObjectiveDecision) {
	after := RepinObjective(prior, pinID, in)
	return after, ctxplan.ReconcileObjective(prior, after)
}

// objectiveContributor folds the pin's identity+digest into the carryover seed as a
// machine-checkable line, so the fresh session's recap names exactly which objective (by id
// and content digest) it claims continuity with — a human-readable complement to
// CarryObjective's programmatic check, not a substitute for it (this contributor never
// reconciles anything; it only renders what PinObjective/RepinObjective already computed).
type objectiveContributor struct {
	// pin is the pin to render. A zero pin (the common case when a host has not opted into
	// PinObjective) makes Contribute decline, so importing this file changes no behavior
	// until a host actually registers a live objectiveContributor via
	// RegisterObjectivePin.
	pin ctxplan.ObjectivePin
}

func (objectiveContributor) Name() string { return "objective_pin" }

func (c objectiveContributor) Contribute(Input) (Part, bool) {
	if c.pin.IsZero() {
		return Part{Name: "objective_pin", Order: ObjectivePinOrder,
			Meta: map[string]string{"skipped": "no_pin"}}, false
	}
	var b strings.Builder
	b.WriteString("Objective pin (must survive verbatim across resets):\n")
	b.WriteString("- id: ")
	b.WriteString(c.pin.PinID)
	b.WriteString("\n- digest: ")
	b.WriteString(shortDigest(c.pin.Digest))
	if strings.TrimSpace(c.pin.Text) != "" {
		b.WriteString("\n- objective: ")
		b.WriteString(clip(c.pin.Text, 280))
	}
	return Part{
		Name:  "objective_pin",
		Order: ObjectivePinOrder,
		Text:  b.String(),
		Meta: map[string]string{
			"pin_id": c.pin.PinID,
			"digest": c.pin.Digest,
		},
	}, true
}

// NewObjectivePinContributor builds the carryover contributor for pin WITHOUT registering
// it — for a host that folds its own registry, or a test that wants it in isolation. A zero
// pin yields a contributor that always declines, mirroring NewModelDistiller's nil-seam
// safety.
func NewObjectivePinContributor(pin ctxplan.ObjectivePin) Contributor {
	return objectiveContributor{pin: pin}
}

// RegisterObjectivePin registers the carryover contributor for pin and returns it (for
// inspection/tests). Like RegisterModelDistiller, this is the on-switch: the objective-pin
// line is absent from the default fold (the four built-ins never mention it) until a host
// calls this with a real pin. Each reset that mints a new pin (via PinObjective/
// RepinObjective) should re-register (or re-build the registry) so BuildSeed renders the
// CURRENT pin, not a stale one from a prior reset — Register does not replace an existing
// entry, so a host that resets more than once should build a scoped Seed some other way,
// or filter Registered() in the assembling layer, rather than relying on this function to
// deduplicate for it.
func RegisterObjectivePin(pin ctxplan.ObjectivePin) Contributor {
	c := NewObjectivePinContributor(pin)
	Register(c)
	return c
}

// shortDigest renders a digest's first 12 hex chars for a human-readable line — full-length
// digests are available via Meta["digest"] for a caller that needs exact comparison.
func shortDigest(d string) string {
	if len(d) <= 12 {
		if d == "" {
			return "(none)"
		}
		return d
	}
	return d[:12]
}

// ObjectiveCarryoverReport is the operator-readable summary of one CarryObjective call —
// the sessionreset-side EXPLAIN companion to ctxplan.ObjectiveDecision, in the same spirit
// as warmPrefix's descriptor: a compact, loggable line a reset pipeline can attach to its
// own audit trail without the caller re-deriving the wording.
type ObjectiveCarryoverReport struct {
	PinID   string `json:"pin_id,omitempty"`
	Outcome string `json:"outcome"`
	Refusal bool   `json:"refusal"`
	Reason  string `json:"reason"`
}

// ReportObjectiveCarryover renders d as an ObjectiveCarryoverReport.
func ReportObjectiveCarryover(d ctxplan.ObjectiveDecision) ObjectiveCarryoverReport {
	return ObjectiveCarryoverReport{
		PinID:   d.PinID,
		Outcome: d.Outcome.String(),
		Refusal: d.Outcome.Refusal(),
		Reason:  d.Reason,
	}
}

// String renders a one-line operator summary, e.g.
// "objective pin obj-1: preserved (pin identity and content are byte-identical...)".
func (r ObjectiveCarryoverReport) String() string {
	return "objective pin " + r.PinID + ": " + r.Outcome + " (" + r.Reason + ")"
}
