// Rung F3 (issue #1886): the load-bearing vs ephemeral classifier that BOUNDS the F2
// externalize gate. F1 (#1884, externalized.go) DETECTS transcript-only state and F2 (#1885,
// externalize_gate.go) REFUSES to rotate while any exists — but not all transcript content is
// load-bearing. Turn-scoped, ephemeral state ("the clock reads 3pm", "currently building")
// would be dropped on rotation WITHOUT loss; a gate that refused on it too would over-refuse,
// and a false-positive-prone gate gets routed around (the issue's failure mode). This rung
// classifies each candidate piece of state as load-bearing vs ephemeral and keeps only the
// load-bearing subset for the gate, so the gate fires on the former alone.
//
// It reuses the durability vocabulary already shared across the context stack — ctxmmu's
// write-time {turn,session,bounded,durable} classifier (CONTEXT-IS-NOT-MEMORY.md) and
// ctxplan's mirror of the same constants (ctxplan.Durability*, which relay already imports) —
// instead of inventing a new vocabulary or an ML model. The text->class step (ctxmmu's
// tense/deixis prior, ClassifyText) lives up-tier where the model-shaped inputs are; relay
// stays tier-1 and consumes the resulting class only. A caller runs ctxmmu.ClassifyText (or
// sets the class explicitly) and hands the classified candidates here. That layering is why
// this reuses the IDEA (the shared vocabulary) without an upward import of the tier-2
// classifier — an upward import would red the architest tier gate.
//
// Fail-closed direction is the OPPOSITE of ctxmmu's promotion gate. There, the expensive
// error is promoting an ephemeral fact into durable memory, so an unclassified fact defaults
// to turn (ephemeral). HERE the expensive error is silently DROPPING a load-bearing fact on
// rotation, so an unclassified or unknown class defaults to LOAD-BEARING: only an EXPLICIT
// turn-scoped class exempts state from the gate. That keeps the gate conservative — it never
// waves through unknown state — while still bounding it away from provably-ephemeral state.
package relay

import "github.com/anthony-chaudhary/fak/internal/ctxplan"

// Candidate is one piece of state the current leg touched, paired with the durability class of
// the underlying claim (the ctxmmu/ctxplan {turn,session,bounded,durable} vocabulary; an empty
// or unknown class is treated as load-bearing, never ephemeral — see IsEphemeral). It embeds
// the LoadBearingFact so a classified-in candidate projects straight to the fact the F1/F2 gate
// already consumes.
type Candidate struct {
	LoadBearingFact
	Durability string `json:"durability,omitempty"`
}

// IsEphemeral reports whether a durability class marks state as turn-scoped — true ONLY for an
// explicit ctxplan.DurabilityTurn. Every other value, including "" and any unknown class, is
// load-bearing: the relay gate fails closed toward NOT dropping state, so it treats
// unclassified state as load-bearing rather than waving it through. This deliberately does NOT
// route through ctxplan.NormDurability, whose fail-closed default is turn for the opposite,
// promotion-side posture.
func IsEphemeral(class string) bool {
	return class == ctxplan.DurabilityTurn
}

// Classify reports a candidate's durability class and whether it is load-bearing — the F3
// verdict. Pure and deterministic: it reads no clock and does no I/O, and the same class
// always yields the same verdict. loadBearing is exactly !IsEphemeral(class).
func Classify(c Candidate) (class string, loadBearing bool) {
	return c.Durability, !IsEphemeral(c.Durability)
}

// LoadBearing returns the load-bearing subset of candidates, projected to LoadBearingFact, in
// input order — the only state a rotation must not silently drop. Ephemeral/turn-scoped
// candidates are filtered out so the F2 gate never fires on them (bounding its false-positive
// surface). A nil result means every candidate was ephemeral: nothing for the gate to hold.
func LoadBearing(candidates []Candidate) []LoadBearingFact {
	var out []LoadBearingFact
	for _, c := range candidates {
		if _, lb := Classify(c); lb {
			out = append(out, c.LoadBearingFact)
		}
	}
	return out
}

// CheckExternalizeGateClassified is the bounded externalize gate: it classifies candidate state
// first, drops the ephemeral/turn-scoped part, then applies the F2 gate to only the
// load-bearing subset. An ephemeral transcript-only candidate therefore never trips the gate; a
// load-bearing transcript-only candidate still does, with the same RELAY_NOT_EXTERNALIZED
// refusal and culprits F2 emits. This is the entry point the safe point should call once a
// leg's state carries durability classes.
func CheckExternalizeGateClassified(candidates []Candidate) ExternalizeGate {
	return CheckExternalizeGate(LoadBearing(candidates))
}
