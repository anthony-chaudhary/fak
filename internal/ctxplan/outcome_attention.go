package ctxplan

// outcome_attention.go — issue #858, rung 7 of the attention-witness epic (#851): give
// ctxplan.Outcome a PRODUCTION producer.
//
// Outcome (learn.go) is the witnessed feedback the planner closes its loop over (Forecast.Learn /
// Weights.Learn). Until now it had NO production producer — it was populated only in the bench
// (lexical overlap, cmd/ctxplanbench) and in tests, so the planner's learning loop never saw a real
// outcome. This file produces a WITNESSED Outcome from the per-turn attention attribution (rung 2,
// #853): a resident span attended above a threshold is a real Hit, a resident span witnessed at ~0
// mass is real Wasted, and an elided span demand-paged back is a Fault. That turns the learner from
// training on lexical-overlap guesses into training on witnessed attention — the loop the whole
// ctxplan substrate was built for, finally closed with ground truth.
//
// Default-off / shadow by construction: these are PURE functions a driver calls OFF the planning
// path. Producing an Outcome — and even folding it through the learners (LearnFromAttention) —
// changes no live plan until a driver adopts the revised Forecast/Weights. That is the #858 two-
// posture honesty pattern: record the witnessed outcome first, change a planning decision only once
// the shadow signal is trusted.

// DefaultHitThreshold is the attention-mass floor (in the normalized [0,1] attribution) at or above
// which a resident span counts as a witnessed Hit; below it, a witnessed span is Wasted (resident
// but the turn did not actually attend to it). Coarse on purpose — the point is to separate "the
// turn drew on this" from "this idled in the window", not to tune a controller.
const DefaultHitThreshold = 0.01

// OutcomeFromAttention produces a WITNESSED Outcome for one turn from the per-span attention
// Attribution (rung 2) over the resident plan, plus the demand-page faults the pager observed.
//
//	Hits   — resident (Selected) spans witnessed at attention mass >= hitThreshold (the turn drew on
//	         them). A PINNED selected span is always a Hit: a pin is needed by construction (the turn
//	         cannot proceed without it), the same rule SignalNoiseFromAttention applies.
//	Wasted — resident spans witnessed at mass < hitThreshold (resident but the turn never attended).
//	Faults — elided spans the turn demand-paged back. Attention covers RESIDENT spans only, so a
//	         fault is an EXTERNAL pager signal (exactly as the boolean path reads it); only an id that
//	         names an elided span counts, so a stray/fabricated id cannot invent a fault.
//
// A resident span the Attribution does not name is UNACCOUNTED — it teaches nothing (skipped),
// fail-closed: no witness means no label, never a fabricated Hit or Wasted. Deterministic: Hits and
// Wasted follow Selected order; Faults follow the faults argument's order filtered to elided ids.
func OutcomeFromAttention(p Plan, attribution Attribution, faults []string, hitThreshold float64) Outcome {
	var o Outcome
	for _, sel := range p.Selected {
		if sel.Pinned {
			o.Hits = append(o.Hits, sel.ID) // needed by construction — a pin is signal
			continue
		}
		mass, witnessed := attribution[sel.ID]
		if !witnessed {
			continue // no witness for this resident span: teach nothing (fail-closed)
		}
		if clampMass(mass) >= hitThreshold {
			o.Hits = append(o.Hits, sel.ID)
		} else {
			o.Wasted = append(o.Wasted, sel.ID)
		}
	}
	if len(faults) > 0 && len(p.Elided) > 0 {
		elided := make(map[string]bool, len(p.Elided))
		for _, el := range p.Elided {
			elided[el.ID] = true
		}
		for _, id := range faults {
			if elided[id] {
				o.Faults = append(o.Faults, id)
			}
		}
	}
	return o
}

// LearnFromAttention closes the planner learning loop on a WITNESSED outcome: it produces the Outcome
// from the per-turn attention attribution (OutcomeFromAttention) and applies BOTH online learners —
// Forecast.Learn (promote faulted spans into the predicted intents) and Weights.Learn (tune the cost
// knobs from hits/faults/wasted) — returning the revised forecast and weights for the next turn.
//
// Shadow by construction: a pure function off the planning path. Until a driver adopts the returned
// Forecast/Weights, planning is byte-identical to before — the #858 default-off posture. spans is the
// candidate set the plan was built from (the learners resolve the Outcome's ids to full spans, and
// fail-closed on an id that names no/sealed/tombstoned span).
func LearnFromAttention(f Forecast, w Weights, p Plan, spans []Span, attribution Attribution, faults []string, hitThreshold float64, maxStep int) (Forecast, Weights) {
	o := OutcomeFromAttention(p, attribution, faults, hitThreshold)
	return f.Learn(o, spans), w.Learn(o, spans, f, maxStep)
}
