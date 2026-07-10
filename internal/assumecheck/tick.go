package assumecheck

// tick.go — the PURE re-witness classifier (#3823, epic #3818 C5): one loop tick
// of the live assumption guard as a pure function. `fak assume check` is a
// one-shot, so a registered premise silently rots between runs; the loop shell
// (cmd/fak/assume_loop.go) re-witnesses every registered assumption on an
// interval and hands the fresh verdicts HERE, and Tick decides — with no I/O, no
// clock, no bus — which transitions deserve a correction event.
//
// This file mirrors internal/doomloop's Classify split exactly: the shell
// samples (gathers evidence through the C3 driver registry and adjudicates
// through the C1 kernel), the pure classifier folds (prev, now) into ledger rows
// plus events, and only the shell's deliver seam ever touches the adjudicated
// steer bus. By construction the classifier cannot queue or deliver — it only
// returns values.
//
// The correction ladder stays SOFT and REVERSIBLE, exactly like doomloop's
// (nudge -> operator escalation, no destructive rung): the only event kind Tick
// emits is a re-anchor payload for the steer channel, and the shell's terminal
// rung on a delivery the bus refuses is an operator-escalation RECORD, never a
// teardown.

import "fmt"

// TickRow is one accountability-ledger row: the fresh verdict for one registered
// assumption, the outcome it carried into the tick, and whether this tick's edge
// emitted an event. One row is recorded per registered assumption per tick —
// calm ticks are logged, not just eventful ones — so the ledger witnesses that
// the guard actually re-checked, not only that something broke.
type TickRow struct {
	Verdict Verdict `json:"verdict"`
	// PrevOutcome is the effective prior the transition was judged against: the
	// outcome recorded for this assumption on the previous tick, or the seeded
	// OutcomeHolds default for a first observation (see Tick).
	PrevOutcome Outcome `json:"prev_outcome"`
	// Transition reports whether this row is the HOLDS->VIOLATED edge that
	// emitted a TickEvent.
	Transition bool `json:"transition"`
}

// TickEvent is one queued correction: a registered assumption whose premise went
// bad THIS tick (prior HOLDS, current VIOLATED). It is pure data — the shell
// stamps the clock, addresses it, and queues/delivers it; the event itself
// carries the identity, the closed outcome-class refusal token, and the soft
// re-anchor payload.
type TickEvent struct {
	AssumptionID string `json:"assumption_id"`
	// RefusalReason is the OUTCOME-CLASS token from the closed DOS refusal
	// vocabulary (Outcome.RefusalReason, #3822 C4) — ASSUMPTION_VIOLATED for the
	// only edge that emits — never the per-assumption label (that travels folded
	// into Reason).
	RefusalReason string `json:"refusal_reason"`
	// Reason is the adjudicated verdict's reason text: what the witness saw.
	Reason string `json:"reason"`
	// Reanchor is the soft, reversible correction payload for the steer channel
	// (ReanchorMessage). Injecting text is the ladder's strongest automatic rung.
	Reanchor string `json:"reanchor"`
	// Reversible is always true: the ladder has no destructive rung by
	// construction, and the field says so on every queued artifact.
	Reversible bool `json:"reversible"`
}

// TickResult is what one pure tick returns: the ledger rows (one per verdict
// handed in), the events to emit (empty when nothing transitioned), and the
// outcome map the NEXT tick should judge against.
type TickResult struct {
	Rows   []TickRow   `json:"rows"`
	Events []TickEvent `json:"events"`
	// Next is prev carried forward with this tick's outcomes folded in — the
	// exact map to hand back as prev on the next tick, so the loop shell keeps
	// no transition logic of its own.
	Next map[string]Outcome `json:"next"`
}

// Tick folds the previous tick's outcomes and this tick's fresh verdicts (one
// per registered assumption, from the SAME gather+adjudicate path `fak assume
// check` uses) into ledger rows and correction events. Pure: no I/O, no clock,
// no hidden state — the same (prev, now) always reproduces the same result.
//
// The event rule: an event is emitted for an assumption IFF its effective prior
// outcome is OutcomeHolds AND its current outcome is OutcomeViolated — the
// HOLDS->VIOLATED edge, the moment a premise the fleet was relying on went bad.
// An assumption absent from prev (or carrying a corrupt/foreign value) is seeded
// as HOLDS, so the FIRST witnessed violation of a newly watched premise still
// emits — conservative in the direction of speaking up, never of suppressing.
// Everything else is row-only: a still-violated premise does not re-emit every
// tick (the edge already fired), a violated premise going back to HOLDS is
// recovery (the reversible ladder's whole point), and HOLDS->UNVERIFIABLE/STALE
// is a witnessing gap the ledger records without steering anyone.
//
// Verdicts are processed in order against a working copy of prev, so a
// duplicate id in now judges its second appearance against the first's outcome
// — one event per premise per edge, never two for one bad turn.
func Tick(prev map[string]Outcome, now []Verdict) TickResult {
	next := make(map[string]Outcome, len(prev)+len(now))
	for id, o := range prev {
		next[id] = o
	}

	res := TickResult{Rows: make([]TickRow, 0, len(now))}
	for _, v := range now {
		prior, ok := next[v.AssumptionID]
		if !ok || !ValidOutcome(prior) {
			prior = OutcomeHolds
		}
		transition := prior == OutcomeHolds && v.Outcome == OutcomeViolated
		res.Rows = append(res.Rows, TickRow{Verdict: v, PrevOutcome: prior, Transition: transition})
		if transition {
			res.Events = append(res.Events, TickEvent{
				AssumptionID:  v.AssumptionID,
				RefusalReason: v.Outcome.RefusalReason(),
				Reason:        v.Reason,
				Reanchor:      ReanchorMessage(v.AssumptionID, v.Reason),
				Reversible:    true,
			})
		}
		next[v.AssumptionID] = v.Outcome
	}
	res.Next = next
	return res
}

// ReanchorMessage is the soft re-anchor payload a HOLDS->VIOLATED event carries
// onto the steer channel — the assumecheck twin of cmd/fak's reanchorMessage
// (doomloop.go): stop relying on the broken premise, re-witness it, re-anchor,
// and prefer a structured refusal over quiet drift. Text only; nothing
// destructive rides on it.
func ReanchorMessage(assumptionID, reason string) string {
	return fmt.Sprintf("assume-guard: registered assumption %q went HOLDS -> VIOLATED: %s. "+
		"Stop relying on that premise. Re-witness it (`fak assume check %s`), re-anchor your plan on the refreshed verdict, "+
		"and if it blocks you, refuse with the structured reason instead of proceeding on a dead assumption.",
		assumptionID, reason, assumptionID)
}
