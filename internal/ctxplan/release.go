package ctxplan

// release.go is the AGENT-DECLARED RELEASE — the free() dual of the pin (#2225; the
// reachability-layer epic #844). A pin is the model declaring "I still reference this
// span"; a release is the model declaring "my ongoing work no longer references this
// span". The pin edge shipped with the facade (PlanQuery.Pins); this file adds the
// opposite edge and the accountability around it.
//
// What makes it SOUND to hand the model this verb is elision recoverability: a released
// span is elided with its recovery handle, exactly like an over-budget elision — one
// demand-page away, never destroyed (Audit's partition and recoverability checks apply
// unchanged). A WRONG release therefore costs one page fault, never a lost fact — the
// same property that makes a bad forecast safe (forecast.go) makes a bad release safe.
//
// Three fences keep the declaration honest:
//
//	pin outranks release — a span both pinned and released stays RESIDENT (the epic's
//	                       over-retain bias: a false-retain costs tokens, a false-free
//	                       costs context). The conflict is REPORTED (PinHeld), never
//	                       silently resolved either way. Structural roots (system, goal,
//	                       last user turn) are pinned, so they are un-releasable by
//	                       construction.
//
//	release is not tombstone — a released span is NOT suppressed. Demand-paging it back
//	                       in is legitimate (and is exactly the recant signal below);
//	                       Tombstoned stays the context-CONTROL verdict, released the
//	                       context-ECONOMY one.
//
//	recant witness       — a released span that later appears in Outcome.Faults proves
//	                       the declaration wrong: the turn needed it after all. It is
//	                       named RECANTED (the fault already paged it back in; nothing
//	                       else to repair) and DropRecantedReleases clears it so a
//	                       carried-forward forecast does not re-elide it into a fault
//	                       loop. A release that never faults is VINDICATED — budget
//	                       genuinely freed. This mirrors refcount.go's posture: a
//	                       self-report is checked against the witnessed outcome, never
//	                       just believed.

// ReleaseReport is what the model's release declarations actually DID — carried on
// PlanView so the declaring agent sees the disposition of every id it released, in the
// same pass that shows it the plan. The four classes are closed and disjoint; every
// distinct released id lands in exactly one.
type ReleaseReport struct {
	// Honored are released spans the planner elided as ElideReleased — cold, recoverable,
	// their budget freed for spans the work still needs.
	Honored []string `json:"honored,omitempty"`
	// PinHeld are released spans a pin kept RESIDENT — pin outranks release, so the
	// release lost. Surfaced, not silently dropped: the model learns its declaration
	// conflicts with a root (its own, or a structural one).
	PinHeld []string `json:"pin_held,omitempty"`
	// Gated are released spans a stronger lane had already excluded (sealed by the trust
	// gate, tombstoned by context control). The release changed nothing.
	Gated []string `json:"gated,omitempty"`
	// Unknown are released ids that named no candidate — an advisory no-op (a stale or
	// fabricated id cannot poison the plan, the same fail-closed posture Outcome ids get).
	Unknown []string `json:"unknown,omitempty"`
}

// buildReleaseReport classifies each distinct released id against the plan that honored
// (or refused) it. Pure and deterministic: ids are deduped and sorted, and the verdict is
// read off the plan's own Selected/Elided accounting, never recomputed.
func buildReleaseReport(p Plan, releases []string) ReleaseReport {
	elidedReason := make(map[string]string, len(p.Elided))
	for _, e := range p.Elided {
		if _, ok := elidedReason[e.ID]; !ok {
			elidedReason[e.ID] = e.Reason
		}
	}
	resident := make(map[string]bool, len(p.Selected))
	for _, s := range p.Selected {
		resident[s.ID] = true
	}

	var out ReleaseReport
	for _, id := range dedupSorted(releases) {
		switch {
		case elidedReason[id] == ElideReleased:
			out.Honored = append(out.Honored, id)
		case resident[id]:
			// The release lane elides every released un-pinned candidate up front, so the
			// only way a released id is resident is that a pin won the conflict.
			out.PinHeld = append(out.PinHeld, id)
		case elidedReason[id] == ElideSealed || elidedReason[id] == ElideTombstoned:
			out.Gated = append(out.Gated, id)
		default:
			out.Unknown = append(out.Unknown, id)
		}
	}
	return out
}

// ReleaseOutcome is the recant witness: each released id checked against the turn's
// witnessed Outcome. Measurement only — by the time it is computed, a recanted span has
// already been demand-paged back in (that fault IS the evidence), so there is nothing to
// roll back; the verdict exists so the declaration loop stays honest and the caller stops
// re-declaring a release reality refuted.
type ReleaseOutcome struct {
	// Vindicated are released spans the turn never needed back — the declaration held and
	// the budget was genuinely freed.
	Vindicated []string `json:"vindicated,omitempty"`
	// Recanted are released spans that appeared in Outcome.Faults — the model's "no longer
	// useful" was wrong; the work referenced the span after all. Cost: one page fault.
	Recanted []string `json:"recanted,omitempty"`
}

// ClassifyReleases folds the witnessed Outcome over a release set (typically the plan
// report's Honored list — a PinHeld/Gated/Unknown id was never elided BY the release, so
// a fault on it proves nothing about the declaration). Deterministic: ids deduped and
// sorted, verdicts read only from Outcome.Faults membership.
func ClassifyReleases(released []string, o Outcome) ReleaseOutcome {
	faulted := make(map[string]bool, len(o.Faults))
	for _, id := range o.Faults {
		faulted[id] = true
	}
	var out ReleaseOutcome
	for _, id := range dedupSorted(released) {
		if faulted[id] {
			out.Recanted = append(out.Recanted, id)
		} else {
			out.Vindicated = append(out.Vindicated, id)
		}
	}
	return out
}

// DropRecantedReleases returns a Forecast whose Releases no longer contain any id the
// witnessed Outcome faulted — the recant applied. A session loop that carries its
// forecast forward calls this beside Learn: Learn promotes the faulted span's tokens into
// the intents (predict it next turn), and this drops the refuted release (stop re-eliding
// it), so one wrong declaration converges to one fault instead of one fault per turn.
// With no overlap the forecast is returned unchanged — a deterministic no-op, exactly
// like Learn with nothing to promote. Order of surviving releases is preserved.
func (f Forecast) DropRecantedReleases(o Outcome) Forecast {
	if len(f.Releases) == 0 || len(o.Faults) == 0 {
		return f
	}
	faulted := make(map[string]bool, len(o.Faults))
	for _, id := range o.Faults {
		faulted[id] = true
	}
	kept := make([]string, 0, len(f.Releases))
	for _, id := range f.Releases {
		if !faulted[id] {
			kept = append(kept, id)
		}
	}
	if len(kept) == len(f.Releases) {
		return f
	}
	out := f
	out.Releases = kept
	return out
}
