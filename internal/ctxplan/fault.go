package ctxplan

import (
	"context"
	"errors"
	"sort"
)

// Fault is the page-fault record: a mid-turn reference to a span the resident view had
// elided (a forecast MISS), and the handler's disposition of it. It is the feedback
// token a caller feeds back to re-plan — DemandPage returns one so the loop can hand it
// to Forecast.Learn (promoting the missed content into the next forecast's intents).
//
// The headline honesty claim this type upholds is "a forecast MISS costs one demand-page,
// never a lost fact": a Served fault spliced the span back into the resident view; a
// Refused fault means the trust gate held (the span is sealed/tombstoned) and the
// lossless store still keeps it — the caller routes around it, the fact is not lost.
type Fault struct {
	ID     string `json:"id"`
	Step   int    `json:"step"`
	Role   string `json:"role,omitempty"`
	Status string `json:"status"`            // served | resident | refused | absent
	Reason string `json:"reason,omitempty"`  // set on refused (the gate's reason)
	Tokens int    `json:"tokens,omitempty"`  // tokens paged in (served only)
}

// Fault dispositions — the closed vocabulary a DemandPage outcome may carry.
const (
	// FaultServed: the span was elided (a MISS); the handler paged it back in through the
	// trust gate and spliced it into the resident View. The cost is the span's token size —
	// the bounded, cheap fault the planned regime pays instead of losing the fact.
	FaultServed = "served"
	// FaultResident: the span was ALREADY resident. DemandPage is idempotent, so a caller
	// may demand-page a list of candidate spans defensively without double-charging.
	FaultResident = "resident"
	// FaultRefused: the trust gate declined the page-in (the span is sealed or
	// tombstoned). The View is unchanged; the lossless store still keeps the bytes for
	// audit, but poison/suppression never enters context, even via the recovery path.
	FaultRefused = "refused"
	// FaultAbsent: the id names no span in the store. Not an error — a caller may
	// demand-page a speculative id list and let the absent ones no-op.
	FaultAbsent = "absent"
)

// DemandPage is the page-fault handler — the rung that makes "a forecast MISS costs one
// demand-page, never a lost fact" a real operation instead of a slogan. When the current
// turn references content the resident View elided, DemandPage pages that span back in
// through the store's trust gate, splices the recovered bytes into the resident View (in
// step order), promotes it in the Plan (Elided -> Selected so the witness partition stays
// honest), re-reconciles the witness, and returns the Fault for re-plan feedback.
//
// It is IDEMPOTENT: a span already resident is returned unchanged (FaultResident). A
// sealed or tombstoned span is REFUSED — the gate holds, the View is unchanged, and the
// fault carries FaultRefused (poison never enters context via the recovery handle). An id
// naming no span is FaultAbsent (not an error, so a caller may demand-page defensively).
//
// The returned View is a fresh copy; the input View and its Plan are not mutated. A served
// fault may take the resident set OVER the forecast Budget — that is the documented miss
// cost (scaling.go's RetrieveFaults term); the witness stays faithful and reconciled, only
// the budget thrift is spent.
func DemandPage(ctx context.Context, store Store, in View, spanID string) (View, Fault, error) {
	fault := Fault{ID: spanID}
	if spanID == "" {
		fault.Status = FaultAbsent
		return in, fault, nil
	}
	// Idempotent: already resident -> no-op.
	for _, r := range in.Rendered {
		if r.ID == spanID {
			fault.Status = FaultResident
			fault.Step = r.Step
			fault.Role = r.Role
			return in, fault, nil
		}
	}
	// Resolve the span metadata + the cost basis (Span.Bytes) the witness reconciles on.
	spans, err := store.Spans(ctx)
	if err != nil {
		return in, fault, err
	}
	var span Span
	found := false
	declared := make(map[string]int64, len(spans))
	for _, s := range spans {
		declared[s.ID] = s.Bytes
		if s.ID == spanID {
			span = s
			found = true
		}
	}
	if !found {
		fault.Status = FaultAbsent
		return in, fault, nil
	}
	fault.Step = span.Step
	fault.Role = span.Role

	// Page in through the trust gate.
	body, err := store.Materialize(ctx, spanID)
	if err != nil {
		// The gate held (sealed / tombstoned / bytes missing). The View is unchanged —
		// the span stays elided (recoverable by handle for audit) and the lossless store
		// keeps it; the caller routes around a refused fault, the fact is not lost.
		reason := "page_in_refused"
		switch {
		case errors.Is(err, ErrSealed):
			reason = "sealed_by_trust_gate"
		case errors.Is(err, ErrTombstoned):
			reason = "tombstoned_by_context_control"
		}
		fault.Status = FaultRefused
		fault.Reason = reason
		return in, fault, nil
	}

	// Served: build a fresh View (input not mutated), splice into the resident view in
	// step order, promote Elided -> Selected so the witness partition stays honest, and
	// re-reconcile against the rendered outcome.
	out := in
	out.Rendered = append([]Rendered(nil), in.Rendered...)
	out.Refused = append([]Refusal(nil), in.Refused...)
	tokens := tokenEstimate(len(body))
	out.Rendered = append(out.Rendered, Rendered{
		ID: spanID, Step: span.Step, Role: span.Role, Descriptor: span.Descriptor,
		Bytes: int64(len(body)), Tokens: tokens,
	})
	sort.SliceStable(out.Rendered, func(i, j int) bool {
		if out.Rendered[i].Step != out.Rendered[j].Step {
			return out.Rendered[i].Step < out.Rendered[j].Step
		}
		return out.Rendered[i].ID < out.Rendered[j].ID
	})
	out.Plan = promoteResident(in.Plan, span)
	out.Witness = Reconcile(out.Plan, Audit(out.Plan), out.Rendered, out.Refused, declared)

	fault.Status = FaultServed
	fault.Tokens = tokens
	return out, fault, nil
}

// promoteResident returns a Plan with span moved from Elided to Selected (or added to
// Selected if it was in neither set), so the witness's Selected/Elided partition stays
// honest after a demand-page. It allocates FRESH Selected/Elided slices so the input
// Plan's backing arrays are never mutated; Cost/Benefit are carried from the Elision when
// the span was elided (the planner's own accounting), falling back to TokenCost / 0 for a
// span that was in neither set (added to the store after planning).
func promoteResident(p Plan, span Span) Plan {
	out := p

	cost := TokenCost(span)
	benefit := 0.0
	elided := make([]Elision, 0, len(out.Elided))
	for _, e := range out.Elided {
		if e.ID == span.ID {
			cost = e.Cost
			benefit = e.Benefit
			continue // drop from Elided
		}
		elided = append(elided, e)
	}
	out.Elided = elided

	sel := make([]Selection, len(out.Selected))
	copy(sel, out.Selected)
	present := false
	for _, s := range sel {
		if s.ID == span.ID {
			present = true
		}
	}
	if !present {
		sel = append(sel, Selection{
			ID: span.ID, Step: span.Step, Role: span.Role, Descriptor: span.Descriptor,
			Cost: cost, Benefit: benefit,
		})
		sort.SliceStable(sel, func(i, j int) bool {
			if sel[i].Step != sel[j].Step {
				return sel[i].Step < sel[j].Step
			}
			return sel[i].ID < sel[j].ID
		})
	}
	out.Selected = sel
	return out
}
