package ctxplan

import (
	"context"
	"errors"
)

// Candidates scores a set of spans against a forecast into planner candidates — the pure,
// I/O-free step (no bytes are paged in; it reads only SAFE span metadata). It computes each
// span's resident token Cost (via cost, or TokenCost if nil) and its Benefit (via the
// forecast), normalizing recency against the largest step in the set.
func Candidates(spans []Span, f Forecast, cost CostModel) []Candidate {
	if cost == nil {
		cost = TokenCost
	}
	maxStep := 0
	for _, s := range spans {
		if s.Step > maxStep {
			maxStep = s.Step
		}
	}
	out := make([]Candidate, 0, len(spans))
	for _, s := range spans {
		out = append(out, Candidate{
			Cell:    s,
			Cost:    cost(s),
			Benefit: f.Benefit(s, maxStep),
		})
	}
	return out
}

// PlanCells is the pure planning entry point: score the spans against the forecast and
// optimize the resident view under the budget, greedily. It performs NO I/O — it is the
// deterministic core a caller can run, EXPLAIN, and audit before materializing anything.
// (It is named PlanCells, not Plan, because Plan is the result type.)
func PlanCells(spans []Span, f Forecast, budget Budget, cost CostModel) Plan {
	cands := Candidates(spans, f, cost)
	p := Optimize(cands, budget, pinSet(f.Pins), ObjGreedy)
	p.Horizon = f.Horizon
	return p
}

// pinSet turns the forecast's pin id list into a lookup set.
func pinSet(pins []string) map[string]bool {
	if len(pins) == 0 {
		return nil
	}
	s := make(map[string]bool, len(pins))
	for _, id := range pins {
		s[id] = true
	}
	return s
}

// View is the materialized O(1) turn: the plan, the bytes actually rendered into the
// fresh history (through the trust gate), any spans the gate refused on page-in, and the
// witness reconciled against that rendered+refused outcome. Rendered is the resident view
// in step order — the history the next turn continues from. The Witness attests both that
// the plan was faithful (no candidate destroyed) AND that the page-in accounted for every
// selected span with bytes matching the Span.Bytes the planner charged.
type View struct {
	Plan     Plan       `json:"plan"`
	Rendered []Rendered `json:"rendered"`
	Refused  []Refusal  `json:"refused,omitempty"`
	Witness  Witness    `json:"witness"`
}

// Materialize is the full pass: scan the store, plan the O(1) view against the forecast
// under the budget, then RENDER the selected spans' bytes into the fresh history — routing
// every page-in through the store's trust gate (Store.Materialize), in step order. A
// selected span the gate refuses (it was concurrently sealed, or its bytes went missing)
// is reported in Refused, never rendered — the poison-never-enters-context invariant. The
// returned View's Witness proves the plan was faithful (every elided span recoverable).
//
// This is the replacement for "compact the transcript": instead of summarizing the
// history into the context, the planner re-derives an O(1) rendering of the lossless
// store each turn, and the elided remainder stays one demand-page away.
func Materialize(ctx context.Context, store Store, f Forecast, budget Budget, cost CostModel) (View, error) {
	spans, err := store.Spans(ctx)
	if err != nil {
		return View{}, err
	}
	p := PlanCells(spans, f, budget, cost)
	// declared[id] = the Span.Bytes the planner priced each span at — the cost basis the
	// page-in must honor. The planner charged ceil(Span.Bytes/4); the render realizes
	// ceil(len(body)/4); the witness pins them equal so the budget is honest.
	declared := make(map[string]int64, len(spans))
	for _, s := range spans {
		declared[s.ID] = s.Bytes
	}
	v := View{Plan: p}
	for _, s := range p.Selected {
		body, err := store.Materialize(ctx, s.ID)
		if err != nil {
			reason := "page_in_refused"
			switch {
			case errors.Is(err, ErrSealed):
				reason = "sealed_by_trust_gate"
			case errors.Is(err, ErrTombstoned):
				reason = "tombstoned_by_context_control"
			}
			v.Refused = append(v.Refused, Refusal{ID: s.ID, Step: s.Step, Role: s.Role, Reason: reason})
			continue
		}
		v.Rendered = append(v.Rendered, Rendered{
			ID: s.ID, Step: s.Step, Role: s.Role, Descriptor: s.Descriptor,
			Bytes: int64(len(body)), Tokens: tokenEstimate(len(body)),
		})
	}
	// The witness is reconciled with the rendered+refused outcome (not just the plan), so it
	// attests that every selected span was accounted for at page-in and that the bytes the
	// gate handed back match the Span.Bytes the planner charged.
	v.Witness = Reconcile(p, Audit(p), v.Rendered, v.Refused, declared)
	return v, nil
}

// RenderedTokens is the total token cost actually paged into the fresh history — the
// realized resident size, which is <= the budget by construction.
func (v View) RenderedTokens() int {
	n := 0
	for _, r := range v.Rendered {
		n += r.Tokens
	}
	return n
}

// tokenEstimate is the 4-bytes/token proxy ((n+3)/4) — the same one the planner's
// TokenCost uses, so resident-budget units and rendered-token units agree.
func tokenEstimate(n int) int {
	if n <= 0 {
		return 0
	}
	return (n + 3) / 4
}
