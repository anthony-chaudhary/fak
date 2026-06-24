package ctxplan

// Witness is the faithfulness audit of a plan — the property that separates a planned
// VIEW from lossy COMPACTION. A plan is Faithful iff (a) its resident and elided sets
// PARTITION every candidate (no span silently vanished, none double-counted) and (b)
// every elided span carries a recovery handle (a content address / id) so it can be
// paged back in on demand. A faithful plan never DESTROYS history; it only decides which
// spans are resident right now. Compaction, by contrast, replaces elided spans with a
// lossy summary and drops the originals — Audit reports that UNFAITHFUL.
//
// This is the honesty gate the whole concept rests on: it is what lets ctxplan claim
// "O(1) resident AND exact recall," because every byte the view leaves out is still
// recoverable. A regression that constructed an Elision with no handle, or that lost a
// candidate from both sets, turns Faithful false — exactly the silent-loss bug it exists
// to catch.
type Witness struct {
	Candidates     int      `json:"candidates"`              // spans the planner considered
	Resident       int      `json:"resident"`                // spans kept in the view
	Elided         int      `json:"elided"`                  // spans kept cold (out of the view)
	Recoverable    int      `json:"recoverable"`             // elided spans WITH a page-back-in handle
	Unrecoverable  []string `json:"unrecoverable,omitempty"` // elided spans with NO handle (destroyed — a compaction tell)
	ResidentTokens int      `json:"resident_tokens"`         // resident token cost: planned selected cost under Audit, REALIZED rendered tokens (== View.RenderedTokens()) after Reconcile
	ElidedTokens   int      `json:"elided_tokens"`           // tokens out of the window but still recoverable
	Partition      bool     `json:"partition"`               // Resident+Elided == Candidates AND the two sets are disjoint
	Faithful       bool     `json:"faithful"`                // Partition AND every elided span is recoverable

	// Materialization reconciliation — set by Reconcile over the rendered+refused outcome of
	// a Materialize pass (zero for a pure-plan Audit, which performs no page-in). Where Audit
	// asks "did the PLAN drop a span?", Reconcile asks "did the PAGE-IN drop a selected span,
	// and did the store hand back bytes that match the Span.Bytes the planner charged?". A View
	// is honest only when the plan is Faithful AND the materialization Reconciled AND the
	// CostContract held — otherwise the budget charged and the bytes rendered disagree.
	Rendered     int      `json:"rendered"`                // spans actually paged into the view
	Refused      int      `json:"refused"`                 // selected spans the gate declined at page-in
	Reconciled   bool     `json:"reconciled"`              // Rendered+Refused == Selected (no selected span silently dropped)
	CostContract bool     `json:"cost_contract"`           // every rendered span's Bytes == its declared Span.Bytes
	CostDiverged []string `json:"cost_diverged,omitempty"` // rendered spans whose bytes != Span.Bytes (the contract break)
}

// Audit computes the faithfulness witness of a plan. It reads only the plan's own
// accounting (no store access), so it is a cheap, deterministic check a caller can run
// before trusting a plan — or a gate can run to REFUSE one that dropped a span with no
// recovery path.
func Audit(p Plan) Witness {
	w := Witness{
		Candidates: p.Candidates,
		Resident:   len(p.Selected),
		Elided:     len(p.Elided),
	}
	// An empty ID anywhere breaks the partition: the disjointness check below is
	// ID-keyed, so a span with no ID could be both resident AND elided undetected. A
	// well-formed plan (and the adapters that build one) must address every span; a blank
	// ID is treated as a partition failure, not waved through.
	wellFormed := true
	selectedIDs := make(map[string]bool, len(p.Selected))
	for _, s := range p.Selected {
		w.ResidentTokens += s.Cost
		if s.ID == "" {
			wellFormed = false
			continue
		}
		selectedIDs[s.ID] = true
	}
	disjoint := true
	for _, e := range p.Elided {
		w.ElidedTokens += e.Cost
		if e.ID == "" {
			wellFormed = false
		}
		if e.ID != "" && selectedIDs[e.ID] {
			disjoint = false // a span both resident AND elided — a double-count, not a partition
		}
		if e.ID == "" && e.Digest == "" {
			// No handle: this span cannot be paged back in. It was DESTROYED, not elided —
			// the defining property of compaction. Record its (best-effort) identity.
			id := e.Reason // fall back to the reason so the report is not blank
			w.Unrecoverable = append(w.Unrecoverable, id)
		} else {
			w.Recoverable++
		}
	}
	w.Partition = wellFormed && disjoint && (w.Resident+w.Elided == w.Candidates)
	w.Faithful = w.Partition && len(w.Unrecoverable) == 0
	return w
}

// CompactionView models what LOSSY COMPACTION does to the same plan: it keeps the same
// resident view but strips the recovery handles off every elided span — i.e. it drops
// the originals and would replace them with a summary. It exists to make the faithful-vs-
// compaction distinction a CHECKABLE contrast, not a slogan: Audit(p) is Faithful while
// Audit(CompactionView(p)) is not, with identical residency. The token savings look the
// same; only recoverability differs, and that is the whole point.
func CompactionView(p Plan) Plan {
	out := p
	out.Objective = "compaction"
	out.Elided = make([]Elision, len(p.Elided))
	for i, e := range p.Elided {
		e.ID = ""     // the original span id is gone
		e.Digest = "" // the bytes are gone — no page-back-in handle
		e.Reason = "compacted_away"
		out.Elided[i] = e
	}
	return out
}

// Reconcile folds a plan's faithfulness witness with its materialization outcome: the
// returned witness keeps the plan's Partition/Faithful verdict and ADDS the rendered+refused
// reconciliation and the Span.Bytes cost contract. It is what Materialize stamps on a View
// so the witness agrees with what actually happened at the page-in boundary, not just with
// what the plan promised before it.
//
// The resident TOKEN accounting is reconciled too: ResidentTokens is reset to the tokens the
// gate actually paged in (sum of Rendered.Tokens), so a span the planner selected but the
// gate refused stops counting as resident. After Reconcile, ResidentTokens == sum of
// Rendered.Tokens == View.RenderedTokens() — the resident size a caller trusts is the
// realized one, not the planned one. (A pure-plan Audit, which performs no page-in, leaves
// ResidentTokens at the plan's selected cost.) The plan's span COUNTS (Resident/Elided) and
// the Partition/Faithful verdict stay as Audit set them — they describe the plan, which a
// page-in refusal does not corrupt; the count-level reconciliation lives in Reconciled.
//
// Two properties are checked, one structural and one economic:
//
//  1. Reconciled — the page-in loop visits every selected span exactly once and routes each
//     to Rendered or Refused, so a faithful materialization accounts for the WHOLE selected
//     set with none silently dropped. The check is count-based (Rendered+Refused == Selected)
//     so it holds under duplicate IDs, which the planner charges by row.
//  2. CostContract — for every rendered span, the bytes the gate paged in (Rendered.Bytes)
//     must equal the Span.Bytes the planner priced it at (declared[id]). A divergence means
//     the budget charged ceil(Span.Bytes/4) while the render realized ceil(len(body)/4):
//     the accounting is fictional, and the divergence is named in CostDiverged. A refused
//     span carries no body, so its contract is vacuous (counted in Refused, never diverged).
//
// declared maps a span id to the Span.Bytes the planner saw; a rendered id missing from it
// is itself a contract break (the store rendered a span it never reported).
func Reconcile(p Plan, w Witness, rendered []Rendered, refused []Refusal, declared map[string]int64) Witness {
	// carry the plan's Candidates/Resident(span count)/Elided/Recoverable/Partition/Faithful
	// as-is — they describe the plan, which the page-in does not change.
	out := w
	out.Rendered = len(rendered)
	out.Refused = len(refused)
	// ResidentTokens is the REALIZED resident size: the tokens the gate paged in, not the
	// tokens the plan selected. Resetting it here is the rendered+refused reconciliation a
	// caller trusts: a refused span (selected, then declined at page-in) no longer counts as
	// resident, so ResidentTokens == View.RenderedTokens() after Reconcile.
	realized := 0
	for _, r := range rendered {
		realized += r.Tokens
	}
	out.ResidentTokens = realized
	out.Reconciled = len(rendered)+len(refused) == len(p.Selected)
	out.CostContract = true
	for _, r := range rendered {
		if want, ok := declared[r.ID]; !ok || want != r.Bytes {
			out.CostContract = false
			out.CostDiverged = append(out.CostDiverged, r.ID)
		}
	}
	return out
}
