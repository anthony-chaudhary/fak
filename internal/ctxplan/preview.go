package ctxplan

import (
	"fmt"
	"sort"
	"strings"
)

// preview.go — issue #1574: the managed-context PREVIEW, a pure read/render layer over a
// Plan already produced by PlanLayout/MaterializeLayout. Before a long run starts, an
// operator wants to see WHAT the planner would keep resident and WHAT it would leave cold
// — without executing anything or mutating the store. This file adds no new planning
// logic: it groups an existing Plan's Selected/Elided rows (already carrying the Area and
// Precision metadata layout.go stamps) into the five regions the product surface promises
// (docs on ctxplan, promptmmu, sessionreset, and CONTEXT-IS-NOT-MEMORY.md): pinned, recent,
// deep, elided, and query-needed. It reuses materialize.go/layout.go's plan assembly
// (PlanLayout, ProbeLayout) rather than reinventing candidate scoring or budget
// enforcement — Preview only reads a Plan's own accounting, exactly like Audit/Explain.
//
// THE FIVE REGIONS.
//
//	pinned        forced resident regardless of score (Precision == exact: the base/current
//	              areas by default, or any candidate pinned via Forecast.Pins).
//	recent        planned candidates reached through the AreaRecent access path.
//	deep          planned candidates reached through the AreaDeep access path (relevance +
//	              durability), OR a resident base/current span whose area is not recent/deep
//	              (rare — a base/current span may still be Planned-precision, not just Exact).
//	elided        cold, recoverable spans that lost the knapsack, or were sealed/tombstoned,
//	              or were deduped as a byte-identical duplicate (Reason != pointer).
//	query_needed  cold, recoverable POINTER spans (Precision == pointer / Reason ==
//	              ElidePointer): never faulted in this turn, but the operator must issue a
//	              follow-up query/demand-page to see them — the "left cold on purpose"
//	              region a dry run must call out by name so it is never mistaken for a
//	              silently dropped fact.
//
// Selected+Elided already partition every candidate (faithful.go's witness), so every row
// in the input Plan lands in EXACTLY one preview region — Preview never drops a row, and
// PreviewOf's row count always equals p.Candidates (checked by TestContextPlanPreviewCoversEveryCandidate).

// Region names — the product-facing vocabulary the issue's "Done condition" asks a dry run
// to show (pinned, recent, deep, elided, query-needed).
const (
	RegionPinned      = "pinned"
	RegionRecent      = "recent"
	RegionDeep        = "deep"
	RegionElided      = "elided"
	RegionQueryNeeded = "query_needed"
)

// PreviewRow is one span's projection into the preview — enough to explain the disposition
// without re-deriving it from the plan: which region it landed in, its area/precision
// provenance, cost/benefit, and (for a cold span) its recovery handle and reason.
type PreviewRow struct {
	ID         string  `json:"id"`
	Step       int     `json:"step"`
	Role       string  `json:"role,omitempty"`
	Descriptor string  `json:"descriptor,omitempty"`
	Area       string  `json:"area,omitempty"`
	Precision  string  `json:"precision,omitempty"`
	Cost       int     `json:"cost"`
	Benefit    float64 `json:"benefit"`
	Resident   bool    `json:"resident"`
	Reason     string  `json:"reason,omitempty"` // elision reason; empty for a resident row
	Digest     string  `json:"digest,omitempty"` // recovery handle for a cold row
}

// Preview is the rendered, human-readable projection of a Plan — what a dry run shows an
// operator before a long run starts. It carries no bytes and performs no I/O or store
// access: it is computed entirely from the Plan's own Selected/Elided accounting, so
// previewing a plan can never page in a sealed span or otherwise touch the trust gate.
type Preview struct {
	PlanID      string       `json:"plan_id,omitempty"`
	Budget      int          `json:"budget"`
	Objective   string       `json:"objective"`
	Horizon     int          `json:"horizon,omitempty"`
	Candidates  int          `json:"candidates"`
	CostUsed    int          `json:"cost_used"`
	Benefit     float64      `json:"benefit"`
	OverBudget  bool         `json:"over_budget"`
	Pinned      []PreviewRow `json:"pinned"`
	Recent      []PreviewRow `json:"recent"`
	Deep        []PreviewRow `json:"deep"`
	Elided      []PreviewRow `json:"elided"`
	QueryNeeded []PreviewRow `json:"query_needed"`
	Faithful    bool         `json:"faithful"` // Audit(plan).Faithful, carried so a preview never has to be paired with a second call to trust
}

// PreviewOf renders p into the five-region Preview. It is pure and total: every row of
// p.Selected and p.Elided is classified into exactly one region, so len(Pinned)+
// len(Recent)+len(Deep)+len(Elided)+len(QueryNeeded) always equals p.Candidates for a
// well-formed (Faithful) plan.
func PreviewOf(p Plan) Preview {
	pv := Preview{
		PlanID: p.ID, Budget: p.Budget, Objective: p.Objective, Horizon: p.Horizon,
		Candidates: p.Candidates, CostUsed: p.CostUsed, Benefit: p.Benefit, OverBudget: p.OverBudget,
		Faithful: Audit(p).Faithful,
	}
	for _, s := range p.Selected {
		row := PreviewRow{
			ID: s.ID, Step: s.Step, Role: s.Role, Descriptor: s.Descriptor,
			Area: s.Area, Precision: s.Precision, Cost: s.Cost, Benefit: s.Benefit, Resident: true,
		}
		switch {
		case s.Pinned || s.Precision == PrecisionExact:
			pv.Pinned = append(pv.Pinned, row)
		case s.Area == AreaRecent:
			pv.Recent = append(pv.Recent, row)
		default:
			// AreaDeep, AreaBase/AreaCurrent-without-exact-precision, or unlabeled (a
			// hand-built Plan with no layout metadata) all land in deep — "resident but
			// reached by relevance/durability rather than a hard pin or the recency tail"
			// is exactly the deep region's definition.
			pv.Deep = append(pv.Deep, row)
		}
	}
	for _, e := range p.Elided {
		row := PreviewRow{
			ID: e.ID, Step: e.Step, Role: e.Role, Area: e.Area, Precision: e.Precision,
			Cost: e.Cost, Benefit: e.Benefit, Resident: false, Reason: e.Reason, Digest: e.Digest,
		}
		if e.Reason == ElidePointer {
			pv.QueryNeeded = append(pv.QueryNeeded, row)
		} else {
			pv.Elided = append(pv.Elided, row)
		}
	}
	sortPreviewRows(pv.Pinned)
	sortPreviewRows(pv.Recent)
	sortPreviewRows(pv.Deep)
	sortPreviewRows(pv.Elided)
	sortPreviewRows(pv.QueryNeeded)
	return pv
}

func sortPreviewRows(rows []PreviewRow) {
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Step != rows[j].Step {
			return rows[i].Step < rows[j].Step
		}
		return rows[i].ID < rows[j].ID
	})
}

// RowCount is the total number of rows across all five regions — for a Faithful plan this
// always equals Candidates (every candidate lands in exactly one region), the invariant
// TestContextPlanPreviewCoversEveryCandidate checks.
func (pv Preview) RowCount() int {
	return len(pv.Pinned) + len(pv.Recent) + len(pv.Deep) + len(pv.Elided) + len(pv.QueryNeeded)
}

// Explain renders the preview as an operator-readable dry run: one line per region with
// its row count and tokens, then a per-span breakdown — the "would page in / would leave
// cold" report the issue's Done condition asks for. It never claims more than the plan
// itself proves: an OverBudget plan says so up front, and the faithfulness line makes clear
// whether every cold span is still recoverable.
func (pv Preview) Explain() string {
	var b strings.Builder
	fmt.Fprintf(&b, "ctxplan preview (dry run): objective=%s budget=%d tokens, %d candidate(s)",
		pv.Objective, pv.Budget, pv.Candidates)
	if pv.PlanID != "" {
		fmt.Fprintf(&b, ", plan_id=%s", pv.PlanID)
	}
	if pv.Horizon > 0 {
		fmt.Fprintf(&b, ", horizon=%d turn(s)", pv.Horizon)
	}
	b.WriteByte('\n')
	if pv.OverBudget {
		b.WriteString("  WARNING: pins alone exceed the budget; nothing else would be paged in\n")
	}
	writeRegion(&b, "PINNED       (forced resident)", pv.Pinned)
	writeRegion(&b, "RECENT       (planned, near-verbatim tail)", pv.Recent)
	writeRegion(&b, "DEEP         (planned, relevance/durability reach)", pv.Deep)
	writeRegion(&b, "ELIDED       (cold, recoverable — lost the knapsack / sealed / tombstoned / duplicate)", pv.Elided)
	writeRegion(&b, "QUERY-NEEDED (cold, recoverable — needs an explicit follow-up query to page in)", pv.QueryNeeded)
	fmt.Fprintf(&b, "  totals: resident=%d tokens used of %d budget, benefit=%.3f, faithful=%v\n",
		pv.CostUsed, pv.Budget, pv.Benefit, pv.Faithful)
	return b.String()
}

func writeRegion(b *strings.Builder, title string, rows []PreviewRow) {
	fmt.Fprintf(b, "  %s: %d span(s)\n", title, len(rows))
	for _, r := range rows {
		if r.Resident {
			fmt.Fprintf(b, "     [step %-4d] %-14s cost=%-5d benefit=%.3f  %s\n",
				r.Step, truncate(r.Role, 14), r.Cost, r.Benefit, truncate(r.Descriptor, 60))
		} else {
			fmt.Fprintf(b, "     [step %-4d] %-14s cost=%-5d reason=%-11s handle=%s\n",
				r.Step, truncate(r.Role, 14), r.Cost, r.Reason, short(r.Digest))
		}
	}
}

// Markdown renders the same five regions as a Markdown report — the shareable form for a
// teammate reviewing the plan for a long run outside a terminal (mirrors the
// Plan.Explain/contextq.Result.Markdown convention already used elsewhere in this repo).
func (pv Preview) Markdown() string {
	var b strings.Builder
	b.WriteString("# Context plan preview (dry run)\n\n")
	fmt.Fprintf(&b, "- objective: `%s`\n", pv.Objective)
	fmt.Fprintf(&b, "- budget: %d tokens (used %d)\n", pv.Budget, pv.CostUsed)
	fmt.Fprintf(&b, "- candidates: %d\n", pv.Candidates)
	if pv.PlanID != "" {
		fmt.Fprintf(&b, "- plan id: `%s`\n", pv.PlanID)
	}
	fmt.Fprintf(&b, "- benefit: %.3f\n", pv.Benefit)
	fmt.Fprintf(&b, "- faithful: %v\n", pv.Faithful)
	if pv.OverBudget {
		b.WriteString("- **WARNING**: pins alone exceed the budget; nothing else would be paged in\n")
	}
	b.WriteByte('\n')
	regions := []struct {
		title string
		rows  []PreviewRow
	}{
		{"Pinned (forced resident)", pv.Pinned},
		{"Recent (planned, near-verbatim tail)", pv.Recent},
		{"Deep (planned, relevance/durability reach)", pv.Deep},
		{"Elided (cold, recoverable)", pv.Elided},
		{"Query-needed (cold, needs a follow-up query)", pv.QueryNeeded},
	}
	for _, r := range regions {
		fmt.Fprintf(&b, "## %s — %d span(s)\n\n", r.title, len(r.rows))
		if len(r.rows) == 0 {
			b.WriteString("_none_\n\n")
			continue
		}
		b.WriteString("| step | role | cost | benefit | detail |\n|---|---|---|---|---|\n")
		for _, row := range r.rows {
			detail := row.Descriptor
			if !row.Resident {
				detail = "reason=" + row.Reason + " handle=" + short(row.Digest)
			}
			fmt.Fprintf(&b, "| %d | %s | %d | %.3f | %s |\n",
				row.Step, mdEscape(row.Role), row.Cost, row.Benefit, mdEscape(detail))
		}
		b.WriteByte('\n')
	}
	return b.String()
}

func mdEscape(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "|", "\\|"), "\n", " ")
}

// PreviewLayout is the one-call convenience an operator/CLI actually wants: given a store's
// spans, a forecast, a budget, and a layout, plan the O(1) view (PlanLayout — the same
// assembly MaterializeLayout uses) and immediately render its Preview, WITHOUT paging any
// bytes in. It performs no I/O beyond what the caller already did to obtain spans — no
// Store.Materialize call is made, so a preview can never surface sealed bytes or mutate
// anything, exactly the "render, don't execute" contract the issue asks for.
func PreviewLayout(spans []Span, f Forecast, b Budget, cost CostModel, layout Layout) Preview {
	p := BuildIndex(spans).PlanLayout(f, b, cost, layout)
	return PreviewOf(p)
}
