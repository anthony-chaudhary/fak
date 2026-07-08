package ctxplan

import (
	"fmt"
	"sort"
	"strings"
)

// retention.go (#3024) — the plan-time RETENTION/PIN OUTCOME surface. release.go already
// answers "what did my releases do?" (ReleaseReport); this is the dual an operator needs
// BEFORE a view is materialized: for each retention target I named — a Forecast.Pin ("keep
// this resident") or a Forecast.Release ("I no longer need this") — was it HONORED, MISSING,
// RELEASED, or REFUSED? Built-in compaction is weak precisely because a summary can look
// coherent while silently dropping the fact that mattered; this surface makes the retention
// guarantee inspectable up front.
//
// It is a pure read layer over a Plan's own Selected/Elided accounting (exactly like
// buildReleaseReport and PreviewOf): it changes nothing about what wins the knapsack and
// pages no bytes — a sealed/tombstoned target is reported REFUSED from the plan's elision
// reason, never by touching the trust gate. The load-bearing property is that a sealed
// retained span shows as `refused`, never silently as `honored`.

// Retention-outcome vocabulary — closed and disjoint. Every requested pin/release target maps
// to exactly one, read from the plan's disposition of that id (never recomputed).
const (
	// RetentionHonored — the target is RESIDENT in the planned view (a pin held, or a pin/floor
	// outranked a conflicting release so the span stayed).
	RetentionHonored = "honored"
	// RetentionReleased — the target is COLD because the agent released it (ElideReleased):
	// recoverable, its budget freed, NOT suppressed.
	RetentionReleased = "released"
	// RetentionRefused — the target was excluded by the TRUST GATE (sealed/tombstoned) before
	// materialization. This is the outcome that must never silently read as honored.
	RetentionRefused = "refused"
	// RetentionMissing — the target named no candidate span (a stale/fabricated id): an
	// advisory no-op that cannot poison the plan.
	RetentionMissing = "missing"
	// RetentionCold — the target is COLD but RECOVERABLE for a layout/economy reason (kept as a
	// query-needed pointer, lost the knapsack, or deduped against a byte-identical span), NOT
	// excluded by the trust gate. It is one demand-page away. Distinct from `refused` on
	// purpose: reporting a pointer/over-budget span as refused would falsely implicate the gate.
	RetentionCold = "cold"
)

// RetentionOutcome is the plan-time disposition of ONE requested retention target — the typed
// answer to "was this pin/retention target honored, missing, released, or refused?".
type RetentionOutcome struct {
	ID       string `json:"id"`
	Request  string `json:"request"`          // pin | release | pin+release (which forecast list named it)
	Outcome  string `json:"outcome"`          // one of the closed vocabulary above
	Reason   string `json:"reason,omitempty"` // the underlying elision reason for released/refused (sealed|tombstoned|released)
	Step     int    `json:"step,omitempty"`
	Role     string `json:"role,omitempty"`
	Resident bool   `json:"resident"`
}

// RetentionOutcomes classifies every requested retention target (Forecast.Pins ∪
// Forecast.Releases, deduped) against the plan that honored or refused it. Pure and
// deterministic: targets are sorted by id, and the verdict is read off the plan's own
// Selected/Elided accounting. It requires the Forecast because a target that named NO span
// (missing) leaves no trace in the Plan alone.
func RetentionOutcomes(p Plan, f Forecast) []RetentionOutcome {
	if len(f.Pins) == 0 && len(f.Releases) == 0 {
		return nil
	}

	resident := make(map[string]Selection, len(p.Selected))
	for _, s := range p.Selected {
		resident[s.ID] = s
	}
	elided := make(map[string]Elision, len(p.Elided))
	for _, e := range p.Elided {
		if _, ok := elided[e.ID]; !ok { // first elision reason wins, matching buildReleaseReport
			elided[e.ID] = e
		}
	}

	// Tag each target by which forecast list(s) named it, preserving a stable id order.
	type tag struct{ pin, release bool }
	tags := map[string]*tag{}
	var order []string
	note := func(id string, pin bool) {
		t, ok := tags[id]
		if !ok {
			t = &tag{}
			tags[id] = t
			order = append(order, id)
		}
		if pin {
			t.pin = true
		} else {
			t.release = true
		}
	}
	for _, id := range f.Pins {
		note(id, true)
	}
	for _, id := range f.Releases {
		note(id, false)
	}
	sort.Strings(order)

	out := make([]RetentionOutcome, 0, len(order))
	for _, id := range order {
		t := tags[id]
		o := RetentionOutcome{ID: id, Request: requestLabel(t.pin, t.release)}
		switch {
		case has(resident, id):
			sel := resident[id]
			o.Outcome, o.Resident, o.Step, o.Role = RetentionHonored, true, sel.Step, sel.Role
		case has(elided, id):
			e := elided[id]
			o.Step, o.Role = e.Step, e.Role
			switch e.Reason {
			case ElideSealed, ElideTombstoned:
				o.Outcome, o.Reason = RetentionRefused, e.Reason
			case ElideReleased:
				o.Outcome, o.Reason = RetentionReleased, e.Reason
			default:
				// over_budget / duplicate / pointer: the LAYOUT or knapsack left it cold (e.g. a
				// deep durable span kept as a query-needed pointer), NOT the trust gate. Report
				// `cold` (recoverable, one demand-page away), never `refused` — that would falsely
				// implicate the gate — and never a silent `honored`.
				o.Outcome, o.Reason = RetentionCold, e.Reason
			}
		default:
			o.Outcome = RetentionMissing
		}
		out = append(out, o)
	}
	return out
}

func has[V any](m map[string]V, k string) bool {
	_, ok := m[k]
	return ok
}

func requestLabel(pin, release bool) string {
	switch {
	case pin && release:
		return "pin+release"
	case release:
		return "release"
	default:
		return "pin"
	}
}

// writeRetentionExplain renders the retention outcomes as a text block for Preview.Explain.
// Nothing is written when there are no requested targets.
func writeRetentionExplain(b *strings.Builder, outcomes []RetentionOutcome) {
	if len(outcomes) == 0 {
		return
	}
	fmt.Fprintf(b, "  RETENTION    (requested pin/release targets): %d target(s)\n", len(outcomes))
	for _, o := range outcomes {
		detail := ""
		if o.Reason != "" {
			detail = " (" + o.Reason + ")"
		}
		fmt.Fprintf(b, "     %-9s %-14s %s%s\n", o.Outcome, truncate(o.Role, 14), o.ID, detail)
	}
}

// retentionMarkdown renders the retention outcomes as a Markdown section for Preview.Markdown.
func retentionMarkdown(outcomes []RetentionOutcome) string {
	if len(outcomes) == 0 {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "## Retention / pin outcomes — %d target(s)\n\n", len(outcomes))
	b.WriteString("| target | request | outcome | reason |\n|---|---|---|---|\n")
	for _, o := range outcomes {
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", mdEscape(o.ID), o.Request, o.Outcome, mdEscape(o.Reason))
	}
	b.WriteByte('\n')
	return b.String()
}
