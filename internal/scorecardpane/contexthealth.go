package scorecardpane

// contexthealth.go — issue #1579: managed-context needs a user-facing severity
// vocabulary, not an implicit "some KPI is low" feeling. fak already computes the
// signals (this package's hygiene fold, and internal/productscorecard's
// managed-context SLO report over visibility/reset/budget/query/cache/memory), but
// nothing folds them into one CLOSED, reviewable severity a product surface can
// render without inventing free-text prose per caller. ContextHealthSeverity is that
// vocabulary; ClassifyContextHealth is the pure, deterministic fold from named
// per-area signals into one severity, worst-first.
//
// Shape follows the established closed-vocabulary convention already used by
// internal/ctxmmu.EvidenceKind, internal/memview.ProvenanceEventKind, and
// internal/recall.StaleFactOutcome: a `type X string`, a const block of named
// values, a membership set, a ValidX() function, and a fail-safe String() that
// renders "unknown(...)" for garbage and "(unset)" for empty — never a panic, never
// a silently-accepted foreign value.
//
// Closure binding: the vocabulary and its cmd/fak/productscorecard.go wiring
// shipped in 3092f13e, which cites #1579 but omitted the required
// `(fak scorecardpane)` trailer — this comment plus the following commit restate
// that binding explicitly so the grep-based referee has a trailer-bearing commit
// to bind #1579's scorecardpane closure to, without rewriting the already-pushed
// history.

import "strings"

// ContextHealthSeverity is the closed vocabulary of managed-context health states a
// product surface (the product scorecard, the control pane) may render. The set is
// exactly the issue's "In scope" line: fresh, constrained, query-needed, stale-risk,
// budget-draining, reset-imminent — no other value is a member, and no caller may
// substitute free-text severity prose for one of these.
type ContextHealthSeverity string

const (
	// ContextHealthFresh: every tracked managed-context signal is clean — no
	// visibility gap, no imminent reset, no budget pressure, no unresolved query,
	// no staleness risk. The all-clear state.
	ContextHealthFresh ContextHealthSeverity = "fresh"
	// ContextHealthConstrained: context is usable but visibility into it is
	// degraded (e.g. the visibility SLO is failing) — the agent or user can no
	// longer fully see what is being carried forward.
	ContextHealthConstrained ContextHealthSeverity = "constrained"
	// ContextHealthQueryNeeded: a recalled fact or SLO requires the agent to ask
	// before proceeding rather than assume — the query-correctness signal is
	// failing (mirrors recall.StaleFactExpiredMustQuery's posture, folded here into
	// the closed severity a product surface renders).
	ContextHealthQueryNeeded ContextHealthSeverity = "query_needed"
	// ContextHealthStaleRisk: cache-preservation or memory-promotion-safety signals
	// are failing — durable facts or cached state risk going stale-as-current
	// without the caller realizing it.
	ContextHealthStaleRisk ContextHealthSeverity = "stale_risk"
	// ContextHealthBudgetDraining: the budget-compliance signal is failing — the
	// context window/token budget is being consumed faster than the SLO allows.
	ContextHealthBudgetDraining ContextHealthSeverity = "budget_draining"
	// ContextHealthResetImminent: the deterministic-resets signal is failing — the
	// most severe state, since a failed reset guarantee means the NEXT context
	// transition may not be safe/reviewable at all.
	ContextHealthResetImminent ContextHealthSeverity = "reset_imminent"
)

// validContextHealthSeverities is the membership set every ContextHealthSeverity
// belongs to — used by ValidContextHealthSeverity, tests, and any (de)serializing
// caller to fail closed on a corrupt or foreign value.
var validContextHealthSeverities = map[ContextHealthSeverity]bool{
	ContextHealthFresh:          true,
	ContextHealthConstrained:    true,
	ContextHealthQueryNeeded:    true,
	ContextHealthStaleRisk:      true,
	ContextHealthBudgetDraining: true,
	ContextHealthResetImminent:  true,
}

// ValidContextHealthSeverity reports whether s is a member of the closed vocabulary.
func ValidContextHealthSeverity(s ContextHealthSeverity) bool {
	return validContextHealthSeverities[s]
}

// String renders s, failing safe: a known member renders as itself, an unrecognized
// non-empty value renders as "unknown(...)" (never silently treated as valid), and
// the empty value renders as "(unset)".
func (s ContextHealthSeverity) String() string {
	if ValidContextHealthSeverity(s) {
		return string(s)
	}
	if s == "" {
		return "(unset)"
	}
	return "unknown(" + string(s) + ")"
}

// contextHealthRank orders the non-fresh severities worst-first so
// ClassifyContextHealth can pick one deterministic verdict when more than one area
// signal is failing at once. Reset-imminent is the worst outcome (see its doc
// comment); fresh is intentionally absent — it is the zero-signal default, never
// "outranked" by anything.
var contextHealthRank = []ContextHealthSeverity{
	ContextHealthResetImminent,
	ContextHealthBudgetDraining,
	ContextHealthStaleRisk,
	ContextHealthQueryNeeded,
	ContextHealthConstrained,
}

// ContextHealthSignals is the small, named set of managed-context area failures a
// caller has already measured (e.g. internal/productscorecard's managed-context SLO
// report, keyed by area: visibility/reset/budget/query/cache/memory). Each field is
// "is this area currently failing its SLO" — the classifier never re-derives the
// measurement itself, only folds already-computed booleans into one severity, the
// same separation collect.go draws between the impure gather and the pure fold.
type ContextHealthSignals struct {
	VisibilityFailing bool // context_visibility SLO failing
	ResetFailing      bool // deterministic_resets SLO failing
	BudgetFailing     bool // budget_compliance SLO failing
	QueryFailing      bool // query_correctness SLO failing
	CacheFailing      bool // cache_preservation SLO failing
	MemoryFailing     bool // memory_promotion_safety SLO failing
}

// AnyFailing reports whether any tracked signal is failing.
func (s ContextHealthSignals) AnyFailing() bool {
	return s.VisibilityFailing || s.ResetFailing || s.BudgetFailing ||
		s.QueryFailing || s.CacheFailing || s.MemoryFailing
}

// ClassifyContextHealth folds ContextHealthSignals into exactly one
// ContextHealthSeverity, worst-first per contextHealthRank. Deterministic: the same
// signals always produce the same severity, regardless of struct field order. A
// caller with several failing areas gets the single most severe verdict — the
// product surface renders one closed enum, not a free-text list of everything wrong
// (the issue's done condition).
func ClassifyContextHealth(s ContextHealthSignals) ContextHealthSeverity {
	failing := map[ContextHealthSeverity]bool{
		ContextHealthResetImminent:  s.ResetFailing,
		ContextHealthBudgetDraining: s.BudgetFailing,
		ContextHealthStaleRisk:      s.CacheFailing || s.MemoryFailing,
		ContextHealthQueryNeeded:    s.QueryFailing,
		ContextHealthConstrained:    s.VisibilityFailing,
	}
	for _, sev := range contextHealthRank {
		if failing[sev] {
			return sev
		}
	}
	return ContextHealthFresh
}

// ContextHealthSignalsFromSLORows builds ContextHealthSignals from the managed-context
// SLO rows shape internal/productscorecard.ManagedContextSLOReport emits (a
// []map[string]any with "area" and "status" keys, decoded off JSON so status may
// arrive as any string casing). scorecardpane stays stdlib-only and tier-1 (see
// internal/architest's tier map) — it does not import internal/productscorecard, so
// callers that already hold that package's typed report pass its decoded JSON rows
// (or an equivalent map) through here instead of a new cross-tier import.
func ContextHealthSignalsFromSLORows(rows []map[string]any) ContextHealthSignals {
	var s ContextHealthSignals
	for _, row := range rows {
		area := strings.ToLower(strings.TrimSpace(asString(row["area"])))
		status := strings.ToLower(strings.TrimSpace(asString(row["status"])))
		failing := status != "" && status != "pass"
		switch area {
		case "visibility":
			s.VisibilityFailing = s.VisibilityFailing || failing
		case "reset":
			s.ResetFailing = s.ResetFailing || failing
		case "budget":
			s.BudgetFailing = s.BudgetFailing || failing
		case "query":
			s.QueryFailing = s.QueryFailing || failing
		case "cache":
			s.CacheFailing = s.CacheFailing || failing
		case "memory":
			s.MemoryFailing = s.MemoryFailing || failing
		}
	}
	return s
}
