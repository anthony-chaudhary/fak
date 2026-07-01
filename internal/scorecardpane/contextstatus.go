package scorecardpane

// contextstatus.go — issue #1577: managed-context needs a CONCISE, one-line status a
// live session can render, not just the closed severity enum #1579 added. The parent
// epic (#1570) promise is that managed context stays legible in "plain product
// surfaces" (`fak info` and the split-pane overlay) instead of staying an invisible
// runtime mechanism. This file adds the one-line renderer; it does not add a new
// gather step or a new severity vocabulary — it composes ContextHealthSignals /
// ClassifyContextHealth (#1579, contexthealth.go) with the six raw counters the issue
// names (resident tokens, budget left, cache state, reset count, query-needed count,
// stale assumption count) into ONE deterministic string a caller fmt.Fprintln's as-is.
//
// Shape follows internal/ctxplan/preview.go's house style for a terse, deterministic
// render function: a plain input struct with no behavior, a pure Render function with
// no I/O, and a matching golden-output test proving each severity tier renders
// distinctly (TestRenderContextStatusLine*).

import (
	"fmt"
	"strconv"
	"strings"
)

// ContextStatusSignals is the small, named set of live managed-context counters a
// caller (a running gateway, an offline transcript scan) has already measured, plus
// the already-classified ContextHealthSeverity (#1579) for the same snapshot. The
// renderer never re-derives severity itself — exactly the gather/fold separation
// collect.go and contexthealth.go both draw — so a caller that already holds a
// ContextHealthSeverity (e.g. from ClassifyContextHealth or
// ContextHealthSignalsFromSLORows) passes it straight through.
type ContextStatusSignals struct {
	// Severity is the closed #1579 verdict for this snapshot. The zero value
	// ("") renders as ContextHealthSeverity's own fail-safe "(unset)" rather than
	// being silently treated as fresh — a caller MUST classify before rendering.
	Severity ContextHealthSeverity

	ResidentTokens       int    // tokens currently resident (paged in) for this session
	BudgetTokens         int    // the session's total token budget (0 = unknown/not tracked)
	CacheState           string // e.g. "stable", "mutated", "unknown" — free text from the caller's own cache tracker (cachemeta et al.); rendered verbatim, empty means "not reported"
	ResetCount           int    // deterministic resets performed so far this session
	QueryNeededCount     int    // spans left cold on purpose, needing an explicit follow-up query (mirrors ctxplan.Preview's QueryNeeded region)
	StaleAssumptionCount int    // recalled facts/assumptions flagged stale (recall.StaleFact* territory) not yet refreshed or re-queried
}

// BudgetLeftTokens returns the remaining token budget (BudgetTokens - ResidentTokens),
// floored at 0 so a caller that over-ran its budget never reports a negative "left"
// figure — the status line says 0 left, not a confusing negative number.
func (s ContextStatusSignals) BudgetLeftTokens() int {
	left := s.BudgetTokens - s.ResidentTokens
	if left < 0 {
		return 0
	}
	return left
}

// RenderContextStatusLine renders s as ONE concise, deterministic line: severity
// first (the single most-important word, per #1579's worst-first classification),
// then resident/budget-left tokens, cache state, reset count, query-needed count, and
// stale-assumption count — exactly the issue's "In scope" list, in that order, every
// time. It performs no I/O and never panics: an empty/zero-value input still renders a
// well-formed line (severity "(unset)", zero counts, cache "unknown").
//
// The line is intentionally FLAT text (no JSON), the same "plain words a non-technical
// watcher can read at a glance" contract cmd/fak/info.go's renderGuardInfoLine already
// promises for the sibling cache/safety/liveness line — this is the managed-context
// sibling of that line, meant to be appended or shown alongside it.
func RenderContextStatusLine(s ContextStatusSignals) string {
	cache := strings.TrimSpace(s.CacheState)
	if cache == "" {
		cache = "unknown"
	}
	return fmt.Sprintf(
		"context: %s · resident %s tok · budget left %s tok · cache %s · resets %d · query-needed %d · stale %d",
		s.Severity.String(),
		groupThousandsCS(s.ResidentTokens),
		groupThousandsCS(s.BudgetLeftTokens()),
		cache,
		s.ResetCount,
		s.QueryNeededCount,
		s.StaleAssumptionCount,
	)
}

// groupThousandsCS formats a non-negative int with comma separators (12345 ->
// "12,345"), mirroring cmd/fak/info.go's groupThousands for the same human-readable
// token counts. Kept local (the "CS" suffix disambiguates it from that cmd/fak
// helper) rather than imported: scorecardpane is a tier-1, dependency-light package
// (see internal/architest's tier map) and this is a five-line formatting helper, not
// a shared abstraction worth a cross-package seam for.
func groupThousandsCS(n int) string {
	if n < 0 {
		n = 0
	}
	s := strconv.Itoa(n)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	pre := len(s) % 3
	if pre > 0 {
		b.WriteString(s[:pre])
		if len(s) > pre {
			b.WriteByte(',')
		}
	}
	for i := pre; i < len(s); i += 3 {
		b.WriteString(s[i : i+3])
		if i+3 < len(s) {
			b.WriteByte(',')
		}
	}
	return b.String()
}
