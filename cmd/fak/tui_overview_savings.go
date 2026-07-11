// tui_overview_savings.go folds the Track-2 OBSERVED-$ savings ledger into the compact
// hero model the `fak console overview` renders above its pane table. It is deliberately
// small and pure (rows in, model out) so the fold is easy to test and easy to edit apart
// from the renderer (tui_overview_savings_render.go) and the wiring (tui.go).
package main

import (
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

// publishedSavingsLedgerRel is the tracked historical snapshot the overview falls back to
// when the live (gitignored) ledger has no rows yet, so a fresh checkout still shows the
// last published savings rather than a blank hero. It never shares a row with the live
// ledger — the two provenances stay distinct (see cachevaluereport.DefaultSavingsLedgerRel).
const publishedSavingsLedgerRel = "docs/nightrun/cache-savings.jsonl"

// resolveOverviewSavingsRows reads the Track-2 savings rows the hero folds. An explicit
// path wins; otherwise it prefers the live ledger and falls back to the published
// snapshot. A missing file yields no rows (never an error) — a savings-less overview is a
// valid state, not a failure.
func resolveOverviewSavingsRows(explicit string) []cachevaluereport.SavingsRow {
	if strings.TrimSpace(explicit) != "" {
		return cachevaluereport.ReadSavingsLedgerFile(explicit)
	}
	for _, p := range []string{cachevaluereport.DefaultSavingsLedgerRel, publishedSavingsLedgerRel} {
		if rows := cachevaluereport.ReadSavingsLedgerFile(p); len(rows) > 0 {
			return rows
		}
	}
	return nil
}

// buildTUIOverviewSavings reconciles the savings rows into the hero model via the shared
// audit fold, returning nil when there is nothing to reconcile (so the overview omits the
// hero entirely and its JSON stays backward-compatible). `at` only stamps the fold's
// GeneratedAt; the reconciled numbers are a pure function of the rows.
func buildTUIOverviewSavings(rows []cachevaluereport.SavingsRow, at time.Time) *tuiOverviewSavings {
	rep := cachevaluereport.FoldAudit(rows, at)
	if len(rep.Dates) == 0 {
		return nil
	}
	c := rep.Cumulative
	s := &tuiOverviewSavings{
		Verdict:           rep.Verdict,
		ReductionPct:      c.DollarReductionPct,
		NetUSD:            c.NetUSD,
		CounterfactualUSD: c.CounterfactualUSD,
		SavedTokenEquiv:   c.SavedTokenEquiv,
		CacheReadFraction: c.CacheReadFraction,
		Dates:             len(rep.Dates),
		Rows:              c.Rows,
		Fidelity:          savingsFidelityLabel(rep.FidelitySplit),
	}
	for _, d := range rep.Dates {
		s.NetTrend = append(s.NetTrend, d.CumulativeNetUSD)
	}
	return s
}

// savingsFidelityLabel collapses the audit's fidelity split into one hero tag — e.g.
// "lossless" or "bounded+lossless" — so a reader sees at a glance how faithful the saving
// is without re-reading the full audit. The split arrives sorted by fidelity, so the tag
// is deterministic; empty-row classes are dropped.
func savingsFidelityLabel(split []cachevaluereport.AuditFidelityRow) string {
	labels := make([]string, 0, len(split))
	for _, f := range split {
		if f.Rows == 0 {
			continue
		}
		labels = append(labels, f.Fidelity)
	}
	return strings.Join(labels, "+")
}
