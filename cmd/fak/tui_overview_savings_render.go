// tui_overview_savings_render.go renders the cache-savings hero the overview places above
// its pane table. It is kept apart from the fold (tui_overview_savings.go) so the visual
// can be edited without touching the reconciliation, and reuses the shared TUI primitives
// (gaugeBarTUI, sparklineTUI, groupThousands, trimTUI) rather than re-deriving them.
package main

import (
	"fmt"
	"strings"
)

// renderTUIOverviewSavingsHero draws the two-line savings hero: a headline gauge bar with
// the net-$ reduction, then a row of supporting stat chips plus a net-trend sparkline. It
// returns "" for a nil model, so the caller can hoist it above the fold unconditionally.
// The returned string carries no leading/trailing blank lines — the caller spaces it.
func renderTUIOverviewSavingsHero(s *tuiOverviewSavings, width int) string {
	if s == nil {
		return ""
	}
	frac, headline := savingsHeroHeadline(s)
	bar := gaugeBarTUI(frac, savingsBarWidth(width))

	detail := strings.Join(savingsHeroStats(s), " · ")
	if spark := savingsHeroSpark(s); spark != "" {
		detail += "  " + spark
	}

	var b strings.Builder
	fmt.Fprintf(&b, "  Cache savings  %s  %s\n", bar, headline)
	fmt.Fprintf(&b, "  %s", trimTUI(detail, maxTUI(20, width-2)))
	return b.String()
}

// savingsHeroHeadline picks the bar fraction and headline. A priced window leads with its
// net-$ cost reduction; a fully dollar-blind window falls back to the cache-read fraction
// (labeled so it is never mistaken for a dollar figure); a token-only window says so.
func savingsHeroHeadline(s *tuiOverviewSavings) (float64, string) {
	switch {
	case s.ReductionPct != nil:
		return *s.ReductionPct / 100, fmt.Sprintf("%.1f%% net API $ cost reduction", *s.ReductionPct)
	case s.CacheReadFraction != nil:
		return *s.CacheReadFraction, fmt.Sprintf("%.0f%% prompt tokens served from cache (dollar-blind)", *s.CacheReadFraction*100)
	default:
		return 0, "token-only savings recorded (no priced counterfactual)"
	}
}

// savingsHeroStats is the ordered list of supporting chips shown beneath the headline. Net
// dollars always lead; the cache-read fraction is added only when it is not already the
// headline, so the two lines never repeat the same number.
func savingsHeroStats(s *tuiOverviewSavings) []string {
	stats := []string{"net " + formatSavingsUSD(s.NetUSD)}
	if s.ReductionPct != nil && s.CacheReadFraction != nil {
		stats = append(stats, fmt.Sprintf("%.0f%% from cache", *s.CacheReadFraction*100))
	}
	if s.SavedTokenEquiv >= 1 {
		stats = append(stats, groupThousands(int64(s.SavedTokenEquiv+0.5))+" tok-equiv")
	}
	dateWord := "dates"
	if s.Dates == 1 {
		dateWord = "date"
	}
	stats = append(stats, fmt.Sprintf("%s rows / %d %s", groupThousands(int64(s.Rows)), s.Dates, dateWord))
	if s.Fidelity != "" {
		stats = append(stats, s.Fidelity)
	}
	return stats
}

// savingsHeroSpark renders the cumulative-net trend as a sparkline, or "" when there is
// only one date (a single point is not a trend).
func savingsHeroSpark(s *tuiOverviewSavings) string {
	if len(s.NetTrend) < 2 {
		return ""
	}
	return sparklineTUI(s.NetTrend, minTUI(len(s.NetTrend), 16))
}

// savingsBarWidth scales the headline gauge with the terminal width, clamped to a legible
// band so it neither vanishes on a narrow pane nor sprawls on a wide one.
func savingsBarWidth(width int) int {
	return minTUI(28, maxTUI(12, width/5))
}

// formatSavingsUSD renders a dollar figure compactly for the hero: $/k/M with a sign, so a
// large cumulative saving reads at a glance without a wall of digits.
func formatSavingsUSD(v float64) string {
	sign := ""
	a := v
	if a < 0 {
		sign = "-"
		a = -a
	}
	switch {
	case a >= 1_000_000:
		return fmt.Sprintf("%s$%.1fM", sign, a/1_000_000)
	case a >= 1_000:
		return fmt.Sprintf("%s$%.1fk", sign, a/1_000)
	default:
		return fmt.Sprintf("%s$%.2f", sign, a)
	}
}
