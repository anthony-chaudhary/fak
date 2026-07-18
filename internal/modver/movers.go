package modver

// movers.go folds the append-only module-versions ledger into the `fak doctor`
// MOVERS section (#2472): the fastest-moving modules (biggest rev delta over the
// recorded window) and the dormant modules that still carry open issues. Both
// sub-sections are the existing pure finders (Trend + Dormant), so this file only
// ranks, truncates, and renders — it adds NO new ledger semantics and no new
// fak-module-versions/1 schema surface. The fold is PURE and deterministic:
// ledger bytes + the caller's open-issue feed + a `now` in, a section out; no
// git, no clock, no I/O — so the render is a fixture-testable witness.

import (
	"bytes"
	"fmt"
	"time"
)

// DefaultMoversTop is how many top movers / dormant candidates the doctor movers
// section shows by default — the "top-5" the version-everything dogfood surface
// (#2472) names.
const DefaultMoversTop = 5

// MoversSection is the structured doctor movers readout: the ledger window it
// folded, the fastest-moving modules (biggest rev delta first), and the
// dormant-with-open-issues candidates. Every field is JSON-tagged so `fak doctor
// movers --json` can emit the section for a downstream reader.
type MoversSection struct {
	Window     [2]string     `json:"window"`      // [earliest ts, latest ts] across the ledger ("" when empty)
	LedgerRows int           `json:"ledger_rows"` // parseable ledger rows folded
	TopMovers  []ModuleTrend `json:"top_movers"`  // modules whose rev grew over the window, biggest delta first
	Dormant    DormantReport `json:"dormant"`     // dormant modules that still carry open issues
}

// Movers folds the ledger (and the caller's open-issue feed) into the doctor
// movers section: the top `top` modules by rev delta over the recorded window
// that actually grew, plus the dormant-with-open-issues candidates judged
// against `now` with a `dormantDays` idle window (<= 0 uses DefaultDormantDays).
// `top` <= 0 uses DefaultMoversTop. A module whose rev did not move over the
// window is not a mover and is dropped — a "fastest-moving" list of flat modules
// would be noise. `issues` is the caller's feed (Dormant never fetches issues
// itself); pass nil and the dormant sub-section is empty rather than errored.
func Movers(ledger []byte, issues []OpenIssue, now time.Time, dormantDays, top int) MoversSection {
	if top <= 0 {
		top = DefaultMoversTop
	}
	tr := Trend(ledger) // sorted by rev delta desc already
	sec := MoversSection{Window: tr.Window, LedgerRows: tr.Rows}
	for _, m := range tr.Modules {
		if m.RevDelta <= 0 {
			continue // did not grow over the window: not a mover
		}
		sec.TopMovers = append(sec.TopMovers, m)
		if len(sec.TopMovers) >= top {
			break
		}
	}
	dr := Dormant(ledger, issues, now, dormantDays)
	if top < len(dr.Candidates) {
		dr.Candidates = dr.Candidates[:top] // keep Scanned honest; show only the top slice
	}
	sec.Dormant = dr
	return sec
}

// Render prints the human-readable doctor movers section. An empty sub-section
// renders an explicit honest line naming WHY it is empty (an empty ledger, a
// missing issue feed, or a fresh corpus) rather than a blank, so "0 movers"
// never silently reads as "all healthy" when the real cause is no data.
func (s MoversSection) Render() string {
	var b bytes.Buffer
	fmt.Fprintln(&b, "== fak doctor: module movers ==")

	window := "empty ledger"
	if s.Window[0] != "" {
		window = shortDate(s.Window[0]) + ".." + shortDate(s.Window[1])
	}
	fmt.Fprintf(&b, "top movers (rev delta over %s, %d ledger rows):\n", window, s.LedgerRows)
	if len(s.TopMovers) == 0 {
		fmt.Fprintln(&b, "  none — no module grew over the recorded window (seed with `fak version modules --stamp`)")
	} else {
		for _, m := range s.TopMovers {
			line := fmt.Sprintf("  Δr%+d  r%d→r%d  %s", m.RevDelta, m.FirstRev, m.LastRev, m.Module)
			if m.ScoreDelta != nil {
				line += fmt.Sprintf("  Δscore %+g", *m.ScoreDelta)
			}
			fmt.Fprintln(&b, line)
		}
	}

	fmt.Fprintf(&b, "dormant modules with open issues (idle ≥ %dd, judged %s):\n",
		s.Dormant.Days, shortDate(s.Dormant.Now))
	switch {
	case len(s.Dormant.Candidates) > 0:
		for _, c := range s.Dormant.Candidates {
			var refs bytes.Buffer
			for i, ref := range c.Issues {
				if i > 0 {
					refs.WriteByte(' ')
				}
				fmt.Fprintf(&refs, "#%d", ref.Number)
			}
			fmt.Fprintf(&b, "  %-24s %dd idle  r%d  issues: %s\n", c.Module, c.DaysIdle, c.Rev, refs.String())
		}
	case s.Dormant.Scanned == 0:
		fmt.Fprintln(&b, "  none — no open-issue feed cross-referenced (pass --issues <file> of {number,paths})")
	default:
		fmt.Fprintln(&b, "  none — every referenced module was touched inside the window")
	}
	return b.String()
}

// shortDate trims an RFC3339 timestamp to its YYYY-MM-DD date for compact
// display; a shorter or empty string passes through unchanged.
func shortDate(ts string) string {
	if len(ts) > 10 {
		return ts[:10]
	}
	return ts
}
