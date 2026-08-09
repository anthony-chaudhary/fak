package modver

// regression.go — the advisory score-regression check (#2470). It compares the
// score a module carries in the CURRENT report (whatever `--scores`,
// `--coverage`, or `--maturity` just joined) against the score its most recent
// scored row in the append-only fak-module-versions/1 ledger recorded, and names
// the modules that went DOWN. That closes the loop from measurement to steering:
// the ledger already remembers what a module scored at rev N, so a drop at rev
// N+k is detectable at the moment the next stamp is about to be written.
//
// It is ADVISORY in the PRIOR_ART sense: the fold returns findings and the
// renderer prints a warning, but no caller changes its exit code and no stamp is
// withheld. Making it blocking is a separate decision with its own escape hatch;
// a first-of-its-kind check that refuses commits before anyone has seen its false
// positives is how a useful signal gets disabled wholesale.
//
// The fold is pure — a Report plus the prior ledger bytes in, rows out; no git,
// no clock, no I/O — so both directions (a fixture regression warns, an unchanged
// or improved corpus stays silent) are fixture-testable. It adds NO ledger schema
// surface: it only READS rows the /1 schema already carries.

import (
	"fmt"
	"io"
	"sort"
)

// ScoreDrop is one module whose current joined score sits below the score its
// last scored ledger row recorded. Prev/PrevRev/PrevTS describe that remembered
// observation and Current/Rev the fresh one, so the advisory can name both ends
// of the movement rather than only asserting that it happened.
type ScoreDrop struct {
	Module  string  `json:"module"`
	Prev    float64 `json:"prev"`
	Current float64 `json:"current"`
	Delta   float64 `json:"delta"` // Current - Prev, always negative for a drop
	PrevRev int     `json:"prev_rev"`
	Rev     int     `json:"rev"`
	PrevTS  string  `json:"prev_ts,omitempty"` // when the remembered score was stamped
}

// ScoreDrops folds the current report against the prior ledger and returns one
// entry per module whose score fell, worst drop first (ties by module name) so
// the head of the list is the movement most worth looking at.
//
// Three cases are deliberately SILENT, because each is an absence of evidence
// rather than a regression:
//
//   - the module carries no score this run (nothing was joined) — there is no
//     current observation to compare;
//   - the ledger remembers no scored row for it — a first observation cannot
//     regress against nothing;
//   - the score is unchanged or higher — flat and improving are not drops.
//
// The comparison anchors on the last row that actually CARRIES a score, not
// simply the last row. The ledger is delta-encoded and a stamp taken without a
// score join writes a scoreless row, so anchoring on the literal last row would
// let an unscored stamp erase the memory of what the module scored before it and
// silently swallow a real regression across it.
func ScoreDrops(rep Report, prevLedger []byte) []ScoreDrop {
	scored := map[string]LedgerRow{}
	for _, row := range parseLedgerRows(prevLedger) {
		if row.Score == nil {
			continue
		}
		scored[row.Module] = row // later rows win: the most recent scored observation
	}
	var drops []ScoreDrop
	for _, m := range rep.Modules {
		if m.Score == nil {
			continue
		}
		last, ok := scored[m.Name]
		if !ok || *m.Score >= *last.Score {
			continue
		}
		drops = append(drops, ScoreDrop{
			Module:  m.Name,
			Prev:    *last.Score,
			Current: *m.Score,
			Delta:   *m.Score - *last.Score,
			PrevRev: last.Rev,
			Rev:     m.Rev,
			PrevTS:  last.TS,
		})
	}
	sort.Slice(drops, func(i, j int) bool {
		if drops[i].Delta != drops[j].Delta {
			return drops[i].Delta < drops[j].Delta // most negative first
		}
		return drops[i].Module < drops[j].Module
	})
	return drops
}

// ScoreDropAdvisory writes the human advisory for drops and reports whether it
// wrote anything. An empty set writes NOTHING — not a "0 regressions" line — so
// the check is invisible until it has something to say and a caller can wire it
// into a routine command without adding noise to every run. The header says the
// advisory blocks nothing out loud, so a reader never has to guess whether the
// warning just failed their command.
func ScoreDropAdvisory(w io.Writer, drops []ScoreDrop) bool {
	if len(drops) == 0 {
		return false
	}
	fmt.Fprintf(w, "modver: score regression advisory — %d module(s) scored below their last stamped score (advisory only, nothing is blocked)\n", len(drops))
	for _, d := range drops {
		// The delta is rendered at 6 significant digits on purpose: it is a
		// subtraction of two floats, so the exact difference of 0.8 and 0.5 prints
		// as -0.30000000000000004 at full width and reads like a precision claim
		// the score never made. The stored Delta stays exact for a JSON consumer.
		fmt.Fprintf(w, "  %-28s %g -> %g (%+.6g)  r%d -> r%d  last stamped %s\n",
			d.Module, d.Prev, d.Current, d.Delta, d.PrevRev, d.Rev, shortDate(d.PrevTS))
	}
	return true
}
