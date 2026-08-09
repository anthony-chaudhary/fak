package modver

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// TrendPoint is one stamped observation in a module's series: the timestamp of
// the ledger row, the revision it recorded, and the score joined at that stamp
// (nil when that stamp carried no score). The series is what makes a score
// readable as a curve rather than only its endpoints.
type TrendPoint struct {
	TS    string   `json:"ts"`
	Rev   int      `json:"rev"`
	Score *float64 `json:"score,omitempty"`
}

// ModuleTrend summarizes how one module moved across the ledger window: its
// first and last stamped revision (and score, when scores were joined), the
// deltas between those bounds, how many times it was stamped, and the
// timestamps that bound its window. It is derived purely from the append-only
// module-versions ledger — no git, no working tree — so it answers "what has
// been churning over this window?" from what was actually recorded.
type ModuleTrend struct {
	Module     string   `json:"module"`
	Kind       string   `json:"kind"`
	Stamps     int      `json:"stamps"` // ledger rows seen for this module IN the window
	FirstTS    string   `json:"first_ts"`
	LastTS     string   `json:"last_ts"`
	FirstRev   int      `json:"first_rev"`
	LastRev    int      `json:"last_rev"`
	RevDelta   int      `json:"rev_delta"` // LastRev - FirstRev (commits added over the window)
	FirstScore *float64 `json:"first_score,omitempty"`
	LastScore  *float64 `json:"last_score,omitempty"`
	ScoreDelta *float64 `json:"score_delta,omitempty"` // set only when both bounds carry a score
	LastCommit string   `json:"last_commit"`
	// RevsPerWeek is the rev velocity across the module's OWN observed span:
	// RevDelta divided by the weeks between FirstTS and LastTS, rounded to two
	// decimals. Zero when the module did not grow or when both bounds are the
	// same stamp — one observation cannot witness a rate.
	RevsPerWeek float64 `json:"revs_per_week"`
	// Dormant is true when the module added no revisions over the window
	// (RevDelta <= 0). Movers and dormants partition the report: every module
	// is in exactly one of the two lists.
	Dormant bool `json:"dormant"`
	// Series is the module's stamps in timestamp order — the score/rev curve.
	// Under a --since bound it opens with the baseline stamp that set the
	// module's state as of the bound, so the series always spans FirstTS..LastTS.
	Series []TrendPoint `json:"series,omitempty"`
}

// TrendReport is the ledger-derived movement across every module seen.
type TrendReport struct {
	Rows    int           `json:"rows"`   // total parseable ledger rows
	Window  [2]string     `json:"window"` // [earliest ts, latest ts] across in-window rows ("" when empty)
	Modules []ModuleTrend `json:"modules"`
	// Since is the RFC3339 lower bound the fold applied ("" = the whole
	// ledger). Rows still counts every parseable row, so the header reports the
	// ledger that was asked, not just the slice that answered.
	Since string `json:"since,omitempty"`
}

// Trend folds an append-only module-versions ledger (the JSONL that
// `fak version modules --stamp` writes) into a per-module movement summary.
// For each module it keeps the earliest- and latest-timestamped rows as the
// window bounds; timestamps are RFC3339, which sorts lexically, so the ledger
// need not already be ordered. Unparseable lines are skipped, not fatal — an
// append-only ledger a fleet writes will have scars. The returned modules are
// sorted by revision delta descending (the most-churned lead), ties broken by
// name, so the default output already answers "what moved most".
func Trend(ledger []byte) TrendReport { return TrendSince(ledger, "") }

// TrendSince is Trend bounded below by `since` (RFC3339, UTC — run a caller's
// input through NormalizeSince first): only stamps at or after the bound count
// as movement. The bound does NOT simply drop the older rows. The ledger is
// delta-encoded — `fak version modules --stamp` writes a row only when a module
// actually moved — so a module with a single in-window row has still grown, by
// exactly the distance from the last stamp at or before the bound. That earlier
// stamp is therefore kept as the module's BASELINE and becomes FirstRev/FirstTS,
// which is what makes RevDelta read "revisions added since T" rather than
// "revisions between the two rows that happened to land inside the window".
//
// A module with no in-window stamp at all is not dropped either: it is the
// dormant case the bound exists to surface, reported at its baseline with
// Stamps 0 and RevDelta 0.
func TrendSince(ledger []byte, since string) TrendReport {
	rep := TrendReport{Since: since}
	byModule := map[string][]LedgerRow{}
	var order []string
	for _, row := range parseLedgerRows(ledger) {
		rep.Rows++
		if row.TS >= since { // since == "" admits every row
			if rep.Window[0] == "" || row.TS < rep.Window[0] {
				rep.Window[0] = row.TS
			}
			if row.TS > rep.Window[1] {
				rep.Window[1] = row.TS
			}
		}
		if _, ok := byModule[row.Module]; !ok {
			order = append(order, row.Module)
		}
		byModule[row.Module] = append(byModule[row.Module], row)
	}
	for _, name := range order {
		rows := byModule[name]
		// RFC3339 sorts lexically, so the ledger need not already be ordered;
		// a stable sort keeps file order among same-timestamp rows.
		sort.SliceStable(rows, func(i, j int) bool { return rows[i].TS < rows[j].TS })
		split := 0
		for split < len(rows) && rows[split].TS < since {
			split++
		}
		first, last := rows[len(rows)-1], rows[len(rows)-1] // all-before-bound: both are the baseline
		if split < len(rows) {
			first, last = rows[split], rows[len(rows)-1]
			if split > 0 {
				first = rows[split-1] // baseline: the module's state as of the bound
			}
		}
		seriesFrom := split
		if seriesFrom > 0 {
			seriesFrom--
		}
		series := make([]TrendPoint, 0, len(rows)-seriesFrom)
		for _, r := range rows[seriesFrom:] {
			series = append(series, TrendPoint{TS: r.TS, Rev: r.Rev, Score: r.Score})
		}
		mt := ModuleTrend{
			Module:     name,
			Kind:       last.Kind,
			Stamps:     len(rows) - split,
			FirstTS:    first.TS,
			LastTS:     last.TS,
			FirstRev:   first.Rev,
			LastRev:    last.Rev,
			RevDelta:   last.Rev - first.Rev,
			FirstScore: first.Score,
			LastScore:  last.Score,
			LastCommit: last.LastCommit,
			Series:     series,
		}
		if first.Score != nil && last.Score != nil {
			d := *last.Score - *first.Score
			mt.ScoreDelta = &d
		}
		mt.Dormant = mt.RevDelta <= 0
		mt.RevsPerWeek = revsPerWeek(mt.FirstTS, mt.LastTS, mt.RevDelta)
		rep.Modules = append(rep.Modules, mt)
	}
	sortTrend(rep.Modules, "delta")
	return rep
}

// revsPerWeek is the rev velocity in revisions per week over the span the delta
// was actually observed across. Anything it cannot measure — a module that did
// not grow, a zero-length span, an unparseable timestamp — is 0 rather than an
// invented rate or an infinity. The result is rounded to two decimals so the
// human table, the JSON, and a fixture test all agree on one number.
func revsPerWeek(firstTS, lastTS string, delta int) float64 {
	if delta <= 0 {
		return 0
	}
	from, err := time.Parse(time.RFC3339, firstTS)
	if err != nil {
		return 0
	}
	to, err := time.Parse(time.RFC3339, lastTS)
	if err != nil {
		return 0
	}
	weeks := to.Sub(from).Hours() / (24 * 7)
	if weeks <= 0 {
		return 0
	}
	return math.Round(float64(delta)/weeks*100) / 100
}

// NormalizeSince validates a --since bound and returns it in the ledger's own
// UTC RFC3339 shape, which is what makes the fold's lexical timestamp
// comparison sound: comparing a "+02:00" offset against the ledger's "Z" text
// character by character would silently mis-bound the window. A bare
// YYYY-MM-DD reads as that day's midnight UTC; "" passes through as "no bound".
func NormalizeSince(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	if day, err := time.Parse("2006-01-02", s); err == nil {
		return day.UTC().Format(time.RFC3339), nil
	}
	ts, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return "", fmt.Errorf("modver: bad since %q (want RFC3339 or YYYY-MM-DD)", s)
	}
	return ts.UTC().Format(time.RFC3339), nil
}

// Select returns a filtered/sorted/truncated view of the trend: keep modules
// whose name has the `only` prefix, sort by sortKey ("" or "delta" | "rev" |
// "name"), then cap to the first `top` (top <= 0 = all). Like Report.View it
// never mutates the receiver and rejects an unknown sort key rather than
// silently falling back. Rows and Window are preserved so the header still
// reports the full ledger even when the module list is filtered.
func (t TrendReport) Select(only, sortKey string, top int) (TrendReport, error) {
	switch sortKey {
	case "", "delta", "rev", "velocity", "name":
	default:
		return t, fmt.Errorf("modver: bad sort %q (want delta|rev|velocity|name)", sortKey)
	}
	mods := make([]ModuleTrend, 0, len(t.Modules))
	for _, m := range t.Modules {
		if only == "" || strings.HasPrefix(m.Module, only) {
			mods = append(mods, m)
		}
	}
	sortTrend(mods, sortKey)
	if top > 0 && top < len(mods) {
		mods = mods[:top]
	}
	out := t
	out.Modules = mods
	return out, nil
}

// SelectModule narrows the report to a single module by exact name — the
// --module focus. An unknown name yields an empty module list rather than an
// error: asking a ledger about a module it never stamped is a legitimate empty
// answer, and the preserved Rows/Window header still says which ledger was
// asked. An empty name is a no-op.
func (t TrendReport) SelectModule(name string) TrendReport {
	if name == "" {
		return t
	}
	out := t
	out.Modules = nil
	for _, m := range t.Modules {
		if m.Module == name {
			out.Modules = append(out.Modules, m)
		}
	}
	return out
}

// TopMovers returns the report's movers — the modules that added revisions over
// the window — fastest first by revs/week, capped at `top` (<= 0 = all).
func (t TrendReport) TopMovers(top int) []ModuleTrend {
	out := make([]ModuleTrend, 0, len(t.Modules))
	for _, m := range t.Modules {
		if !m.Dormant {
			out = append(out, m)
		}
	}
	sortTrend(out, "velocity")
	if top > 0 && top < len(out) {
		out = out[:top]
	}
	return out
}

// DormantModules returns the modules that added no revisions over the window —
// the dormant list — stalest first (oldest last stamp), capped at `top` (<= 0 =
// all). Together with TopMovers this partitions the report.
func (t TrendReport) DormantModules(top int) []ModuleTrend {
	out := make([]ModuleTrend, 0, len(t.Modules))
	for _, m := range t.Modules {
		if m.Dormant {
			out = append(out, m)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].LastTS != out[j].LastTS {
			return out[i].LastTS < out[j].LastTS
		}
		return out[i].Module < out[j].Module
	})
	if top > 0 && top < len(out) {
		out = out[:top]
	}
	return out
}

// Cap narrows the report to the top `top` movers and the top `top` dormant
// modules, re-laying Modules as movers-then-dormants; top <= 0 is a no-op.
// Capping HERE rather than inside Select is what lets --top mean "N per
// section": one combined truncation under the default rev-delta sort would drop
// every dormant module before the reader ever saw the dormant list.
func (t TrendReport) Cap(top int) TrendReport {
	if top <= 0 {
		return t
	}
	out := t
	out.Modules = append(t.TopMovers(top), t.DormantModules(top)...)
	return out
}

// sortTrend orders trends in place by the chosen key. "delta", "rev" and
// "velocity" are DESCENDING (biggest movement / highest revision / fastest
// first); "name" is ascending. Every tie is broken by module name so the order
// is total and deterministic.
func sortTrend(mods []ModuleTrend, sortKey string) {
	byName := func(i, j int) bool { return mods[i].Module < mods[j].Module }
	switch sortKey {
	case "velocity":
		sort.SliceStable(mods, func(i, j int) bool {
			if mods[i].RevsPerWeek != mods[j].RevsPerWeek {
				return mods[i].RevsPerWeek > mods[j].RevsPerWeek
			}
			if mods[i].RevDelta != mods[j].RevDelta {
				return mods[i].RevDelta > mods[j].RevDelta
			}
			return byName(i, j)
		})
	case "rev":
		sort.SliceStable(mods, func(i, j int) bool {
			if mods[i].LastRev != mods[j].LastRev {
				return mods[i].LastRev > mods[j].LastRev
			}
			return byName(i, j)
		})
	case "name":
		sort.SliceStable(mods, byName)
	default: // "" or "delta"
		sort.SliceStable(mods, func(i, j int) bool {
			if mods[i].RevDelta != mods[j].RevDelta {
				return mods[i].RevDelta > mods[j].RevDelta
			}
			return byName(i, j)
		})
	}
}
