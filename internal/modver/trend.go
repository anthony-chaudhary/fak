package modver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ModuleTrend summarizes how one module moved across the ledger window: its
// first and last stamped revision (and score, when scores were joined), the
// deltas between those bounds, how many times it was stamped, and the
// timestamps that bound its window. It is derived purely from the append-only
// module-versions ledger — no git, no working tree — so it answers "what has
// been churning over this window?" from what was actually recorded.
type ModuleTrend struct {
	Module     string   `json:"module"`
	Kind       string   `json:"kind"`
	Stamps     int      `json:"stamps"` // ledger rows seen for this module
	FirstTS    string   `json:"first_ts"`
	LastTS     string   `json:"last_ts"`
	FirstRev   int      `json:"first_rev"`
	LastRev    int      `json:"last_rev"`
	RevDelta   int      `json:"rev_delta"` // LastRev - FirstRev (commits added over the window)
	FirstScore *float64 `json:"first_score,omitempty"`
	LastScore  *float64 `json:"last_score,omitempty"`
	ScoreDelta *float64 `json:"score_delta,omitempty"` // set only when both bounds carry a score
	LastCommit string   `json:"last_commit"`
}

// TrendReport is the ledger-derived movement across every module seen.
type TrendReport struct {
	Rows    int           `json:"rows"`   // total parseable ledger rows
	Window  [2]string     `json:"window"` // [earliest ts, latest ts] across all rows ("" when empty)
	Modules []ModuleTrend `json:"modules"`
}

// Trend folds an append-only module-versions ledger (the JSONL that
// `fak version modules --stamp` writes) into a per-module movement summary.
// For each module it keeps the earliest- and latest-timestamped rows as the
// window bounds; timestamps are RFC3339, which sorts lexically, so the ledger
// need not already be ordered. Unparseable lines are skipped, not fatal — an
// append-only ledger a fleet writes will have scars. The returned modules are
// sorted by revision delta descending (the most-churned lead), ties broken by
// name, so the default output already answers "what moved most".
func Trend(ledger []byte) TrendReport {
	type bounds struct {
		first, last LedgerRow
		count       int
	}
	seen := map[string]*bounds{}
	var rep TrendReport
	for _, line := range bytes.Split(ledger, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var row LedgerRow
		if err := json.Unmarshal(line, &row); err != nil || row.Module == "" {
			continue
		}
		rep.Rows++
		if rep.Window[0] == "" || row.TS < rep.Window[0] {
			rep.Window[0] = row.TS
		}
		if row.TS > rep.Window[1] {
			rep.Window[1] = row.TS
		}
		b := seen[row.Module]
		if b == nil {
			seen[row.Module] = &bounds{first: row, last: row, count: 1}
			continue
		}
		b.count++
		if row.TS < b.first.TS {
			b.first = row
		}
		if row.TS >= b.last.TS {
			b.last = row
		}
	}
	for name, b := range seen {
		mt := ModuleTrend{
			Module:     name,
			Kind:       b.last.Kind,
			Stamps:     b.count,
			FirstTS:    b.first.TS,
			LastTS:     b.last.TS,
			FirstRev:   b.first.Rev,
			LastRev:    b.last.Rev,
			RevDelta:   b.last.Rev - b.first.Rev,
			FirstScore: b.first.Score,
			LastScore:  b.last.Score,
			LastCommit: b.last.LastCommit,
		}
		if b.first.Score != nil && b.last.Score != nil {
			d := *b.last.Score - *b.first.Score
			mt.ScoreDelta = &d
		}
		rep.Modules = append(rep.Modules, mt)
	}
	sortTrend(rep.Modules, "delta")
	return rep
}

// Select returns a filtered/sorted/truncated view of the trend: keep modules
// whose name has the `only` prefix, sort by sortKey ("" or "delta" | "rev" |
// "name"), then cap to the first `top` (top <= 0 = all). Like Report.View it
// never mutates the receiver and rejects an unknown sort key rather than
// silently falling back. Rows and Window are preserved so the header still
// reports the full ledger even when the module list is filtered.
func (t TrendReport) Select(only, sortKey string, top int) (TrendReport, error) {
	switch sortKey {
	case "", "delta", "rev", "name":
	default:
		return t, fmt.Errorf("modver: bad sort %q (want delta|rev|name)", sortKey)
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

// sortTrend orders trends in place by the chosen key. "delta" and "rev" are
// DESCENDING (biggest movement / highest revision first); "name" is ascending.
// Every tie is broken by module name so the order is total and deterministic.
func sortTrend(mods []ModuleTrend, sortKey string) {
	byName := func(i, j int) bool { return mods[i].Module < mods[j].Module }
	switch sortKey {
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
