package modver

// weekly.go folds the append-only module-versions ledger into a WEEKLY
// module-growth digest — the version-everything spine's contribution to the
// trend surfaces operators already read. It rides internal/trendreport (the
// generic report-envelope substrate the cadence / milestone reports embed): the
// digest embeds trendreport.Envelope, ranks movers with trendreport.DirectionWord,
// and reconciles its OK/verdict through trendreport.AdvisoryGate, so a fourth
// trend surface is authored without a second envelope. The fold is PURE and
// deterministic — ledger bytes + the caller's live-module set + a `now` in, a
// digest out; no git, no clock, no I/O — exactly like Trend and the cadence fold.

import (
	"bytes"
	"fmt"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trendreport"
)

// WeeklyDigestSchema versions the weekly module-growth digest envelope so a
// downstream reader can pin it, mirroring the other trend-report schemas.
const WeeklyDigestSchema = "fak-module-growth-weekly-digest/1"

// DefaultWeeklyWindowDays is the trailing window the digest reports over.
const DefaultWeeklyWindowDays = 7

// weeklyUnmeasured is the AdvisoryGate finding token for a digest that could not
// be measured: the ledger held no parseable rows at all, so there is nothing to
// digest. Every other finding (even a quiet week with zero movers) is a measured
// tick — a trend report is a MIRROR, not a second quality gate.
const weeklyUnmeasured = "module_growth_unmeasured"

// MoverRow is one module's movement across the weekly window: its revision (and,
// when scored, score) at the window's open and close, and the signed deltas
// between them. StartRev is the module's rev as of the last ledger row BEFORE the
// window opened; EndRev its rev as of the last row within the window — so RevDelta
// is the commits the module actually added this week, not its all-time rev.
type MoverRow struct {
	Module     string   `json:"module"`
	Kind       string   `json:"kind"`
	StartRev   int      `json:"start_rev"`
	EndRev     int      `json:"end_rev"`
	RevDelta   int      `json:"rev_delta"`
	Direction  string   `json:"direction"` // up|down|flat, via trendreport.DirectionWord
	StartScore *float64 `json:"start_score,omitempty"`
	EndScore   *float64 `json:"end_score,omitempty"`
	ScoreDelta *float64 `json:"score_delta,omitempty"` // set only when both bounds carry a score
	LastCommit string   `json:"last_commit"`
}

// WeeklyDigest is the weekly module-growth digest: which modules grew, which were
// born, which retired, and which moved on score over the trailing window. It
// embeds trendreport.Envelope, so the JSON envelope matches the cadence/milestone
// reports the operator already reads.
type WeeklyDigest struct {
	trendreport.Envelope
	WindowDays  int        `json:"window_days"`
	WindowStart string     `json:"window_start"` // RFC3339, inclusive
	WindowEnd   string     `json:"window_end"`   // RFC3339 (`now`), inclusive
	LedgerRows  int        `json:"ledger_rows"`  // parseable rows whose ts falls in the window
	TopMovers   []MoverRow `json:"top_movers"`   // pre-existing modules whose rev moved this week, biggest first
	ScoreMovers []MoverRow `json:"score_movers"` // pre-existing modules whose score moved this week
	NewModules  []string   `json:"new_modules"`  // modules whose first-ever ledger row falls in the window
	Deaths      []string   `json:"deaths"`       // ledger-seen modules absent from the live set (nil when live is unknown)
}

// weeklyAgg accumulates one module's ledger rows into the window bounds the digest
// needs: the last row strictly before the window (the baseline), the last row at
// or before `now` (the window close), whether any row landed inside the window,
// and the earliest timestamp ever seen (to decide a birth).
type weeklyAgg struct {
	kind      string
	firstTS   string
	hasBase   bool
	baseTS    string
	baseRev   int
	baseScore *float64
	hasEnd    bool
	endTS     string
	endRev    int
	endScore  *float64
	endCommit string
	inWindow  bool
}

// FoldWeekly folds an append-only module-versions ledger (the JSONL that
// `fak version modules --stamp` writes) into the weekly module-growth digest.
// Window membership comes from each row's own ts, which is RFC3339 and therefore
// sorts lexically, so the ledger need not already be ordered: the window is the 7
// days ending at `now` (inclusive of `now`, exclusive of anything after it).
//
// `live` is the set of modules that exist at HEAD — pass modver.Snapshot's module
// names. Deaths (a module the ledger has seen but that no longer exists at HEAD)
// are derivable only against this latest snapshot, never from a delta-only ledger
// alone; when `live` is nil the digest reports no deaths rather than guessing.
// Unparseable lines are skipped, not fatal — an append-only ledger a fleet writes
// will have scars.
func FoldWeekly(ledger []byte, live map[string]bool, now time.Time) WeeklyDigest {
	nowUTC := now.UTC()
	startUTC := nowUTC.AddDate(0, 0, -DefaultWeeklyWindowDays)
	startStr := startUTC.Format(time.RFC3339)
	nowStr := nowUTC.Format(time.RFC3339)

	d := WeeklyDigest{
		Envelope:    trendreport.Stamp(WeeklyDigestSchema, trendreport.Opts{GeneratedAt: nowStr, Date: nowUTC.Format("2006-01-02")}),
		WindowDays:  DefaultWeeklyWindowDays,
		WindowStart: startStr,
		WindowEnd:   nowStr,
	}

	agg := map[string]*weeklyAgg{}
	totalParsed := 0
	for _, row := range parseLedgerRows(ledger) {
		totalParsed++
		if row.TS >= startStr && row.TS <= nowStr {
			d.LedgerRows++
		}
		a := agg[row.Module]
		if a == nil {
			a = &weeklyAgg{firstTS: row.TS}
			agg[row.Module] = a
		}
		if row.TS < a.firstTS {
			a.firstTS = row.TS
		}
		if row.TS < startStr && (!a.hasBase || row.TS > a.baseTS) {
			a.hasBase, a.baseTS, a.baseRev, a.baseScore = true, row.TS, row.Rev, row.Score
		}
		if row.TS <= nowStr && (!a.hasEnd || row.TS >= a.endTS) {
			a.hasEnd, a.endTS, a.endRev, a.endScore, a.endCommit, a.kind = true, row.TS, row.Rev, row.Score, row.LastCommit, row.Kind
		}
		if row.TS >= startStr && row.TS <= nowStr {
			a.inWindow = true
		}
	}

	for name, a := range agg {
		// Born this week: the module's first-ever ledger row landed in the window.
		if a.firstTS >= startStr && a.firstTS <= nowStr {
			d.NewModules = append(d.NewModules, name)
		}
		// Retired: seen in the ledger but gone from the live snapshot (only decidable
		// when the caller supplied the live set).
		if live != nil && !live[name] {
			d.Deaths = append(d.Deaths, name)
		}
		// Movers: a module that existed BEFORE the window (has a baseline) and moved
		// inside it. A born-this-week module has no baseline, so it is reported as new
		// rather than double-counted as a mover of its whole rev count.
		if !a.hasBase || !a.inWindow || !a.hasEnd {
			continue
		}
		revDelta := a.endRev - a.baseRev
		m := MoverRow{
			Module:     name,
			Kind:       a.kind,
			StartRev:   a.baseRev,
			EndRev:     a.endRev,
			RevDelta:   revDelta,
			Direction:  trendreport.DirectionWord(revDelta),
			StartScore: a.baseScore,
			EndScore:   a.endScore,
			LastCommit: a.endCommit,
		}
		if a.baseScore != nil && a.endScore != nil {
			sd := *a.endScore - *a.baseScore
			m.ScoreDelta = &sd
		}
		if revDelta != 0 {
			d.TopMovers = append(d.TopMovers, m)
		}
		if m.ScoreDelta != nil && *m.ScoreDelta != 0 {
			d.ScoreMovers = append(d.ScoreMovers, m)
		}
	}

	sort.Slice(d.TopMovers, func(i, j int) bool {
		if d.TopMovers[i].RevDelta != d.TopMovers[j].RevDelta {
			return d.TopMovers[i].RevDelta > d.TopMovers[j].RevDelta
		}
		return d.TopMovers[i].Module < d.TopMovers[j].Module
	})
	sort.Slice(d.ScoreMovers, func(i, j int) bool {
		si, sj := *d.ScoreMovers[i].ScoreDelta, *d.ScoreMovers[j].ScoreDelta
		if si != sj {
			return si > sj
		}
		return d.ScoreMovers[i].Module < d.ScoreMovers[j].Module
	})
	sort.Strings(d.NewModules)
	sort.Strings(d.Deaths)

	finding, reason, nextAction := d.summarize(totalParsed)
	gate := trendreport.AdvisoryGate("MODULE-GROWTH", finding, reason, weeklyUnmeasured)
	d.OK = gate.Exit == 0
	if gate.Exit == 0 {
		d.Verdict = trendreport.VerdictOK
	} else {
		d.Verdict = trendreport.VerdictAction
	}
	d.Finding = finding
	d.Reason = reason
	d.NextAction = nextAction
	return d
}

// summarize renders the digest's finding/reason/next-action. An empty or
// all-scar ledger is the one INCOMPLETE case (the unmeasured token); every other
// week — including a quiet one with zero movers — is a measured tick.
func (d WeeklyDigest) summarize(totalParsed int) (finding, reason, nextAction string) {
	if totalParsed == 0 {
		return weeklyUnmeasured,
			"the module-versions ledger held no parseable rows; nothing to digest",
			"seed the ledger with `fak version modules --stamp`, then re-check next week"
	}
	reason = fmt.Sprintf("%d moved, %d new, %d retired, %d score moves in the 7 days ending %s",
		len(d.TopMovers), len(d.NewModules), len(d.Deaths), len(d.ScoreMovers), d.Date)
	return "MEASURED — " + reason, reason,
		"read the digest; a burst of new/retired modules or a large rev/score swing is the cue to look"
}

// Render prints the human-readable weekly module-growth section. Movers show the
// window's rev bounds and signed delta; score movers the score swing; new and
// retired modules their names. A quiet week renders a single honest header line.
func (d WeeklyDigest) Render() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "module-growth (week ending %s): %d moved · %d new · %d retired · %d score moves\n",
		d.Date, len(d.TopMovers), len(d.NewModules), len(d.Deaths), len(d.ScoreMovers))
	if d.Verdict == trendreport.VerdictAction {
		fmt.Fprintf(&b, "  INCOMPLETE: %s\n", d.Reason)
		return b.String()
	}
	if len(d.TopMovers) > 0 {
		fmt.Fprintln(&b, "  top movers:")
		for _, m := range d.TopMovers {
			fmt.Fprintf(&b, "    %-24s r%d -> r%d  %s %+d\n", m.Module, m.StartRev, m.EndRev, m.Direction, m.RevDelta)
		}
	}
	if len(d.ScoreMovers) > 0 {
		fmt.Fprintln(&b, "  score movers:")
		for _, m := range d.ScoreMovers {
			fmt.Fprintf(&b, "    %-24s score %g -> %g  (%+g)\n", m.Module, *m.StartScore, *m.EndScore, *m.ScoreDelta)
		}
	}
	if len(d.NewModules) > 0 {
		fmt.Fprintln(&b, "  new modules:")
		for _, name := range d.NewModules {
			fmt.Fprintf(&b, "    %s\n", name)
		}
	}
	if len(d.Deaths) > 0 {
		fmt.Fprintln(&b, "  retired:")
		for _, name := range d.Deaths {
			fmt.Fprintf(&b, "    %s\n", name)
		}
	}
	return b.String()
}
