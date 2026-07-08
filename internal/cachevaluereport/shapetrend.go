package cachevaluereport

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// ShapeTrendSchema versions the by-week session-shape DRIFT envelope independently of
// the static ShapeSchema snapshot and the by-week reuse-trend Schema. FoldShapes answers
// "what shapes do we run and which earn reuse?" as a single all-corpus snapshot;
// FoldShapeTrend answers the LONGITUDINAL complement — "is a shape's share of reused
// tokens drifting week over week?" — the early-regression signal (a shrinking warm×long
// share) the static snapshot cannot show.
//
// #1066 honesty fence rides along exactly as FoldShapes/Fold: outcome bands are cut on
// the WITNESSED realized reuse ratio (reused/prompt), never the forbidden vs-naive
// re-prefill multiple, and the self-labels (PublishableValueFamily,
// VsNaiveMultipleExcluded) are carried on the report.
const ShapeTrendSchema = "fak-cache-value-shape-trend/1"

// ShapeWeekPoint is one shape cluster's reading in one ISO week, plus its drift versus
// the same shape's PREVIOUS week in the chronological series. ShareOfReusedTokens and
// ShareOfSessions are WITHIN-week shares (this shape's tokens/sessions over that week's
// total), so a falling share tells you the shape is losing ground week over week even if
// the absolute reuse held steady.
type ShapeWeekPoint struct {
	Period string `json:"period"` // ISO week, e.g. "2026-W26"
	Start  string `json:"start"`  // earliest row date in the week (YYYY-MM-DD)

	Sessions     uint64 `json:"sessions"`
	Turns        uint64 `json:"turns"`
	PromptTokens uint64 `json:"prompt_tokens"`
	ReusedTokens uint64 `json:"reused_tokens"`

	RealizedReuseRatio  float64 `json:"realized_reuse_ratio"`   // reused/prompt within the shape's week rows
	ShareOfSessions     float64 `json:"share_of_sessions"`      // this shape's sessions / all sessions that week
	ShareOfReusedTokens float64 `json:"share_of_reused_tokens"` // this shape's reused tokens / all reused tokens that week

	Trend                    Trend   `json:"trend"`                        // vs the same shape's prior week
	DeltaShareOfReusedTokens float64 `json:"delta_share_of_reused_tokens"` // this week's share minus prior week's
}

// ShapeSeries is one shape cluster (length × outcome) and its chronological weekly
// series. Latest* mirror the last point so a card can render the headline drift without
// walking the whole series.
type ShapeSeries struct {
	Length  LengthBand  `json:"length"`
	Outcome OutcomeBand `json:"outcome"`
	Health  ShapeHealth `json:"health"`

	Weeks []ShapeWeekPoint `json:"weeks"`

	LatestWeek                     string  `json:"latest_week"`
	LatestTrend                    Trend   `json:"latest_trend"`
	LatestShareOfReusedTokens      float64 `json:"latest_share_of_reused_tokens"`
	LatestDeltaShareOfReusedTokens float64 `json:"latest_delta_share_of_reused_tokens"`
}

// ShapeTrendReport is the rolled-up by-week shape-drift envelope. Series are ordered by
// the same stable (length, outcome) key FoldShapes uses so the render and JSON are
// deterministic.
type ShapeTrendReport struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Since       string `json:"since,omitempty"`
	Granularity string `json:"granularity"` // "week"

	Weeks []string `json:"weeks"` // chronological ISO-week keys covered

	TotalSessions     uint64 `json:"total_sessions"`
	MultiTurnSessions uint64 `json:"multi_turn_sessions"`

	Series []ShapeSeries `json:"series"`

	// LatestWeek headline: which shapes gained/lost reuse share most recently. A shape
	// is "gained" when its latest-week trend is improved, "lost" when regressed; both
	// lists are ordered by the magnitude of the most-recent drift.
	LatestWeek   string   `json:"latest_week,omitempty"`
	LatestGained []string `json:"latest_gained,omitempty"`
	LatestLost   []string `json:"latest_lost,omitempty"`

	// #1066 fence self-labels — a downstream reader can never mistake the realized
	// reuse for the forbidden vs-naive multiple.
	PublishableValueFamily  string `json:"publishable_value_family"`
	VsNaiveMultipleExcluded bool   `json:"vs_naive_multiple_excluded"`

	OK         bool   `json:"ok"`
	Verdict    string `json:"verdict"` // MEASURED | INSUFFICIENT
	Finding    string `json:"finding"`
	Reason     string `json:"reason"`
	NextAction string `json:"next_action"`
}

// shapeLabel renders a (length × outcome) pair as a compact "long×warm" headline token.
func shapeLabel(l LengthBand, o OutcomeBand) string {
	return fmt.Sprintf("%s×%s", l, o)
}

// FoldShapeTrend rolls a slice of ledger rows up into a by-week session-shape DRIFT
// report. It is PURE and deterministic: the only time input is `now`, used solely to
// stamp GeneratedAt — bucketing comes from each row's own Date. Within each ISO week it
// runs the same (length × outcome) clustering as FoldShapes, then trends each shape
// cluster's within-week share of reused tokens across weeks using the same reuseEpsilon
// dead-band as Fold. Rows with zero turns are skipped, the same way Fold/FoldShapes skip
// them; thin weeks fall open INSUFFICIENT exactly as Fold does.
func FoldShapeTrend(rows []cachevalueledger.Row, now time.Time) ShapeTrendReport {
	r := ShapeTrendReport{
		Schema:                  ShapeTrendSchema,
		GeneratedAt:             now.UTC().Format(time.RFC3339),
		Granularity:             "week",
		PublishableValueFamily:  PublishableValueFamily,
		VsNaiveMultipleExcluded: true,
		Verdict:                 "INSUFFICIENT",
		OK:                      true,
	}

	type cell struct {
		length                          LengthBand
		outcome                         OutcomeBand
		sessions, turns, prompt, reused uint64
	}
	type weekAgg struct {
		start         time.Time
		totalSessions uint64
		totalReused   uint64
		cells         map[string]*cell // by shapeKey
	}
	byWeek := map[string]*weekAgg{}

	for _, row := range rows {
		if row.Turns == 0 {
			continue
		}
		d, err := time.Parse("2006-01-02", row.Date)
		if err != nil {
			continue
		}
		wk := isoWeek(d)
		wa := byWeek[wk]
		if wa == nil {
			wa = &weekAgg{start: d, cells: map[string]*cell{}}
			byWeek[wk] = wa
		}
		if d.Before(wa.start) {
			wa.start = d
		}
		l := lengthBand(row.Turns)
		o := outcomeBand(row.Turns, row.ReuseRatio)
		key := shapeKey(l, o)
		c := wa.cells[key]
		if c == nil {
			c = &cell{length: l, outcome: o}
			wa.cells[key] = c
		}
		c.sessions++
		c.turns += row.Turns
		c.prompt += row.PromptTokens
		c.reused += row.ReusedTokens

		wa.totalSessions++
		wa.totalReused += row.ReusedTokens

		r.TotalSessions++
		if row.Turns >= MinShortTurns {
			r.MultiTurnSessions++
		}
	}

	weeks := sortedPeriodKeys(byWeek)
	r.Weeks = weeks
	if n := len(weeks); n > 0 {
		r.LatestWeek = weeks[n-1]
	}

	// Every shape key that appears in any week, ordered by the stable (length, outcome)
	// key so the series and JSON are deterministic.
	shapeKeysSet := map[string]struct{}{}
	for _, wa := range byWeek {
		for k := range wa.cells {
			shapeKeysSet[k] = struct{}{}
		}
	}
	shapeKeys := make([]string, 0, len(shapeKeysSet))
	for k := range shapeKeysSet {
		shapeKeys = append(shapeKeys, k)
	}
	sort.Strings(shapeKeys)

	type drift struct {
		label string
		delta float64
	}
	var gained, lost []drift

	for _, sk := range shapeKeys {
		var series ShapeSeries
		var prev *ShapeWeekPoint
		for _, wk := range weeks {
			wa := byWeek[wk]
			c := wa.cells[sk]
			if c == nil {
				continue // shape absent this week — no point in its chronological series
			}
			if series.Weeks == nil {
				series.Length = c.length
				series.Outcome = c.outcome
				series.Health = classifyHealth(c.length, c.outcome)
			}
			p := ShapeWeekPoint{
				Period:       wk,
				Start:        wa.start.Format("2006-01-02"),
				Sessions:     c.sessions,
				Turns:        c.turns,
				PromptTokens: c.prompt,
				ReusedTokens: c.reused,
			}
			if c.prompt > 0 {
				p.RealizedReuseRatio = float64(c.reused) / float64(c.prompt)
			}
			if wa.totalSessions > 0 {
				p.ShareOfSessions = float64(c.sessions) / float64(wa.totalSessions)
			}
			if wa.totalReused > 0 {
				p.ShareOfReusedTokens = float64(c.reused) / float64(wa.totalReused)
			}
			if prev == nil {
				p.Trend = TrendNew
			} else {
				p.DeltaShareOfReusedTokens = p.ShareOfReusedTokens - prev.ShareOfReusedTokens
				switch {
				case p.DeltaShareOfReusedTokens > reuseEpsilon:
					p.Trend = TrendImproved
				case p.DeltaShareOfReusedTokens < -reuseEpsilon:
					p.Trend = TrendRegressed
				default:
					p.Trend = TrendFlat
				}
			}
			series.Weeks = append(series.Weeks, p)
			prev = &series.Weeks[len(series.Weeks)-1]
		}
		if len(series.Weeks) == 0 {
			continue
		}
		last := series.Weeks[len(series.Weeks)-1]
		series.LatestWeek = last.Period
		series.LatestTrend = last.Trend
		series.LatestShareOfReusedTokens = last.ShareOfReusedTokens
		series.LatestDeltaShareOfReusedTokens = last.DeltaShareOfReusedTokens
		r.Series = append(r.Series, series)

		// Headline: only shapes present in the most recent week count toward gained/lost.
		if last.Period == r.LatestWeek {
			switch last.Trend {
			case TrendImproved:
				gained = append(gained, drift{shapeLabel(series.Length, series.Outcome), last.DeltaShareOfReusedTokens})
			case TrendRegressed:
				lost = append(lost, drift{shapeLabel(series.Length, series.Outcome), last.DeltaShareOfReusedTokens})
			}
		}
	}

	// Order the headline lists by the magnitude of the most-recent drift (largest mover
	// first); the shapeKey-ordered series already broke ties deterministically upstream.
	sort.SliceStable(gained, func(i, j int) bool { return gained[i].delta > gained[j].delta })
	sort.SliceStable(lost, func(i, j int) bool { return lost[i].delta < lost[j].delta })
	for _, d := range gained {
		r.LatestGained = append(r.LatestGained, d.label)
	}
	for _, d := range lost {
		r.LatestLost = append(r.LatestLost, d.label)
	}

	r.fillShapeTrendVerdict()
	return r
}

// fillShapeTrendVerdict sets the report-contract fields. This is a REPORT, not a gate: OK
// stays true; Verdict is INSUFFICIENT only when no multi-turn session exists to give a
// shape any realized reuse to trend (mirroring Fold/FoldShapes falling open on a thin
// corpus).
func (r *ShapeTrendReport) fillShapeTrendVerdict() {
	if r.MultiTurnSessions == 0 {
		r.Verdict = "INSUFFICIENT"
		r.Finding = fmt.Sprintf("%d session(s) across %d week(s), all single-turn; no multi-turn shape to trend reuse drift on yet", r.TotalSessions, len(r.Weeks))
		r.Reason = "realized KV-prefix reuse needs sessions with >= 2 turns; single-turn cold runs have no prior turn to reuse from"
		r.NextAction = "accumulate multi-turn guard/serve sessions into docs/nightrun/cache-value.jsonl, then re-fold"
		return
	}
	r.Verdict = "MEASURED"
	r.Finding = fmt.Sprintf("%d shape series over %d week(s), %d session(s); latest week %s: %d shape(s) gained / %d lost reuse share",
		len(r.Series), len(r.Weeks), r.TotalSessions, r.LatestWeek, len(r.LatestGained), len(r.LatestLost))
	r.Reason = "WITNESSED in-kernel KV-prefix reuse, clustered by session length × realized-reuse outcome and trended week over week; " + PublishableValueFamily
	r.NextAction = "watch which shapes are losing share_of_reused_tokens week over week — a shrinking warm×long share is an early regression signal the static snapshot cannot show"
}

// RenderShapeTrend produces a compact, deterministic terminal table of the per-shape
// weekly drift.
func RenderShapeTrend(r ShapeTrendReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "cache-value session-shape drift (Track 1, WITNESSED kernel reuse) — %s\n", r.Verdict)
	fmt.Fprintf(&sb, "  %s\n", r.Finding)
	fmt.Fprintf(&sb, "  fence: %s\n", PublishableValueFamily)
	if len(r.Series) == 0 {
		return sb.String()
	}
	if r.LatestWeek != "" {
		fmt.Fprintf(&sb, "  latest week %s — gained: [%s]  lost: [%s]\n",
			r.LatestWeek, strings.Join(r.LatestGained, ", "), strings.Join(r.LatestLost, ", "))
	}
	for _, s := range r.Series {
		fmt.Fprintf(&sb, "\n  shape %s×%s (%s)\n", s.Length, s.Outcome, s.Health)
		fmt.Fprintf(&sb, "    %-9s  %10s  %7s  %-10s\n", "week", "reuse-tok%", "sess%", "trend")
		for _, w := range s.Weeks {
			fmt.Fprintf(&sb, "    %-9s  %9.1f%%  %6.1f%%  %-10s\n",
				w.Period, 100*w.ShareOfReusedTokens, 100*w.ShareOfSessions, w.Trend)
		}
	}
	return sb.String()
}
