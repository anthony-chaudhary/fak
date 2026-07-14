package gatewayusageledger

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// compaction_segments.go — the compaction-effectiveness fold, SEGMENTED by budget
// regime × session-length band.
//
// It exists because the fleet's compaction-shed numbers are only comparable WITHIN a
// budget regime. The interactive default (gateway.DefaultCompactHistoryBudget, 48000)
// and the headless-worker default (gateway.HeadlessCompactHistoryBudget, 96000) fire
// against structurally different resident shapes: a 96k-budget worker's 30-80-turn
// band legitimately sits under budget and bails `under_budget`, where the same session
// under 48k would have shed heavily. Blend the two regimes into one shed-fraction and a
// deliberate budget change reads as a compaction REGRESSION — the exact false alarm
// this fold prevents by keeping each regime in its own row.
//
// Two more traps the fold closes structurally:
//   - A per-session shed fraction needs shed/(shed+cached+input). Some rows carry
//     shed>0 but cached+input==0 (provider usage never populated on that path); a naive
//     ratio reads a phantom 100%. Those rows are QUARANTINED (counted, never folded into
//     a shed percentile) so one class of unpopulated rows cannot inflate the median.
//   - The fold reads whichever ledger the caller passes; the CLI defaults to the LIVE
//     DefaultLedgerRel, not the committed docs mirror, so a stale published copy cannot
//     masquerade as a recent cliff.
//
// The fold is PURE (rows in, report out) so a golden test can pin it, mirroring the
// cachevaluereport.FoldShapes / RenderShapes pair.

// compactionLengthBand is one session-length band keyed on Counters.ObservedTurns. The
// bands mirror the observed session-length distribution the compaction budget is
// calibrated against (internal/gateway: median ~7 turns, p90 ~52): the sub-40 mass is
// the short-session majority, and 40+ is where a budget change actually bites.
type compactionLengthBand struct {
	label string
	lo    uint64
	hi    uint64 // exclusive; 0 == +inf
}

var compactionLengthBands = []compactionLengthBand{
	{"0-20", 0, 20},
	{"20-40", 20, 40},
	{"40-80", 40, 80},
	{"80-160", 80, 160},
	{"160+", 160, 0},
}

func compactionBandFor(turns uint64) string {
	for _, b := range compactionLengthBands {
		if turns >= b.lo && (b.hi == 0 || turns < b.hi) {
			return b.label
		}
	}
	return "?"
}

// compactionBudgetRegimeLabel names a compact_history_budget so the regime, not just the
// number, is legible. The two live defaults are cross-referenced from internal/gateway
// (DefaultCompactHistoryBudget=48000 interactive, HeadlessCompactHistoryBudget=96000
// headless); this package is a leaf that imports neither, so the mapping is spelled out
// here rather than referenced, and any other value renders as its own regime rather than
// being force-fit into a named bucket.
func compactionBudgetRegimeLabel(budget int) string {
	switch budget {
	case 0:
		return "off"
	case 48000:
		return "interactive"
	case 96000:
		return "headless"
	default:
		return fmt.Sprintf("custom(%d)", budget)
	}
}

// CompactionSegment is the folded compaction stats for one (budget × length-band) cell.
// ShedPctMedian/Mean are over the segment's VALID-denominator fired rows only; a segment
// can have Sessions>0 yet an absent (NaN-free) percentile if every row bailed or every
// row was quarantined — DenomZeroRows and FiredSessions tell those two zeroes apart.
type CompactionSegment struct {
	// Period is the time bucket this segment falls in when the fold is bucketed by
	// --by day|week (day = "2006-01-02", week = ISO "2006-Www"); empty for the default
	// un-bucketed single-window fold, so the existing render/JSON stays byte-for-byte.
	Period        string  `json:"period,omitempty"`
	Budget        int     `json:"budget"`
	BudgetRegime  string  `json:"budget_regime"`
	Band          string  `json:"band"`
	Sessions      int     `json:"sessions"`       // exit rows in this cell
	FiredSessions int     `json:"fired_sessions"` // rows with CompactionFired>0
	Fires         uint64  `json:"fires"`          // sum CompactionFired
	Bails         uint64  `json:"bails"`          // sum CompactionBailed
	BailRate      float64 `json:"bail_rate"`      // Bails / (Fires+Bails); 0 when neither
	ShedTokens    uint64  `json:"shed_tokens"`    // sum CompactionShedTokens

	// ValidDenomRows is the count of rows usable for a shed fraction (fired>0 AND
	// cached+input>0); ShedPctMedian/Mean fold over exactly these. DenomZeroRows is the
	// quarantined phantom-100% class (shed>0 but cached+input==0), reported but never
	// folded into the percentile.
	ValidDenomRows int     `json:"valid_denom_rows"`
	DenomZeroRows  int     `json:"denom_zero_rows"`
	ShedPctMedian  float64 `json:"shed_pct_median"`
	ShedPctMean    float64 `json:"shed_pct_mean"`

	// TopBailReason is the most common CompactReason across the cell's bailed attempts
	// (empty when nothing bailed), the WHY behind a low shed slice.
	TopBailReason string `json:"top_bail_reason,omitempty"`

	// TopBailShare is the top reason's fraction of all CLASSIFIED bails in the cell
	// (0 when nothing bailed). It separates a bail slice that is ONE dominant reason —
	// e.g. under_budget·0.89, a headless band correctly declining under a right-sized
	// budget — from a split mix where a second reason (burst_unprofitable, a tuning
	// call) is quietly eating fires. TopBailReason alone cannot tell those apart: both
	// render the same label, so a real tuning regression hides behind a correct-by-design
	// one until the mix is read.
	TopBailShare float64 `json:"top_bail_share,omitempty"`

	// BailReasons is the full per-reason bail breakdown (the closed agent.CompactReason*
	// vocabulary) for this cell, surfaced so a consumer reads the WHY behind a low shed
	// slice without re-folding the ledger — the durable twin of TopBailReason/Share the
	// metrics exposition projects per (regime × band × reason). nil/omitted when nothing
	// bailed, so a clean cell stays terse.
	BailReasons map[string]uint64 `json:"bail_reasons,omitempty"`

	bailReasons map[string]uint64
	fracs       []float64
}

// CompactionReport is the whole segmentation, plus corpus-level totals so a reader can
// tell "present but zero" from "no rows".
type CompactionReport struct {
	Since           string              `json:"since,omitempty"`
	ExitRows        int                 `json:"exit_rows"`        // exit rows folded (non-exit ignored)
	QuarantinedRows int                 `json:"quarantined_rows"` // total denom-zero-but-shed rows across all cells
	Segments        []CompactionSegment `json:"segments"`
}

// FoldCompaction folds the ledger rows into the (budget × length-band) segmentation over
// the whole --since window. It is FoldCompactionByPeriod with no time bucketing, kept as
// the stable entry point the metrics exposition and the default CLI path call: the report
// it returns is byte-for-byte what it always was (every segment's Period is empty).
func FoldCompaction(rows []Row, since string) CompactionReport {
	return FoldCompactionByPeriod(rows, since, "")
}

// FoldCompactionByPeriod is FoldCompaction with an optional time axis. granularity "" (or
// "none") keeps the single-window fold; "day" buckets each cell by GeneratedAt's calendar
// day, "week" by its ISO week. Bucketing turns the point-in-time regime×band table into a
// trend — the shape that answers "did shed% move WITHIN a regime recently?", which the
// un-bucketed fold structurally cannot (it collapses the whole window into one number per
// cell). Only Kind=="exit" rows are folded (periodic/carryforward snapshots would
// double-count a live session). Rows are assumed already --since filtered by the caller;
// the fold does not re-read the clock, keeping it pure. Segments are returned sorted by
// period, then budget ascending, then the canonical band order.
func FoldCompactionByPeriod(rows []Row, since, granularity string) CompactionReport {
	rep := CompactionReport{Since: since}
	cells := map[[3]string]*CompactionSegment{}
	// keyFor preserves a total, stable ordering for the later sort.
	keyFor := func(period string, budget int, band string) [3]string {
		return [3]string{period, fmt.Sprintf("%012d", budget), band}
	}

	for _, r := range rows {
		if r.Kind != "exit" {
			continue
		}
		rep.ExitRows++
		budget := 0
		if r.Provenance != nil {
			budget = r.Provenance.CompactHistoryBudget
		}
		band := compactionBandFor(r.Counters.ObservedTurns)
		period := compactionPeriodKey(r.GeneratedAt, granularity)
		k := keyFor(period, budget, band)
		seg := cells[k]
		if seg == nil {
			seg = &CompactionSegment{
				Period:       period,
				Budget:       budget,
				BudgetRegime: compactionBudgetRegimeLabel(budget),
				Band:         band,
				bailReasons:  map[string]uint64{},
			}
			cells[k] = seg
		}
		c := r.Counters
		seg.Sessions++
		seg.Fires += c.CompactionFired
		seg.Bails += c.CompactionBailed
		seg.ShedTokens += c.CompactionShedTokens
		for reason, n := range c.CompactionBailReasons {
			seg.bailReasons[reason] += n
		}
		if c.CompactionFired > 0 {
			seg.FiredSessions++
			denom := c.CompactionShedTokens + c.CachedPromptTokens + c.InputTokens
			if c.CachedPromptTokens+c.InputTokens > 0 {
				seg.ValidDenomRows++
				seg.fracs = append(seg.fracs, float64(c.CompactionShedTokens)/float64(denom))
			} else if c.CompactionShedTokens > 0 {
				seg.DenomZeroRows++
				rep.QuarantinedRows++
			}
		}
	}

	for _, seg := range cells {
		if tot := seg.Fires + seg.Bails; tot > 0 {
			seg.BailRate = float64(seg.Bails) / float64(tot)
		}
		seg.ShedPctMedian = medianPct(seg.fracs)
		seg.ShedPctMean = meanPct(seg.fracs)
		seg.TopBailReason, seg.TopBailShare = topReasonWithShare(seg.bailReasons)
		if len(seg.bailReasons) > 0 {
			seg.BailReasons = seg.bailReasons
		}
		rep.Segments = append(rep.Segments, *seg)
	}
	sort.Slice(rep.Segments, func(i, j int) bool {
		if rep.Segments[i].Period != rep.Segments[j].Period {
			return rep.Segments[i].Period < rep.Segments[j].Period
		}
		if rep.Segments[i].Budget != rep.Segments[j].Budget {
			return rep.Segments[i].Budget < rep.Segments[j].Budget
		}
		return bandOrder(rep.Segments[i].Band) < bandOrder(rep.Segments[j].Band)
	})
	return rep
}

// compactionPeriodKey derives the time-bucket label for a row's GeneratedAt under the
// requested granularity. "" / "none" (no bucketing) returns "" so the default fold keeps a
// single window and the byte-for-byte render is preserved. "day" returns the calendar day
// (2006-01-02), "week" the ISO week (2006-Www). A GeneratedAt that will not parse falls
// back to its leading 10 chars (day) or "unknown" (week), so a malformed timestamp lands in
// its own visible bucket rather than crashing the fold or silently joining a real period.
func compactionPeriodKey(generatedAt, granularity string) string {
	switch granularity {
	case "", "none":
		return ""
	case "day":
		if t, err := time.Parse(time.RFC3339, generatedAt); err == nil {
			return t.UTC().Format("2006-01-02")
		}
		if len(generatedAt) >= 10 {
			return generatedAt[:10]
		}
		return "unknown"
	case "week":
		if t, err := time.Parse(time.RFC3339, generatedAt); err == nil {
			y, w := t.UTC().ISOWeek()
			return fmt.Sprintf("%04d-W%02d", y, w)
		}
		return "unknown"
	default:
		return ""
	}
}

func bandOrder(band string) int {
	for i, b := range compactionLengthBands {
		if b.label == band {
			return i
		}
	}
	return len(compactionLengthBands)
}

// medianPct returns the median of fracs as a percentage (0 when empty). Callers hold
// only finite fractions in [0,1], so no NaN can leak into the report.
func medianPct(fracs []float64) float64 {
	if len(fracs) == 0 {
		return 0
	}
	s := append([]float64(nil), fracs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2] * 100
	}
	return (s[n/2-1] + s[n/2]) / 2 * 100
}

func meanPct(fracs []float64) float64 {
	if len(fracs) == 0 {
		return 0
	}
	sum := 0.0
	for _, f := range fracs {
		sum += f
	}
	return sum / float64(len(fracs)) * 100
}

// topReasonWithShare returns the most common reason AND its fraction of all classified
// bails in the map (the total of every reason's count). The share is what turns the bare
// label into a health read: under_budget at 0.89 is a headless band correctly declining,
// while under_budget at 0.51 means a second reason is eating nearly half the attempts —
// same TopBailReason, different diagnosis. Keys are sorted for a deterministic tie-break
// so the render is stable. Empty map → ("", 0).
func topReasonWithShare(m map[string]uint64) (string, float64) {
	best, bestN, total := "", uint64(0), uint64(0)
	// Sort keys for a deterministic tie-break so the render is stable.
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
		total += m[k]
	}
	sort.Strings(keys)
	for _, k := range keys {
		if m[k] > bestN {
			best, bestN = k, m[k]
		}
	}
	if total == 0 {
		return "", 0
	}
	return best, float64(bestN) / float64(total)
}

// RenderCompaction renders the report as a terse, deterministic table grouped by budget
// regime. It is PURE (report in, text out). A shed percentile is shown only when the cell
// has valid-denominator fired rows; otherwise a dash marks an honest absence, with the
// fire/bail columns explaining WHY (all-bail vs never-fired vs quarantined).
func RenderCompaction(rep CompactionReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "compaction by budget regime × session length")
	if rep.Since != "" {
		fmt.Fprintf(&b, " (since %s)", rep.Since)
	}
	b.WriteByte('\n')
	if rep.ExitRows == 0 {
		b.WriteString("  (no exit rows — is the ledger path right, and is it the LIVE .fak/nightrun copy?)\n")
		return b.String()
	}
	fmt.Fprintf(&b, "  %d exit rows folded", rep.ExitRows)
	if rep.QuarantinedRows > 0 {
		fmt.Fprintf(&b, "; %d quarantined (shed>0 but cached+input==0 — excluded from shed%%)", rep.QuarantinedRows)
	}
	b.WriteString("\n\n")

	header := fmt.Sprintf("  %-19s %-8s %5s %6s %6s %6s %8s  %-9s  %s",
		"regime", "band", "sess", "fired", "fires", "bails", "shed%med", "bailrate", "top_bail")
	b.WriteString(header + "\n")

	const noPeriod = "\x00" // sentinel distinct from a real "" (default un-bucketed) period
	lastPeriod := noPeriod
	lastBudget := -1
	for _, s := range rep.Segments {
		if s.Period != lastPeriod {
			if lastPeriod != noPeriod {
				b.WriteByte('\n')
			}
			if s.Period != "" {
				fmt.Fprintf(&b, "  [%s]\n", s.Period)
			}
			lastPeriod = s.Period
			lastBudget = -1
		}
		if s.Budget != lastBudget {
			if lastBudget != -1 {
				b.WriteByte('\n')
			}
			lastBudget = s.Budget
		}
		shed := "     -  "
		if s.ValidDenomRows > 0 {
			shed = fmt.Sprintf("%7.1f%%", s.ShedPctMedian)
		}
		// top_bail carries the reason's SHARE of classified bails, not just the label, so a
		// reader tells a dominant correct-by-design decline (under_budget·89%) from a split
		// mix (under_budget·51%) where a tuning-driven reason is quietly eating fires.
		topBail := s.TopBailReason
		if topBail != "" {
			topBail = fmt.Sprintf("%s·%.0f%%", topBail, s.TopBailShare*100)
		}
		regime := fmt.Sprintf("%s(%d)", s.BudgetRegime, s.Budget)
		fmt.Fprintf(&b, "  %-19s %-8s %5d %6d %6d %6d %8s  %8.2f  %s\n",
			regime, s.Band, s.Sessions, s.FiredSessions, s.Fires, s.Bails, shed, s.BailRate, topBail)
	}
	return b.String()
}
