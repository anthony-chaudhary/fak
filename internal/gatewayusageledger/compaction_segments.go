package gatewayusageledger

import (
	"fmt"
	"sort"
	"strings"
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

// FoldCompaction folds the ledger rows into the (budget × length-band) segmentation.
// Only Kind=="exit" rows are folded (periodic/carryforward snapshots would double-count
// a live session). Rows are assumed already --since filtered by the caller; the fold does
// not re-read the clock, keeping it pure. Segments are returned sorted by budget ascending
// then by the canonical band order.
func FoldCompaction(rows []Row, since string) CompactionReport {
	rep := CompactionReport{Since: since}
	cells := map[[2]string]*CompactionSegment{}
	// keyOrder preserves first-seen insertion so the later sort is total and stable.
	keyFor := func(budget int, band string) [2]string {
		return [2]string{fmt.Sprintf("%012d", budget), band}
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
		k := keyFor(budget, band)
		seg := cells[k]
		if seg == nil {
			seg = &CompactionSegment{
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
		seg.TopBailReason = topReason(seg.bailReasons)
		rep.Segments = append(rep.Segments, *seg)
	}
	sort.Slice(rep.Segments, func(i, j int) bool {
		if rep.Segments[i].Budget != rep.Segments[j].Budget {
			return rep.Segments[i].Budget < rep.Segments[j].Budget
		}
		return bandOrder(rep.Segments[i].Band) < bandOrder(rep.Segments[j].Band)
	})
	return rep
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

func topReason(m map[string]uint64) string {
	best, bestN := "", uint64(0)
	// Sort keys for a deterministic tie-break so the render is stable.
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if m[k] > bestN {
			best, bestN = k, m[k]
		}
	}
	return best
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

	lastBudget := -1
	for _, s := range rep.Segments {
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
		regime := fmt.Sprintf("%s(%d)", s.BudgetRegime, s.Budget)
		fmt.Fprintf(&b, "  %-19s %-8s %5d %6d %6d %6d %8s  %8.2f  %s\n",
			regime, s.Band, s.Sessions, s.FiredSessions, s.Fires, s.Bails, shed, s.BailRate, s.TopBailReason)
	}
	return b.String()
}
