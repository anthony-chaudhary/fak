package cachevaluepost

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

// cardTitle is the fixed headline naming the surface and the track it reports. Track 1 is
// the WITNESSED in-kernel reuse; the title states that so a reader never reads the realized
// reuse as the (excluded) vs-naive multiple.
const cardTitle = "fak cache-value — Track 1 (WITNESSED kernel reuse)"

// glyph maps the report verdict to a leading status emoji: MEASURED reads as a chart (a
// real trend to look at), anything thinner reads as an hourglass (accumulating, no page).
func (c Card) glyph() string {
	if c.Report.Verdict == "MEASURED" {
		return ":bar_chart:"
	}
	return ":hourglass_flowing_sand:"
}

// trendArrow renders a bucket's direction as a compact arrow so the channel reads the
// movement at a glance. It mirrors the cachevaluereport.Trend vocabulary.
func trendArrow(t cachevaluereport.Trend) string {
	switch t {
	case cachevaluereport.TrendImproved:
		return "↑ improved"
	case cachevaluereport.TrendRegressed:
		return "↓ regressed"
	case cachevaluereport.TrendFlat:
		return "→ flat"
	default: // TrendNew and the zero value
		return "• new"
	}
}

// sparkChars are the eight block glyphs the reuse sparkline draws with. They render
// identically in Slack mrkdwn, GitHub markdown, and a terminal — the trend-aware form of
// the █/▌ bar idiom the ablation notes use.
var sparkChars = []rune("▁▂▃▄▅▆▇█")

// sparkline maps a series of reuse ratios (0..1) to the block glyphs, normalized against
// the series' own min/max so a flat-but-high series still reads as flat rather than
// maxed-out. An empty series renders "", a single point a mid-level block.
func sparkline(vals []float64) string {
	if len(vals) == 0 {
		return ""
	}
	min, max := vals[0], vals[0]
	for _, v := range vals {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	span := max - min
	var sb strings.Builder
	for _, v := range vals {
		idx := len(sparkChars) / 2 // flat series → mid block
		if span > 0 {
			idx = int((v - min) / span * float64(len(sparkChars)-1))
			if idx < 0 {
				idx = 0
			}
			if idx >= len(sparkChars) {
				idx = len(sparkChars) - 1
			}
		}
		sb.WriteRune(sparkChars[idx])
	}
	return sb.String()
}

// reuseSeries pulls the per-bucket realized reuse ratios in chronological order for the
// sparkline and the first→last delta line.
func (c Card) reuseSeries() []float64 {
	out := make([]float64, 0, len(c.Report.Buckets))
	for _, b := range c.Report.Buckets {
		out = append(out, b.RealizedReuseRatio)
	}
	return out
}

// trendLine renders the sparkline plus the first→last reuse endpoints, the one-glance
// "where did the trend start and end" summary. Returns "" when there is nothing to trend.
func (c Card) trendLine() string {
	series := c.reuseSeries()
	if len(series) == 0 {
		return ""
	}
	spark := sparkline(series)
	first, last := series[0], series[len(series)-1]
	if len(series) == 1 {
		return fmt.Sprintf("trend %s  %.1f%%", spark, 100*last)
	}
	return fmt.Sprintf("trend %s  %.1f%% → %.1f%%", spark, 100*first, 100*last)
}

// bucketLines renders one row per weekly bucket: period, realized reuse, direction, and
// the session/multi-turn counts that back it, flagging a thin bucket so a reader does not
// over-trust a reuse number computed over too few turns.
func (c Card) bucketLines() []string {
	lines := make([]string, 0, len(c.Report.Buckets))
	for _, b := range c.Report.Buckets {
		thin := ""
		if b.Thin {
			thin = " (thin)"
		}
		lines = append(lines, fmt.Sprintf("%s  %5.1f%%  %-11s  sessions %d · m-turns %d%s",
			b.Period, 100*b.RealizedReuseRatio, trendArrow(b.Trend), b.Sessions, b.MultiTurnTurns, thin))
	}
	return lines
}

// fence is the #1066 honesty-fence label the card carries verbatim into the channel.
func (c Card) fence() string {
	return "fence: " + cachevaluereport.PublishableValueFamily
}

// Text renders the plain-text fallback — the line Slack shows in notifications and any
// client without Block Kit. It is also what tests and --dry-run assert on. It leads with
// the verdict + headline, then the finding, the trend sparkline, the per-bucket rows, and
// the honesty fence.
func (c Card) Text() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "%s *%s* — %s", c.glyph(), cardTitle, c.Report.Verdict)
	if f := strings.TrimSpace(c.Report.Finding); f != "" {
		fmt.Fprintf(&sb, "\n%s", f)
	}
	if tl := c.trendLine(); tl != "" {
		fmt.Fprintf(&sb, "\n%s", tl)
	}
	for _, ln := range c.bucketLines() {
		fmt.Fprintf(&sb, "\n• %s", ln)
	}
	if c.Report.Verdict != "MEASURED" {
		if na := strings.TrimSpace(c.Report.NextAction); na != "" {
			fmt.Fprintf(&sb, "\nnext: %s", na)
		}
	}
	fmt.Fprintf(&sb, "\n%s", c.fence())
	if src := strings.TrimSpace(c.Source); src != "" {
		fmt.Fprintf(&sb, "\n_posted by %s_", src)
	}
	return sb.String()
}

// Blocks renders the Block Kit payload for a richer card. It carries the same facts as
// Text so a non-Block client loses nothing: a headline section, the finding, the trend +
// per-bucket body, and a context line with the fence, schema, and source.
func (c Card) Blocks() []any {
	blocks := []any{
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": fmt.Sprintf("*%s %s* — %s", c.glyph(), cardTitle, c.Report.Verdict)},
		},
	}
	if f := strings.TrimSpace(c.Report.Finding); f != "" {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": f},
		})
	}

	var body []string
	if tl := c.trendLine(); tl != "" {
		body = append(body, tl)
	}
	for _, ln := range c.bucketLines() {
		body = append(body, "• "+ln)
	}
	if c.Report.Verdict != "MEASURED" {
		if na := strings.TrimSpace(c.Report.NextAction); na != "" {
			body = append(body, "next: "+na)
		}
	}
	if len(body) > 0 {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": strings.Join(body, "\n")},
		})
	}

	ctxParts := []string{c.fence()}
	if s := strings.TrimSpace(c.Report.Schema); s != "" {
		ctxParts = append(ctxParts, "schema: "+s)
	}
	if src := strings.TrimSpace(c.Source); src != "" {
		ctxParts = append(ctxParts, "posted by "+src)
	}
	blocks = append(blocks, map[string]any{
		"type":     "context",
		"elements": []any{map[string]any{"type": "mrkdwn", "text": strings.Join(ctxParts, "  ·  ")}},
	})
	return blocks
}
