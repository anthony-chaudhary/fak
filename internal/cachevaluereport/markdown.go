package cachevaluereport

import (
	"fmt"
	"strings"
)

// markdown.go — the #1305 visual render layer (rung D of epic #1301). RenderTwoTrackMarkdown
// emits a GitHub/Slack-markdown view of the two-track P&L: a mermaid xychart-beta trend block
// per track, an ASCII sparkline per trended metric, and a KPI table whose every row names its
// provenance (WITNESSED kernel vs OBSERVED $ projection) so a reader can never mistake the
// cost projection for a fak-witnessed claim. It is pure and deterministic.

// sparkGlyphs are the eight block glyphs the sparkline draws with; they render identically in
// Slack mrkdwn, GitHub markdown, and a terminal.
var sparkGlyphs = []rune("▁▂▃▄▅▆▇█")

// markdownSparkline maps a series to the block glyphs, normalized against the series' own
// min/max so a flat series reads flat and a trend reads as a slope. An empty series is "".
func markdownSparkline(vals []float64) string {
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
	var b strings.Builder
	for _, v := range vals {
		idx := 0
		if span > 0 {
			idx = int((v - min) / span * float64(len(sparkGlyphs)-1))
			if idx < 0 {
				idx = 0
			}
			if idx >= len(sparkGlyphs) {
				idx = len(sparkGlyphs) - 1
			}
		} else {
			idx = len(sparkGlyphs) / 2 // a flat series sits mid-glyph
		}
		b.WriteRune(sparkGlyphs[idx])
	}
	return b.String()
}

// mermaidXYChart renders a mermaid `xychart-beta` line block from parallel period labels and
// values. An empty series yields "" (no block) so a track with no data prints no chart.
func mermaidXYChart(title, yLabel string, periods []string, vals []float64) string {
	if len(periods) == 0 || len(periods) != len(vals) {
		return ""
	}
	xs := make([]string, len(periods))
	for i, p := range periods {
		xs[i] = `"` + p + `"`
	}
	ys := make([]string, len(vals))
	for i, v := range vals {
		ys[i] = fmt.Sprintf("%.4f", v)
	}
	var b strings.Builder
	b.WriteString("```mermaid\n")
	b.WriteString("xychart-beta\n")
	fmt.Fprintf(&b, "    title \"%s\"\n", title)
	fmt.Fprintf(&b, "    x-axis [%s]\n", strings.Join(xs, ", "))
	fmt.Fprintf(&b, "    y-axis \"%s\"\n", yLabel)
	fmt.Fprintf(&b, "    line [%s]\n", strings.Join(ys, ", "))
	b.WriteString("```\n")
	return b.String()
}

// RenderTwoTrackMarkdown renders the two-track P&L as markdown with mermaid trend charts,
// sparklines, and a provenance-labelled KPI table (#1305). Tracks stay side by side, never
// blended — Track 1 is WITNESSED kernel reuse, Track 2 is the OBSERVED $ cost projection.
func RenderTwoTrackMarkdown(r TwoTrackReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "## cache-value P&L — %s\n\n", r.Verdict)
	if r.Finding != "" {
		fmt.Fprintf(&sb, "%s\n\n", r.Finding)
	}
	fmt.Fprintf(&sb, "> fence: %s\n\n", r.ProjectionFence)

	// Track 1 — WITNESSED realized reuse.
	t1Periods, t1Vals := bucketSeries(r.Track1.Buckets)
	fmt.Fprintf(&sb, "### Track 1 — realized cache reuse (WITNESSED kernel)\n\n")
	if len(t1Periods) > 0 {
		fmt.Fprintf(&sb, "reuse trend: `%s`  (latest %.3f, %s)\n\n", markdownSparkline(t1Vals), r.Track1.LatestReuseRatio, r.Track1.LatestTrend)
		sb.WriteString(mermaidXYChart("Track 1 — realized cache reuse (WITNESSED)", "reuse ratio", t1Periods, t1Vals))
		sb.WriteString("\n")
	} else {
		sb.WriteString("no multi-turn reuse to trend yet.\n\n")
	}

	// Track 2 — OBSERVED $ cost projection.
	t2Periods, t2Net := savingsSeries(r.Track2)
	fmt.Fprintf(&sb, "### Track 2 — net cache economics (OBSERVED $ projection)\n\n")
	if len(t2Periods) > 0 {
		fmt.Fprintf(&sb, "cumulative-net trend: `%s`  (latest net $%.4f, cumulative $%.4f, %s)\n\n",
			markdownSparkline(t2Net), r.LatestNetUSD, r.CumulativeNetUSD, breakEvenLabel(r.BrokeEven))
		sb.WriteString(mermaidXYChart("Track 2 — cumulative net (OBSERVED $)", "cumulative net $", t2Periods, t2Net))
		sb.WriteString("\n")
	} else {
		sb.WriteString("no OBSERVED-$ rows yet (Track-2 ledger empty).\n\n")
	}

	// KPI table — every row names its provenance so the projection is never read as witnessed.
	sb.WriteString("### KPI\n\n")
	sb.WriteString("| metric | value | provenance |\n|---|---|---|\n")
	fmt.Fprintf(&sb, "| latest realized reuse | %.3f (%s) | WITNESSED (kernel) |\n", r.Track1.LatestReuseRatio, r.Track1.LatestTrend)
	fmt.Fprintf(&sb, "| latest net | $%.4f | OBSERVED ($ projection) |\n", r.LatestNetUSD)
	fmt.Fprintf(&sb, "| cumulative net | $%.4f (%s) | OBSERVED ($ projection) |\n", r.CumulativeNetUSD, breakEvenLabel(r.BrokeEven))
	return sb.String()
}

// bucketSeries extracts the (period, realized-reuse) series from Track-1 buckets, in order.
func bucketSeries(buckets []Bucket) ([]string, []float64) {
	periods := make([]string, 0, len(buckets))
	vals := make([]float64, 0, len(buckets))
	for _, b := range buckets {
		periods = append(periods, b.Period)
		vals = append(vals, b.RealizedReuseRatio)
	}
	return periods, vals
}

// savingsSeries extracts the (period, cumulative-net-$) series from Track-2 buckets, in order.
func savingsSeries(buckets []SavingsBucket) ([]string, []float64) {
	periods := make([]string, 0, len(buckets))
	vals := make([]float64, 0, len(buckets))
	for _, b := range buckets {
		periods = append(periods, b.Period)
		vals = append(vals, b.CumulativeNetUSD)
	}
	return periods, vals
}
