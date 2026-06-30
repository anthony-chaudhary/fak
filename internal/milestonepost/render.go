package milestonepost

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/internal/slackmeta"
)

// cardTitle is the bold headline every milestone card leads with.
const cardTitle = "fak milestones — climb + roadmap"

// glyph maps the report's finding to a leading status emoji so the channel reads the
// state at a glance: a recorded tick is a bar chart, an advisory regression a warning,
// an unmeasured (gh-down) report a muted hourglass.
func (c Card) glyph() string {
	switch c.Report.Finding {
	case "milestone_advisory":
		return ":warning:"
	case "milestone_unmeasured":
		return ":hourglass_flowing_sand:"
	default: // milestone_recorded and the zero value
		return ":bar_chart:"
	}
}

// climbLine is the maturity headline: the highest rung reached, the matured floor,
// and the witnessed progress percent.
func (c Card) climbLine() string {
	m := c.Report.Maturity
	if m.Err != "" {
		return "climb: unmeasured (" + m.Err + ")"
	}
	return fmt.Sprintf("climb: %s — %d/%d cell(s) matured (M4+), progress %.1f%%", dash(m.Highest), m.Matured, m.Cells, m.ProgressPct)
}

// ladderLine renders the M0..M7 distribution as a compact one-liner.
func (c Card) ladderLine() string {
	return "ladder: " + milestonereport.RenderDist(c.Report.Maturity.Dist)
}

// roadmapLine is the roadmap headline: the completion across DISCRETE epics, plus the
// count of ongoing optimization programs (which carry no completion %).
func (c Card) roadmapLine() string {
	e := c.Report.Epics
	if e.Err != "" {
		return "roadmap: unmeasured (" + e.Err + ")"
	}
	return fmt.Sprintf("roadmap: %.1f%% across %d discrete epic(s); %d ongoing program(s)", e.OverallPct, e.Discrete, e.Programs)
}

// epicLines renders one bullet per tracked epic, split by work class: a discrete
// epic shows its completion % (-> done), an ongoing program shows frontier activity
// (shipped / in-flight, no % — it has no 100%). An unreadable row is surfaced
// honestly as "gh read failed" rather than a fabricated 0%.
func (c Card) epicLines() []string {
	var out []string
	for _, row := range c.Report.Epics.Rows {
		if row.Err != "" {
			out = append(out, fmt.Sprintf("#%d %s — gh read failed", row.Number, row.Title))
			continue
		}
		if row.Ongoing() {
			open := row.Total - row.Closed
			src := ""
			if row.Source != "" {
				src = " {" + row.Source + "}"
			}
			out = append(out, fmt.Sprintf("#%d %s [%s] — %d shipped / %d in-flight%s", row.Number, row.Title, row.Class.Label(), row.Closed, open, src))
			continue
		}
		src := ""
		if row.Source != "" {
			src = " [" + row.Source + "]"
		}
		out = append(out, fmt.Sprintf("#%d %s — %.0f%% (%d/%d)%s", row.Number, row.Title, row.Pct, row.Closed, row.Total, src))
	}
	return out
}

// trendLine renders the per-tick trend summary, or "" when there is no trend.
func (c Card) trendLine() string {
	if c.Report.Trend == nil {
		return ""
	}
	return "trend: " + c.Report.Trend.Summary
}

// Text renders the plain-text fallback — the line Slack shows in notifications and any
// client without Block Kit. It is also what tests and --dry-run assert on. It leads
// with the verdict + headline, then the climb, ladder, roadmap rows, the trend, and
// the next action.
func (c Card) Text() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "%s *%s* — %s", c.glyph(), cardTitle, c.Report.Verdict)
	fmt.Fprintf(&sb, "\n%s", c.climbLine())
	fmt.Fprintf(&sb, "\n%s", c.ladderLine())
	fmt.Fprintf(&sb, "\n%s", c.roadmapLine())
	for _, ln := range c.epicLines() {
		fmt.Fprintf(&sb, "\n• %s", ln)
	}
	if note := strings.TrimSpace(c.Report.Epics.PartialNote); note != "" {
		fmt.Fprintf(&sb, "\n(%s)", note)
	}
	if tl := c.trendLine(); tl != "" {
		fmt.Fprintf(&sb, "\n%s", tl)
	}
	if na := strings.TrimSpace(c.Report.NextAction); na != "" {
		fmt.Fprintf(&sb, "\nnext: %s", na)
	}
	if src := strings.TrimSpace(c.Source); src != "" {
		fmt.Fprintf(&sb, "\n_posted by %s_", src)
	}
	return slackmeta.AppendText(sb.String(), c.signalNoise())
}

// Blocks renders the Block Kit payload for a richer card. It carries the same facts as
// Text so a non-Block client loses nothing: a headline section, a climb+ladder section,
// a roadmap section with the per-epic rows, the trend, and a context line with the
// schema and source.
func (c Card) Blocks() []any {
	blocks := []any{
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": fmt.Sprintf("*%s %s* — %s", c.glyph(), cardTitle, c.Report.Verdict)},
		},
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": c.climbLine() + "\n" + c.ladderLine()},
		},
	}

	road := []string{c.roadmapLine()}
	for _, ln := range c.epicLines() {
		road = append(road, "• "+ln)
	}
	if note := strings.TrimSpace(c.Report.Epics.PartialNote); note != "" {
		road = append(road, "("+note+")")
	}
	blocks = append(blocks, map[string]any{
		"type": "section",
		"text": map[string]any{"type": "mrkdwn", "text": strings.Join(road, "\n")},
	})

	var tail []string
	if tl := c.trendLine(); tl != "" {
		tail = append(tail, tl)
	}
	if na := strings.TrimSpace(c.Report.NextAction); na != "" {
		tail = append(tail, "next: "+na)
	}
	if len(tail) > 0 {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": strings.Join(tail, "\n")},
		})
	}

	ctxParts := []string{}
	if s := strings.TrimSpace(c.Report.Schema); s != "" {
		ctxParts = append(ctxParts, "schema: "+s)
	}
	if src := strings.TrimSpace(c.Source); src != "" {
		ctxParts = append(ctxParts, "posted by "+src)
	}
	if len(ctxParts) > 0 {
		blocks = append(blocks, map[string]any{
			"type":     "context",
			"elements": []any{map[string]any{"type": "mrkdwn", "text": strings.Join(ctxParts, "  ·  ")}},
		})
	}
	return slackmeta.AppendContext(blocks, c.signalNoise())
}

func (c Card) signalNoise() slackmeta.Score {
	signal := 1 + slackmeta.NonEmpty(c.Report.Finding, c.climbLine(), c.roadmapLine(), c.trendLine(), c.Report.NextAction) + len(c.epicLines())
	noise := 1 + slackmeta.NonEmpty(c.Report.Schema, c.Source)
	return slackmeta.New(signal, noise, "climb + ladder + roadmap rows + trend vs schema/source")
}

// dash renders an empty string as "-" so a missing rung label reads cleanly.
func dash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}
