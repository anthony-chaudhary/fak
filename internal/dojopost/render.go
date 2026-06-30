package dojopost

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/slackmeta"
)

// Post is one dojo-channel message, decoupled from which fold produced it. The two
// folds (RollupFromReport, TrendFromLedger) each build one, so the renderer
// (Text/Blocks) has a single input shape — the same pattern as benchpost.Post and
// scoreboard.Update.
type Post struct {
	Emoji  string   // leading status glyph
	Title  string   // headline, e.g. "dojo rollup — latest run"
	Lead   string   // one-line summary / honesty banner under the title
	Lines  []string // the body: one line per episode / trend row
	Source string   // who posted: "ci" | "agent" | hostname (optional)
}

// Text renders the plain-text fallback — the line Slack shows in notifications and any
// client without Block Kit, and what tests and --dry-run assert on.
func (p Post) Text() string {
	var b strings.Builder
	emoji := p.Emoji
	if emoji == "" {
		emoji = ":dart:"
	}
	fmt.Fprintf(&b, "%s *%s*", emoji, p.Title)
	if p.Lead != "" {
		fmt.Fprintf(&b, "\n%s", p.Lead)
	}
	for _, ln := range p.Lines {
		fmt.Fprintf(&b, "\n• %s", ln)
	}
	if p.Source != "" {
		fmt.Fprintf(&b, "\n_posted by %s_", p.Source)
	}
	return slackmeta.AppendText(b.String(), p.signalNoise())
}

// Blocks renders the Block Kit payload. It carries the same facts as Text so a
// non-Block client loses nothing.
func (p Post) Blocks() []any {
	emoji := p.Emoji
	if emoji == "" {
		emoji = ":dart:"
	}
	blocks := []any{
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": fmt.Sprintf("*%s %s*", emoji, p.Title)},
		},
	}
	if p.Lead != "" {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": p.Lead},
		})
	}
	if len(p.Lines) > 0 {
		body := "• " + strings.Join(p.Lines, "\n• ")
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": body},
		})
	}
	if p.Source != "" {
		blocks = append(blocks, map[string]any{
			"type":     "context",
			"elements": []any{map[string]any{"type": "mrkdwn", "text": "posted by " + p.Source}},
		})
	}
	return slackmeta.AppendContext(blocks, p.signalNoise())
}

func (p Post) signalNoise() slackmeta.Score {
	signal := 1 + slackmeta.NonEmpty(p.Lead) + len(p.Lines)
	noise := 1 + slackmeta.NonEmpty(p.Source)
	return slackmeta.New(signal, noise, "dojo calibration headline, trend/episode rows vs source/context")
}

// gradeEmoji maps a dojo letter grade to a status glyph so the channel reads the
// calibration health at a glance: A/B is green, C/D amber, F red, n/a (nothing
// measured) a neutral mirror.
func gradeEmoji(grade string) string {
	switch strings.ToUpper(strings.TrimSpace(grade)) {
	case "A", "B":
		return ":white_check_mark:"
	case "C", "D":
		return ":large_yellow_circle:"
	case "F":
		return ":red_circle:"
	default:
		return ":dart:"
	}
}

// RollupFromReport folds the latest dojo run into a Post: the per-run aggregate in
// the lead (levers/episodes/calibrated/mean calib-err/grade) and one line per scored
// episode, worst-first (the report's episodes are already sorted worst-first by the
// CLI). At most maxEpisodes lines are shown; the rest is summarized so a wide run does
// not flood the channel. Each line keeps the conflation-honest WITNESSED/OBSERVED
// provenance the episode carries — the dojo never blurs a number fak authored with one
// the provider billed.
func RollupFromReport(r dojo.Report, maxEpisodes int) Post {
	p := Post{
		Emoji: gradeEmoji(r.Grade),
		Title: "dojo rollup — latest run",
		Lead: fmt.Sprintf("%d lever(s) · %d episode(s) · %d measured · %d calibrated · mean calib-err %.3f · grade %s · @%s",
			r.LeverCount, r.EpisodeCount, r.Measured, r.Calibrated, r.MeanCalibErr, r.Grade, shortCommit(r.Commit)),
	}
	appendRollupOperatorLines(&p, r)
	// A run that measured nothing is the dojo's "ACTION" state — surface the reason so
	// the channel sees the gym needs a corpus, not a silent empty card.
	if r.Measured == 0 {
		if r.Reason != "" {
			p.Lead += "\n" + r.Reason
		}
		return p
	}

	eps := append([]dojo.Episode(nil), r.Episodes...)
	// Defensive: the CLI sorts worst-first before folding, but a caller handing us an
	// unsorted report (or the JSON round-trip) should still read worst-first.
	sort.SliceStable(eps, func(i, j int) bool {
		if eps[i].CalibErr != eps[j].CalibErr {
			return eps[i].CalibErr > eps[j].CalibErr
		}
		if eps[i].Lever != eps[j].Lever {
			return eps[i].Lever < eps[j].Lever
		}
		return eps[i].Metric < eps[j].Metric
	})

	shown := eps
	if maxEpisodes > 0 && len(eps) > maxEpisodes {
		shown = eps[:maxEpisodes]
	}
	for _, e := range shown {
		// UNMEASURED episodes carry no scored gap; render them as a mirror rather than a
		// misleading "claimed vs realized 0.000".
		if e.Verdict == dojo.VerdictUnmeasured {
			p.Lines = append(p.Lines,
				fmt.Sprintf("`%s/%s` · UNMEASURED (no ground truth)", e.Lever, e.Metric))
			continue
		}
		p.Lines = append(p.Lines,
			fmt.Sprintf("`%s/%s` · claimed %.3f → realized %.3f · %s · grade %s · calib-err %.3f · %s · n=%d",
				e.Lever, e.Metric, e.Claimed, e.Realized, e.Verdict, e.Grade, e.CalibErr, e.Provenance, e.Sample))
	}
	if len(eps) > len(shown) {
		p.Lines = append(p.Lines, fmt.Sprintf("…and %d more episode(s) (worst-first; see `fak dojo run`)", len(eps)-len(shown)))
	}
	// If the report attached a trend (the CLI fills it from the ledger), append the
	// one-line across-tick direction so a rollup also answers "are we improving".
	if r.Trend != nil && r.Trend.Summary != "" {
		p.Lines = append(p.Lines, "trend: "+r.Trend.Summary)
	}
	return p
}

// TrendFromLedger folds the durable history ledger into a Post that answers the dojo's
// reason to exist: are our predictors getting better calibrated over time? It reads the
// committed docs/dojo/history.jsonl (parsed into rows), trends the latest row against
// the prior one, and shows the last n rows worst-recent-first. This fold needs NO
// corpus scan — it reports the committed evidence — so CI can post the trend cheaply on
// a cadence. An empty ledger yields a "no history yet" card (the channel still sees the
// state).
func TrendFromLedger(rows []dojo.LedgerRow, n int) Post {
	if len(rows) == 0 {
		return Post{
			Emoji: ":dart:",
			Title: "dojo trend — calibration over time",
			Lead:  "no dojo history yet — run `fak dojo run --corpus DIR --append-history` to start the series",
			Lines: []string{"operator: append a measured dojo run before treating the channel as a trend"},
		}
	}
	// Rows are in file (append) order; the latest is the last.
	latest := rows[len(rows)-1]
	prior := rows[:len(rows)-1]
	trend := dojo.TrendVsLast(latest, prior)

	emoji := gradeEmoji(latest.Grade)
	switch trend.Direction {
	case "improved":
		emoji = ":chart_with_upwards_trend:"
	case "regressed":
		emoji = ":chart_with_downwards_trend:"
	}

	p := Post{
		Emoji: emoji,
		Title: "dojo trend — calibration over time",
		Lead: fmt.Sprintf("latest: mean calib-err %.3f · grade %s · %d/%d calibrated · @%s (%s) — %s",
			latest.MeanCalibErr, latest.Grade, latest.Calibrated, latest.Measured, shortCommit(latest.Commit), latest.Date, trend.Summary),
	}
	p.Lines = append(p.Lines,
		"current: "+coverageSummary(latest.LeverCount, latest.EpisodeCount, latest.Measured, latest.Calibrated),
		"operator: "+trendOperatorMeaning(trend.Direction),
	)

	// Show the last n rows, most-recent first, as a compact history strip.
	if n <= 0 || n > len(rows) {
		n = len(rows)
	}
	for i := 0; i < n; i++ {
		row := rows[len(rows)-1-i]
		p.Lines = append(p.Lines,
			fmt.Sprintf("%s · mean calib-err %.3f · grade %s · %d/%d calibrated · @%s",
				row.Date, row.MeanCalibErr, row.Grade, row.Calibrated, row.Measured, shortCommit(row.Commit)))
	}
	return p
}

func appendRollupOperatorLines(p *Post, r dojo.Report) {
	if r.NextAction != "" {
		p.Lines = append(p.Lines, "operator: "+r.NextAction)
	}
	p.Lines = append(p.Lines, "current: "+coverageSummary(r.LeverCount, r.EpisodeCount, r.Measured, r.Calibrated))
	if line := worstLeverLine(r.Episodes); line != "" {
		p.Lines = append(p.Lines, line)
	}
}

func coverageSummary(leverCount, episodeCount, measured, calibrated int) string {
	unmeasured := episodeCount - measured
	if unmeasured < 0 {
		unmeasured = 0
	}
	return fmt.Sprintf("%d lever(s), %d episode(s), %d measured, %d unmeasured, %d calibrated",
		leverCount, episodeCount, measured, unmeasured, calibrated)
}

func worstLeverLine(eps []dojo.Episode) string {
	board := dojo.BoardFromEpisodes(eps)
	for _, row := range board.Rows {
		if row.Measured == 0 {
			continue
		}
		if row.WorstMetric == "" {
			return fmt.Sprintf("worst lever: `%s` · grade %s · mean calib-err %.3f",
				row.Lever, row.Grade, row.MeanCalibErr)
		}
		return fmt.Sprintf("worst lever: `%s` · grade %s · mean calib-err %.3f · worst metric `%s` (%.3f)",
			row.Lever, row.Grade, row.MeanCalibErr, row.WorstMetric, row.WorstCalib)
	}
	for _, row := range board.Rows {
		if row.Unmeasured > 0 {
			return fmt.Sprintf("attention: `%s` has no measured ground truth yet", row.Lever)
		}
	}
	return ""
}

func trendOperatorMeaning(direction string) string {
	switch direction {
	case "improved":
		return "claims moved closer to billed reality; keep the current theory and watch the next tick"
	case "regressed":
		return "claims drifted away from billed reality; inspect the latest rollup's worst lever before changing policy"
	case "flat":
		return "no displayable movement; use the latest rollup if you need the current worst lever"
	case "new":
		return "first tick only; append another measured run before reading this as a trend"
	default:
		return "trend direction unknown; inspect the ledger before acting"
	}
}

// shortCommit trims a commit to 12 chars for a compact channel line (mirrors the dojo
// package's own short form).
func shortCommit(c string) string {
	if c == "" {
		return "unknown"
	}
	if len(c) > 12 {
		return c[:12]
	}
	return c
}
