package dojopost

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/slackmeta"
)

// Post formats structured notification messages for Slack publication.
type Post slackmeta.Post

// Text renders the post as plain text according to signal/noise scoring.
func (p Post) Text() string { return slackmeta.Post(p).Text(p.signalNoise()) }

// Blocks renders the post as Slack Block Kit structures according to signal/noise scoring.
func (p Post) Blocks() []any { return slackmeta.Post(p).Blocks(p.signalNoise()) }

func (p Post) signalNoise() slackmeta.Score {
	signal := 1 + slackmeta.NonEmpty(p.Lead) + len(p.Lines)
	noise := 1 + slackmeta.NonEmpty(p.Source)
	return slackmeta.New(signal, noise, "dojo calibration headline, trend/episode rows vs source/context")
}

// gradeEmoji maps a calibration grade letter to a visual status emoji.
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

// RollupFromReport formats a single calibration run into a Slack message with worst-first episodes.
func RollupFromReport(r dojo.Report, maxEpisodes int) Post {
	p := Post{
		Emoji: gradeEmoji(r.Grade),
		Title: "dojo rollup — latest run",
		Lead: fmt.Sprintf("%d lever(s) · %d episode(s) · %d measured · %d calibrated · mean calib-err %.3f · grade %s · @%s",
			r.LeverCount, r.EpisodeCount, r.Measured, r.Calibrated, r.MeanCalibErr, r.Grade, shortCommit(r.Commit)),
	}
	appendRollupOperatorLines(&p, r)
	if r.Measured == 0 {
		if r.Reason != "" {
			p.Lead += "\n" + r.Reason
		}
		return p
	}

	eps := append([]dojo.Episode(nil), r.Episodes...)
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
	if r.Trend != nil && r.Trend.Summary != "" {
		p.Lines = append(p.Lines, "trend: "+r.Trend.Summary)
	}
	return p
}

// TrendFromLedger summarizes historical calibration trend rows into a Slack notification.
func TrendFromLedger(rows []dojo.LedgerRow, n int) Post {
	if len(rows) == 0 {
		return Post{
			Emoji: ":dart:",
			Title: "dojo trend — calibration over time",
			Lead:  "no dojo history yet — run `fak dojo run --corpus DIR --append-history` to start the series",
			Lines: []string{"operator: append a measured dojo run before treating the channel as a trend"},
		}
	}
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
