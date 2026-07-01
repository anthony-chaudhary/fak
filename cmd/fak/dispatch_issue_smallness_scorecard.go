package main

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/issuesmallness"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// issueSmallnessScorecardSchema names the control-pane payload this fold emits so a
// downstream `fak scoreboard post --from -` (and the fold's own consumers) can tell a
// rated-issues card apart from every other scorecard on the board.
const issueSmallnessScorecardSchema = "fak-issue-smallness-scorecard/1"

// issueSmallnessDebtKey is the headline integer the scoreboard reads: the number of open
// issues that are too big to dispatch in one sitting (verdict == fail). It rides in
// corpus[issueSmallnessDebtKey]; pass `--debt-key issue_smallness_debt` to the poster.
const issueSmallnessDebtKey = "issue_smallness_debt"

// issueSmallnessScorecard folds a rated open-issue report into the shared control-pane
// payload. This is the missing bridge between issue-smallness-lint (which *rates* each
// open issue pass/warn/fail) and `fak scoreboard post --from -` (which publishes a
// control-pane card): it lets an agent or a CI job scoreboard the rated backlog.
//
// The fold follows the shared anti-gaming convention: a fail issue (unsplittable, no done
// condition, no goal section) is HARD debt — one defect each, so debt == the fail count and
// ok == (no fails). A warn issue is a soft nudge that never counts as debt and never reds
// the gate. The KPI score is the well-scoped rate: pass / scanned as a percentage.
func issueSmallnessScorecard(report issuesmallness.OpenReport) scorecard.Payload {
	var defects, soft []string
	for _, row := range report.Flagged {
		line := fmt.Sprintf("#%d %q -> %s", row.Number, row.Title, row.Reason)
		switch row.Verdict {
		case issuesmallness.Fail:
			defects = append(defects, line)
		case issuesmallness.Warn:
			soft = append(soft, line)
		}
	}
	sort.Strings(defects)
	sort.Strings(soft)

	score := 100.0
	if report.Scanned > 0 {
		score = float64(report.Counts[issuesmallness.Pass]) / float64(report.Scanned) * 100
	}

	kpi := scorecard.KPI{
		Key:     "issue-smallness",
		Group:   "issue-smallness",
		Score:   score,
		Detail:  fmt.Sprintf("%d/%d open issue(s) are scoped small enough to dispatch", report.Counts[issuesmallness.Pass], report.Scanned),
		Defects: defects,
		Soft:    soft,
	}

	msgs := scorecard.Messages{
		Finding:         fmt.Sprintf("%d open issue(s) are too big to dispatch in one sitting", len(defects)),
		FindingClean:    "every open issue is scoped small enough to dispatch in one sitting",
		NextAction:      "split each flagged issue into 30-minute-scoped children with an explicit done condition",
		NextActionClean: "keep new issues single-goal with a goal section and an explicit done condition",
		ExtraCorpus: map[string]any{
			"scanned":    report.Scanned,
			"pass_count": report.Counts[issuesmallness.Pass],
			"warn_count": report.Counts[issuesmallness.Warn],
			"fail_count": report.Counts[issuesmallness.Fail],
		},
	}

	return scorecard.Fold(issueSmallnessScorecardSchema, []scorecard.KPI{kpi}, issueSmallnessDebtKey, nil, msgs)
}
