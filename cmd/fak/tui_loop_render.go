package main

import (
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func buildTUILoopReport(st loopmgr.Status, at time.Time) tuiLoopReport {
	rows := make([]tuiLoopRow, 0, len(st.Loops))
	for _, loop := range st.Loops {
		rows = append(rows, classifyTUILoop(loop, at))
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Attention != rows[j].Attention {
			return rows[i].Attention > rows[j].Attention
		}
		return rows[i].LoopID < rows[j].LoopID
	})
	return tuiLoopReport{
		Schema: tuiLoopsSchema,
		At:     at.UTC().Format(time.RFC3339),
		Ledger: st.LedgerPath,
		Counts: countTUILoops(rows),
		Lanes:  buildTUILoopLanes(rows),
		Rows:   rows,
	}
}

func classifyTUILoop(loop loopmgr.LoopSnapshot, at time.Time) tuiLoopRow {
	state := loop.State
	if strings.TrimSpace(state) == "" {
		state = "-"
	}
	age := int64(0)
	if loop.LastEventUnixNano > 0 {
		d := at.UTC().Sub(time.Unix(0, loop.LastEventUnixNano).UTC())
		if d > 0 {
			age = int64(d.Seconds())
		}
	}
	row := tuiLoopRow{
		LoopID:              loop.LoopID,
		State:               state,
		LastKind:            string(loop.LastKind),
		LastSeq:             loop.LastSeq,
		AgeSeconds:          age,
		CurrentRunID:        loop.CurrentRunID,
		Fires:               loop.Fires,
		Admitted:            loop.Admitted,
		Refused:             loop.Refused,
		ConsecutiveRefusals: loop.ConsecutiveRefusals,
		Started:             loop.Started,
		Ended:               loop.Ended,
		Witnessed:           loop.Witnessed,
		WitnessRefused:      loop.WitnessRefused,
		WitnessUnavailable:  loop.WitnessUnavailable,
		Notifications:       loop.Notifications,
	}
	if loop.LastRun != nil {
		row.LastRunStatus = string(loop.LastRun.Status)
		row.LastRunReason = loop.LastRun.Reason
		row.LastRunSummary = loop.LastRun.Summary
	}
	if loop.Ended > 0 {
		rate := float64(loop.Witnessed) / float64(loop.Ended)
		row.WitnessRate = &rate
	}
	row.Tags, row.Attention = scoreTUILoop(row)
	return row
}

func scoreTUILoop(row tuiLoopRow) ([]string, int) {
	tags := []string{}
	score := 0
	state := strings.ToLower(row.State)
	status := strings.ToLower(row.LastRunStatus)
	if state == string(loopmgr.StateRunning) || status == string(loopmgr.StatusRunning) {
		tags = append(tags, "running")
		score += 70
	}
	if state == string(loopmgr.StatusRefused) || status == string(loopmgr.StatusRefused) || row.ConsecutiveRefusals > 0 {
		tags = append(tags, "refused")
		score += 80 + int(row.ConsecutiveRefusals)*20
	}
	if status == string(loopmgr.StatusFailed) || state == string(loopmgr.StatusFailed) {
		tags = append(tags, "failed")
		score += 100
	}
	if row.Ended > row.Witnessed {
		tags = append(tags, "needs-witness")
		score += int(row.Ended-row.Witnessed) * 15
	}
	if row.WitnessRefused > 0 {
		tags = append(tags, "witness-refused")
		score += int(row.WitnessRefused) * 20
	}
	if row.WitnessUnavailable > 0 {
		tags = append(tags, "witness-unavailable")
		score += int(row.WitnessUnavailable) * 10
	}
	if row.AgeSeconds > int64(6*time.Hour/time.Second) && (state == "running" || status == "running") {
		tags = append(tags, "old-running")
		score += 40
	}
	if score == 0 && (status == string(loopmgr.StatusWitnessedDone) || row.Witnessed > 0) {
		tags = append(tags, "witnessed")
	}
	return tags, score
}

func countTUILoops(rows []tuiLoopRow) tuiLoopCounts {
	var c tuiLoopCounts
	for _, row := range rows {
		c.Loops++
		if tuiLoopHasTag(row, "running") {
			c.Running++
		}
		if tuiLoopHasTag(row, "refused") {
			c.Refused++
		}
		if tuiLoopHasTag(row, "failed") {
			c.Failed++
		}
		if row.Witnessed > 0 {
			c.Witnessed++
		}
		if tuiLoopHasTag(row, "needs-witness") || tuiLoopHasTag(row, "witness-refused") || tuiLoopHasTag(row, "witness-unavailable") {
			c.WitnessGaps++
		}
		if row.Notifications > 0 {
			c.Notifications++
		}
	}
	return c
}

func buildTUILoopLanes(rows []tuiLoopRow) []tuiLoopLane {
	names := []string{"running", "refused", "needs-witness", "witnessed", "other"}
	lanes := make([]tuiLoopLane, 0, len(names))
	for _, name := range names {
		lane := tuiLoopLane{Name: name}
		for _, row := range rows {
			if !rowInTUILoopLane(row, name) {
				continue
			}
			lane.Count++
			if lane.TopLoop == "" {
				lane.TopLoop = row.LoopID
				lane.TopLoopText = row.LastRunSummary
				if lane.TopLoopText == "" {
					lane.TopLoopText = strings.Join(row.Tags, ",")
				}
			}
		}
		lanes = append(lanes, lane)
	}
	return lanes
}

func rowInTUILoopLane(row tuiLoopRow, lane string) bool {
	switch lane {
	case "running":
		return tuiLoopHasTag(row, "running")
	case "refused":
		return tuiLoopHasTag(row, "refused") || tuiLoopHasTag(row, "failed")
	case "needs-witness":
		return tuiLoopHasTag(row, "needs-witness") || tuiLoopHasTag(row, "witness-refused") || tuiLoopHasTag(row, "witness-unavailable")
	case "witnessed":
		return tuiLoopHasTag(row, "witnessed")
	case "other":
		return len(row.Tags) == 0
	default:
		return false
	}
}

func tuiLoopHasTag(row tuiLoopRow, tag string) bool {
	for _, got := range row.Tags {
		if got == tag {
			return true
		}
	}
	return false
}

func renderTUILoops(report tuiLoopReport, top, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console loops  at=%s  ledger=%s\n", report.At, report.Ledger)
	fmt.Fprintf(&b, "loops=%d  running=%d  refused=%d  failed=%d  witnessed=%d  witness-gaps=%d  notified=%d\n",
		report.Counts.Loops, report.Counts.Running, report.Counts.Refused, report.Counts.Failed,
		report.Counts.Witnessed, report.Counts.WitnessGaps, report.Counts.Notifications)
	if report.Integrity != nil && report.Integrity.Broken {
		fmt.Fprintf(&b, "!! LEDGER CHAIN BROKEN at line %d (seq %d): %s -- showing %d recovered loop-event(s); later rows not loaded\n",
			report.Integrity.AtLine, report.Integrity.AtSeq,
			trimTUI(report.Integrity.Reason, maxTUI(40, width-48)), report.Integrity.Recovered)
	}
	if len(report.Rows) == 0 {
		fmt.Fprintln(&b, "\nno loops found")
		return b.String()
	}
	fmt.Fprintln(&b, "\nLanes")
	fmt.Fprintln(&b, "lane             count top")
	for _, lane := range report.Lanes {
		topText := "-"
		if lane.TopLoop != "" {
			topText = lane.TopLoop
			if lane.TopLoopText != "" {
				topText += " " + lane.TopLoopText
			}
		}
		fmt.Fprintf(&b, "%-16s %5d %s\n", lane.Name, lane.Count, trimTUI(topText, maxTUI(20, width-24)))
	}
	fmt.Fprintln(&b, "\nLoop Queue")
	renderTUILoopRows(&b, report.Rows, minTUI(top, len(report.Rows)), width)
	return b.String()
}

func renderTUILoopRows(b *strings.Builder, rows []tuiLoopRow, limit, width int) {
	fmt.Fprintln(b, "attention loop                         state          age    runs             witness tags")
	for _, row := range rows[:limit] {
		runs := fmt.Sprintf("f%d/a%d/r%d/e%d", row.Fires, row.Admitted, row.Refused, row.Ended)
		witness := "-"
		if row.WitnessRate != nil {
			witness = trimFloat(*row.WitnessRate * 100)
			witness += "%"
		}
		tags := strings.Join(row.Tags, ",")
		if tags == "" {
			tags = "-"
		}
		summary := row.LastRunSummary
		if summary == "" {
			summary = row.LastRunReason
		}
		lineTail := tags
		if summary != "" {
			lineTail += "  " + summary
		}
		fmt.Fprintf(b, "%9d %-28s %-14s %-6s %-16s %-7s %s\n",
			row.Attention, trimTUI(row.LoopID, 28), trimTUI(row.State, 14),
			durationTUIText(row.AgeSeconds), runs, witness, trimTUI(lineTail, maxTUI(16, width-88)))
	}
}

func durationTUIText(seconds int64) string {
	if seconds <= 0 {
		return "0s"
	}
	if seconds < 60 {
		return strconv.FormatInt(seconds, 10) + "s"
	}
	minutes := seconds / 60
	if minutes < 60 {
		return strconv.FormatInt(minutes, 10) + "m"
	}
	hours := minutes / 60
	if hours < 48 {
		return strconv.FormatInt(hours, 10) + "h"
	}
	return strconv.FormatInt(hours/24, 10) + "d"
}

func renderTUIIssues(report tuiIssueReport, top, width int) string {
	var b strings.Builder
	title := "fak console issues"
	fmt.Fprintf(&b, "%s  as_of=%s  source=%s\n", title, report.AsOf, report.Source)
	fmt.Fprintf(&b, "open=%d  P0=%d  P1=%d  P2=%d  orphan=%d  stale=%d  needs: prio=%d kind=%d area=%d\n",
		report.Counts.Open, report.Counts.P0, report.Counts.P1, report.Counts.P2,
		report.Counts.Orphan, report.Counts.Stale, report.Counts.NeedsPriority,
		report.Counts.NeedsKind, report.Counts.NeedsArea)
	if report.Epic != nil {
		fmt.Fprintf(&b, "\nEpic #%d  score=%d  idle=%dd\n", report.Epic.Number, report.Epic.Score, report.Epic.IdleDays)
		fmt.Fprintf(&b, "  %s\n", trimTUI(report.Epic.Title, width-2))
		fmt.Fprintf(&b, "  related loaded issues: %d\n", report.Counts.Related)
	}
	fmt.Fprintln(&b, "\nLanes")
	fmt.Fprintln(&b, "lane             count orphan needs-kind needs-area max-idle top")
	for _, lane := range report.Lanes {
		topText := "-"
		if lane.TopIssue != 0 {
			topText = fmt.Sprintf("#%d %s", lane.TopIssue, lane.TopIssueText)
		}
		fmt.Fprintf(&b, "%-16s %5d %6d %10d %10d %8dd %s\n",
			lane.Name, lane.Count, lane.Orphan, lane.NeedsKind, lane.NeedsArea,
			lane.MaxIdleDays, trimTUI(topText, maxTUI(20, width-62)))
	}
	rows := report.Rows
	if report.Epic != nil {
		related := []tuiIssueRow{}
		for _, row := range report.Rows {
			if row.Related && row.Number != report.Epic.Number {
				related = append(related, row)
			}
		}
		if len(related) > 0 {
			fmt.Fprintln(&b, "\nRelated")
			renderTUIIssueRows(&b, related, minTUI(top, len(related)), width)
		}
	}
	fmt.Fprintln(&b, "\nRanked Queue")
	renderTUIIssueRows(&b, rows, minTUI(top, len(rows)), width)
	if len(report.Actions) > 0 {
		fmt.Fprintln(&b, "\nReview Actions")
		limit := minTUI(8, len(report.Actions))
		for _, action := range report.Actions[:limit] {
			fmt.Fprintf(&b, "#%-5d %-23s %s\n", action.Number, action.Kind, trimTUI(action.Reason, width-32))
		}
		if len(report.Actions) > limit {
			fmt.Fprintf(&b, "... %d more actions in --json\n", len(report.Actions)-limit)
		}
	}
	return b.String()
}

func renderTUIIssueRows(b *strings.Builder, rows []tuiIssueRow, limit, width int) {
	fmt.Fprintln(b, "#      score prio idle tags                         title")
	for _, row := range rows[:limit] {
		prio := row.Priority
		if prio == "" {
			prio = "-"
		} else {
			prio = strings.TrimPrefix(prio, "priority/")
		}
		tags := strings.Join(row.Tags, ",")
		if tags == "" {
			tags = "-"
		}
		titleWidth := maxTUI(20, width-49)
		fmt.Fprintf(b, "#%-5d %5d %-4s %4dd %-28s %s\n",
			row.Number, row.Score, prio, row.IdleDays, trimTUI(tags, 28), trimTUI(row.Title, titleWidth))
	}
}

func trimTUI(s string, width int) string {
	s = strings.Join(strings.Fields(s), " ")
	if width <= 0 || len(s) <= width {
		return s
	}
	if width <= 3 {
		return s[:width]
	}
	return s[:width-3] + "..."
}

func minTUI(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxTUI(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func nonEmptyTUI(values []string) []string {
	out := []string{}
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			out = append(out, strings.TrimSpace(value))
		}
	}
	return out
}

func firstNonEmptyTUI(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func hasStringTUI(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func blankTUI(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return strings.TrimSpace(s)
}

func shellLineTUI(args []string) string {
	parts := make([]string, 0, len(args))
	for _, arg := range args {
		parts = append(parts, shellArgTUI(arg))
	}
	return strings.Join(parts, " ")
}

func shellArgTUI(arg string) string {
	if arg == "" {
		return `""`
	}
	if strings.IndexFunc(arg, func(r rune) bool {
		return r == '"' || r == '\'' || r == '\\' || r == '$' || r == '`' || r <= ' '
	}) >= 0 {
		return strconv.Quote(arg)
	}
	return arg
}

func tuiUsage(w io.Writer) {
	fmt.Fprint(w, `fak console - native terminal control panes
Alias: fak tui

  fak console issues [--issues-json FILE] [--json] [--epic N]
                 [--repo owner/repo] [--state open|closed|all]
                 [--limit N] [--top N] [--width N] [--as-of YYYY-MM-DD]
  fak console loops  [--ledger FILE] [--json] [--top N] [--width N]
                 [--at RFC3339|YYYY-MM-DD]
  fak console sessions [--sessions-json FILE] [--json] [--addr URL] [--key K]
                   [--top N] [--width N] [--at RFC3339|YYYY-MM-DD]
  fak console garden [--garden-json FILE] [--json] [--check]
                 [--workspace DIR] [--deep] [--timeout N] [--width N]
  fak console guard  --guard-json FILE [--guard-json FILE ...] [--json]
                 [--width N] [--at RFC3339|YYYY-MM-DD]
  fak console agent [--account NAME | --claude-config-dir DIR] [--dry-run]
                [--prompt STR] [--session-id ID] [--passthrough]
                [--gateway-url URL --gateway-key-env VAR --model MODEL] [--json]
                [--] [claude args...]
  fak console overview [--issues-json FILE] [--ledger FILE] [--sessions-json FILE]
                   [--garden-json FILE] [--guard-json FILE ...] [--json]

The issues pane folds GitHub issues into a ranked terminal model: priority lanes,
orphan/stale/label gaps, optional epic-related rows, and review actions. With no
--issues-json it shells out to gh issue list; fixtures keep the model testable.

The loops pane folds fak's hash-chained loop ledger into the same terminal model:
running/refused/witness-gap lanes, attention ranking, and machine-readable JSON.

The sessions pane reads GET /v1/fak/sessions or a fixture JSON and renders live
DRIVE state: run-state lanes, budgets, pace, priority, lineage, and reasons.

The garden pane reads `+"`fak garden --json`"+` envelopes or runs the read-only garden
bundle and renders member health, gating regressions, and advisory actions.

The guard pane reads existing guard/adjudication JSON artifacts and renders
denials, reasons, audit status, and proof-packet gaps without replaying calls.

The agent pane launches a real Claude Code backend. By default it starts a local
`+"`fak guard`"+` and pins CLAUDE_CONFIG_DIR from `+"`fak accounts`"+`; with
--gateway-url it instead launches Claude Code directly against an already-running
`+"`fak serve`"+` gateway, reading the bearer from --gateway-key-env.

The overview pane composes selected pane models into one ranked spine so
operators can see issue, loop, session, garden, and guard pressure together.
`)
}
