package main

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func loadTUIOverview(opt tuiOverviewOptions) (tuiOverviewReport, error) {
	cards := []tuiOverviewCard{}
	if opt.IssuesJSON != "" {
		issues, source, err := loadTUIIssues(opt.IssuesJSON, "", "open", 1)
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewIssueCard(buildTUIIssueReport(issues, source, opt.AsOf, opt.Epic), opt.Epic))
	} else {
		cards = append(cards, missingOverviewCard("issues", "fak console overview --issues-json issues.json --epic 837"))
	}
	if opt.Ledger != "" {
		st, err := loopmgr.SnapshotFile(opt.Ledger, opt.At)
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewLoopCard(buildTUILoopReport(st, opt.At)))
	} else {
		cards = append(cards, missingOverviewCard("loops", "fak console overview --ledger .fak/loop-ledger.jsonl"))
	}
	if opt.SessionsJSON != "" {
		list, source, err := loadTUISessions(opt.SessionsJSON, "", "")
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewSessionCard(buildTUISessionReport(list, source, opt.At)))
	} else {
		cards = append(cards, missingOverviewCard("sessions", "fak console overview --sessions-json sessions.json"))
	}
	if opt.GardenJSON != "" {
		payload, source, err := loadTUIGarden(opt.GardenJSON, "", false, 0)
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewGardenCard(buildTUIGardenReport(payload, source, opt.At, opt.CheckGarden)))
	} else {
		cards = append(cards, missingOverviewCard("garden", "fak console overview --garden-json garden.json --check"))
	}
	if len(opt.GuardJSON) > 0 {
		artifacts, err := loadTUIGuard(opt.GuardJSON)
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewGuardCard(buildTUIGuardReport(artifacts, opt.At)))
	} else {
		cards = append(cards, missingOverviewCard("guard", "fak console overview --guard-json guard-proof.json"))
	}
	sort.SliceStable(cards, func(i, j int) bool {
		if cards[i].Attention != cards[j].Attention {
			return cards[i].Attention > cards[j].Attention
		}
		return cards[i].Pane < cards[j].Pane
	})
	counts := countTUIOverview(cards)
	return tuiOverviewReport{
		Schema:  tuiOverviewSchema,
		At:      opt.At.UTC().Format(time.RFC3339),
		Source:  "selected panes",
		Counts:  counts,
		Cards:   cards,
		Actions: overviewActions(cards),
	}, nil
}

func overviewIssueCard(report tuiIssueReport, epic int) tuiOverviewCard {
	counts := map[string]int{
		"open":            report.Counts.Open,
		"p0":              report.Counts.P0,
		"p1":              report.Counts.P1,
		"orphan":          report.Counts.Orphan,
		"stale":           report.Counts.Stale,
		"needs_priority":  report.Counts.NeedsPriority,
		"needs_kind":      report.Counts.NeedsKind,
		"needs_area":      report.Counts.NeedsArea,
		"related_to_epic": report.Counts.Related,
	}
	attention := report.Counts.P0*50 + report.Counts.Orphan*15 + report.Counts.Stale*8 +
		report.Counts.NeedsPriority*20 + report.Counts.NeedsKind*10 + report.Counts.NeedsArea*10
	status := "ok"
	tags := []string{"issue-queue"}
	if attention > 0 {
		status = "action"
		tags = append(tags, "triage")
	}
	if epic > 0 {
		tags = append(tags, "epic")
	}
	summary := fmt.Sprintf("open=%d P0=%d orphan=%d stale=%d", report.Counts.Open, report.Counts.P0, report.Counts.Orphan, report.Counts.Stale)
	if epic > 0 {
		summary += fmt.Sprintf(" related=%d", report.Counts.Related)
	}
	return tuiOverviewCard{
		Pane:      "issues",
		Status:    status,
		Source:    report.Source,
		Summary:   summary,
		Command:   "fak console issues --issues-json " + report.Source,
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func overviewLoopCard(report tuiLoopReport) tuiOverviewCard {
	counts := map[string]int{
		"loops":        report.Counts.Loops,
		"running":      report.Counts.Running,
		"refused":      report.Counts.Refused,
		"failed":       report.Counts.Failed,
		"witness_gaps": report.Counts.WitnessGaps,
		"witnessed":    report.Counts.Witnessed,
	}
	attention := report.Counts.Failed*90 + report.Counts.Refused*65 + report.Counts.WitnessGaps*45 + report.Counts.Running*8
	status := "ok"
	tags := []string{"loop-ledger"}
	if report.Counts.Failed > 0 || report.Counts.Refused > 0 || report.Counts.WitnessGaps > 0 {
		status = "action"
		tags = append(tags, "loop-attention")
	} else if report.Counts.Running > 0 {
		status = "warn"
		tags = append(tags, "running")
	}
	return tuiOverviewCard{
		Pane:      "loops",
		Status:    status,
		Source:    report.Ledger,
		Summary:   fmt.Sprintf("loops=%d running=%d refused=%d witness_gaps=%d", report.Counts.Loops, report.Counts.Running, report.Counts.Refused, report.Counts.WitnessGaps),
		Command:   "fak console loops --ledger " + report.Ledger,
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func overviewSessionCard(report tuiSessionReport) tuiOverviewCard {
	counts := map[string]int{
		"sessions":       report.Counts.Sessions,
		"running":        report.Counts.Running,
		"throttled":      report.Counts.Throttled,
		"paused":         report.Counts.Paused,
		"stopped":        report.Counts.Stopped,
		"budgeted":       report.Counts.Budgeted,
		"context_budget": report.Counts.ContextBudget,
		"lineage":        report.Counts.Lineage,
	}
	lowBudget := 0
	for _, row := range report.Rows {
		if hasStringTUI(row.Tags, "low-turns") || hasStringTUI(row.Tags, "low-tokens") || hasStringTUI(row.Tags, "low-context") {
			lowBudget++
		}
	}
	counts["low_budget"] = lowBudget
	attention := report.Counts.Stopped*80 + report.Counts.Paused*45 + report.Counts.Throttled*25 + lowBudget*35
	status := "ok"
	tags := []string{"sessions"}
	if report.Counts.Stopped > 0 || report.Counts.Paused > 0 || lowBudget > 0 {
		status = "action"
		tags = append(tags, "operator-attention")
	} else if report.Counts.Throttled > 0 {
		status = "warn"
		tags = append(tags, "throttled")
	}
	return tuiOverviewCard{
		Pane:      "sessions",
		Status:    status,
		Source:    report.Source,
		Summary:   fmt.Sprintf("sessions=%d running=%d paused=%d stopped=%d low_budget=%d", report.Counts.Sessions, report.Counts.Running, report.Counts.Paused, report.Counts.Stopped, lowBudget),
		Command:   "fak console sessions --sessions-json " + report.Source,
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func overviewGardenCard(report tuiGardenReport) tuiOverviewCard {
	counts := map[string]int{
		"members": report.Counts.Members,
		"ok":      report.Counts.OK,
		"action":  report.Counts.Action,
		"red":     report.Counts.Red,
		"errored": report.Counts.Errored,
		"gating":  report.Counts.Gating,
		"skipped": report.Counts.Skipped,
	}
	attention := report.Counts.Errored*100 + report.Counts.Red*90 + report.Counts.Action*45 + report.GateExit*100
	status := "ok"
	tags := []string{"garden"}
	if report.GateExit != 0 || report.Counts.Red > 0 || report.Counts.Errored > 0 {
		status = "action"
		tags = append(tags, "garden-red")
	} else if report.Counts.Action > 0 || report.Counts.Skipped > 0 {
		status = "warn"
		tags = append(tags, "advisory")
	}
	return tuiOverviewCard{
		Pane:      "garden",
		Status:    status,
		Source:    report.Source,
		Summary:   fmt.Sprintf("finding=%s members=%d red=%d errored=%d gate=%d", report.Finding, report.Counts.Members, report.Counts.Red, report.Counts.Errored, report.GateExit),
		Command:   "fak console garden --garden-json " + report.Source + " --check",
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func overviewGuardCard(report tuiGuardReport) tuiOverviewCard {
	counts := map[string]int{
		"artifacts":    report.Counts.Artifacts,
		"rows":         report.Counts.Rows,
		"allow":        report.Counts.Allow,
		"deny":         report.Counts.Deny,
		"quarantine":   report.Counts.Quarantine,
		"policy_block": report.Counts.PolicyBlock,
		"default_deny": report.Counts.DefaultDeny,
		"expected":     report.Counts.Expected,
		"unexpected":   report.Counts.Unexpected,
	}
	attention := report.Counts.Unexpected*100 + report.Counts.Quarantine*80 + report.Counts.PolicyBlock*45 + report.Counts.DefaultDeny*25
	status := "ok"
	tags := []string{"guard"}
	if report.Counts.Unexpected > 0 || report.Status == "FAIL" {
		status = "action"
		tags = append(tags, "proof-gap")
	} else if report.Counts.Deny == 0 && report.Counts.Quarantine == 0 {
		status = "warn"
		tags = append(tags, "no-deny-proof")
	}
	return tuiOverviewCard{
		Pane:      "guard",
		Status:    status,
		Source:    report.Source,
		Summary:   fmt.Sprintf("artifacts=%d deny=%d policy_block=%d unexpected=%d", report.Counts.Artifacts, report.Counts.Deny, report.Counts.PolicyBlock, report.Counts.Unexpected),
		Command:   "fak console guard --guard-json <artifact>",
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func missingOverviewCard(pane, command string) tuiOverviewCard {
	return tuiOverviewCard{
		Pane:    pane,
		Status:  "missing",
		Summary: "no source selected",
		Command: command,
		Tags:    []string{"missing-source"},
	}
}

func countTUIOverview(cards []tuiOverviewCard) tuiOverviewCounts {
	c := tuiOverviewCounts{Cards: len(cards)}
	for _, card := range cards {
		switch card.Status {
		case "ok":
			c.OK++
		case "action":
			c.Action++
		case "warn":
			c.Warn++
		case "missing":
			c.Missing++
		}
	}
	return c
}

func overviewActions(cards []tuiOverviewCard) []tuiOverviewAction {
	actions := []tuiOverviewAction{}
	for _, card := range cards {
		switch card.Status {
		case "action":
			actions = append(actions, tuiOverviewAction{Pane: card.Pane, Command: card.Command, Reason: card.Summary})
		case "missing":
			actions = append(actions, tuiOverviewAction{Pane: card.Pane, Command: card.Command, Reason: "add this pane's source to the overview"})
		}
	}
	return actions
}

func renderTUIAgent(report tuiAgentReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console agent  at=%s  backend=%s  provider=%s  mode=dry-run\n", report.At, report.Backend, report.Provider)
	fmt.Fprintf(&b, "auth=%s  session=%s", report.Auth, report.SessionID)
	if report.GatewayURL != "" {
		fmt.Fprintf(&b, "  gateway=%s", report.GatewayURL)
	}
	if report.Account != "" {
		fmt.Fprintf(&b, "  account=%s", report.Account)
	}
	if report.ResolvedAccount != "" && report.ResolvedAccount != report.Account {
		fmt.Fprintf(&b, "->%s", report.ResolvedAccount)
	}
	fmt.Fprintln(&b)
	if report.ClaudeConfigDir != "" {
		fmt.Fprintf(&b, "claude_config=%s  source=%s\n", trimTUI(report.ClaudeConfigDir, maxTUI(20, width-31)), report.ConfigSource)
	}
	if report.AccountIdentity != "" {
		fmt.Fprintf(&b, "identity=%s\n", report.AccountIdentity)
	}
	if report.Policy != "" || report.Model != "" || report.ContextBudget > 0 || report.RestartOnBudget || report.DebugStats || report.CompactHistoryLimit > 0 {
		label := "guard_options"
		if report.Provider == "existing-fak-gateway" {
			label = "agent_options"
		}
		fmt.Fprintf(&b, "%s policy=%s model=%s context=%d restart=%v limit=%d\n",
			label, blankTUI(report.Policy), blankTUI(report.Model), report.ContextBudget, report.RestartOnBudget, report.RestartLimit)
		if report.CompactHistoryLimit > 0 || report.ElideResultBytes > 0 || report.DebugStats {
			fmt.Fprintf(&b, "token_savings compact_history=%d elide_result=%d debug_stats=%v\n",
				report.CompactHistoryLimit, report.ElideResultBytes, report.DebugStats)
		}
	}
	if len(report.Env) > 0 {
		fmt.Fprintln(&b, "\nEnv")
		for _, kv := range report.Env {
			fmt.Fprintf(&b, "%-18s %-12s %s\n", kv.Name, kv.Source, trimTUI(displayTUIAgentEnvValue(kv), maxTUI(20, width-33)))
		}
	}
	fmt.Fprintln(&b, "\nBackend Command")
	fmt.Fprintf(&b, "  %s\n", trimTUI(shellLineTUI(report.Command), maxTUI(20, width-2)))
	fmt.Fprintln(&b, "\nLaunch")
	fmt.Fprintf(&b, "  %s\n", trimTUI(shellLineTUI(report.Launch), maxTUI(20, width-2)))
	if len(report.Notes) > 0 {
		fmt.Fprintln(&b, "\nNotes")
		for _, note := range report.Notes {
			fmt.Fprintf(&b, "- %s\n", trimTUI(note, maxTUI(20, width-2)))
		}
	}
	return b.String()
}

func displayTUIAgentEnvValue(kv tuiAgentEnv) string {
	if kv.Sensitive {
		if kv.FromEnv != "" {
			return "<redacted from " + kv.FromEnv + ">"
		}
		return "<redacted>"
	}
	if kv.FromEnv != "" && kv.Value == "" {
		return "$" + kv.FromEnv
	}
	return kv.Value
}

func renderTUIOverview(report tuiOverviewReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console overview  at=%s  source=%s\n", report.At, report.Source)
	fmt.Fprintf(&b, "cards=%d  ok=%d  action=%d  warn=%d  missing=%d\n",
		report.Counts.Cards, report.Counts.OK, report.Counts.Action, report.Counts.Warn, report.Counts.Missing)
	fmt.Fprintln(&b, "\nPanes")
	fmt.Fprintln(&b, "attention pane       status   tags                 summary")
	for _, card := range report.Cards {
		fmt.Fprintf(&b, "%9d %-10s %-8s %-20s %s\n",
			card.Attention, card.Pane, card.Status, trimTUI(displayTUITags(card.Tags, 3), 20),
			trimTUI(card.Summary, maxTUI(20, width-53)))
	}
	if len(report.Actions) > 0 {
		fmt.Fprintln(&b, "\nNext")
		limit := minTUI(len(report.Actions), 8)
		for _, action := range report.Actions[:limit] {
			fmt.Fprintf(&b, "%-10s %s\n", action.Pane, trimTUI(action.Command, maxTUI(20, width-11)))
		}
	}
	return b.String()
}

func buildTUISessionReport(list gateway.SessionListResponse, source string, at time.Time) tuiSessionReport {
	rows := make([]tuiSessionRow, 0, len(list.Sessions))
	for _, st := range list.Sessions {
		rows = append(rows, classifyTUISession(st))
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Attention != rows[j].Attention {
			return rows[i].Attention > rows[j].Attention
		}
		if rows[i].Priority != rows[j].Priority {
			return rows[i].Priority < rows[j].Priority
		}
		return rows[i].TraceID < rows[j].TraceID
	})
	return tuiSessionReport{
		Schema: tuiSessionsSchema,
		At:     at.UTC().Format(time.RFC3339),
		Source: source,
		Counts: countTUISessions(rows),
		Lanes:  buildTUISessionLanes(rows),
		Rows:   rows,
	}
}

func classifyTUISession(st gateway.SessionState) tuiSessionRow {
	run := strings.TrimSpace(st.Run)
	if run == "" {
		run = "running"
	}
	row := tuiSessionRow{
		TraceID:           st.TraceID,
		Run:               run,
		Priority:          st.Priority,
		Rev:               st.Rev,
		Reason:            st.Reason,
		TurnsLeft:         st.Budget.TurnsLeft,
		TokensLeft:        st.Budget.TokensLeft,
		ContextTokensLeft: st.Budget.ContextTokensLeft,
		MaxTokensPerTurn:  st.Pace.MaxTokensPerTurn,
		MinTurnGapMs:      st.Pace.MinTurnGapMs,
		ContinuationID:    st.ContinuationID,
		ParentTrace:       st.ParentTrace,
		Generation:        st.Generation,
	}
	row.Tags, row.Attention = scoreTUISession(row)
	return row
}

func scoreTUISession(row tuiSessionRow) ([]string, int) {
	tags := []string{}
	score := 0
	switch strings.ToLower(row.Run) {
	case "running":
		tags = append(tags, "running")
	case "throttled":
		tags = append(tags, "throttled")
		score += 35
	case "paused":
		tags = append(tags, "paused")
		score += 55
	case "draining":
		tags = append(tags, "draining")
		score += 65
	case "stopped":
		tags = append(tags, "stopped")
		score += 80
	default:
		tags = append(tags, "unknown-run")
		score += 30
	}
	if row.Reason != "" {
		tags = append(tags, "reason")
		score += 10
	}
	if row.TurnsLeft >= 0 {
		if row.TurnsLeft <= 1 {
			tags = append(tags, "low-turns")
			score += 45
		}
		tags = append(tags, "turn-budget")
	}
	if row.TokensLeft >= 0 {
		if row.TokensLeft <= 1000 {
			tags = append(tags, "low-tokens")
			score += 35
		}
		tags = append(tags, "token-budget")
	}
	if row.ContextTokensLeft > 0 {
		if row.ContextTokensLeft <= 2000 {
			tags = append(tags, "low-context")
			score += 25
		}
		tags = append(tags, "context-budget")
	}
	if row.MaxTokensPerTurn > 0 || row.MinTurnGapMs > 0 {
		tags = append(tags, "paced")
	}
	if row.ParentTrace != "" || row.ContinuationID != "" || row.Generation > 0 {
		tags = append(tags, "lineage")
	}
	return tags, score
}

func countTUISessions(rows []tuiSessionRow) tuiSessionCounts {
	var c tuiSessionCounts
	for _, row := range rows {
		c.Sessions++
		switch strings.ToLower(row.Run) {
		case "running":
			c.Running++
		case "throttled":
			c.Throttled++
		case "paused":
			c.Paused++
		case "draining":
			c.Draining++
		case "stopped":
			c.Stopped++
		}
		if row.TurnsLeft >= 0 || row.TokensLeft >= 0 || row.ContextTokensLeft > 0 {
			c.Budgeted++
		}
		if row.ContextTokensLeft > 0 {
			c.ContextBudget++
		}
		if row.ParentTrace != "" || row.ContinuationID != "" || row.Generation > 0 {
			c.Lineage++
		}
		if row.Reason != "" {
			c.WithReason++
		}
	}
	return c
}

func buildTUISessionLanes(rows []tuiSessionRow) []tuiSessionLane {
	names := []string{"running", "throttled", "paused", "draining", "stopped", "other"}
	lanes := make([]tuiSessionLane, 0, len(names))
	for _, name := range names {
		lane := tuiSessionLane{Name: name}
		for _, row := range rows {
			if !rowInTUISessionLane(row, name) {
				continue
			}
			lane.Count++
			if lane.TopSession == "" {
				lane.TopSession = row.TraceID
				lane.TopSummary = sessionSummary(row)
			}
		}
		lanes = append(lanes, lane)
	}
	return lanes
}

func rowInTUISessionLane(row tuiSessionRow, lane string) bool {
	run := strings.ToLower(row.Run)
	if lane == "other" {
		switch run {
		case "running", "throttled", "paused", "draining", "stopped":
			return false
		default:
			return true
		}
	}
	return run == lane
}

func renderTUISessions(report tuiSessionReport, top, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console sessions  at=%s  source=%s\n", report.At, report.Source)
	fmt.Fprintf(&b, "sessions=%d  running=%d  throttled=%d  paused=%d  draining=%d  stopped=%d  budgeted=%d  context=%d  lineage=%d\n",
		report.Counts.Sessions, report.Counts.Running, report.Counts.Throttled, report.Counts.Paused,
		report.Counts.Draining, report.Counts.Stopped, report.Counts.Budgeted, report.Counts.ContextBudget, report.Counts.Lineage)
	if len(report.Rows) == 0 {
		fmt.Fprintln(&b, "\nno sessions found")
		return b.String()
	}
	fmt.Fprintln(&b, "\nLanes")
	fmt.Fprintln(&b, "lane             count top")
	for _, lane := range report.Lanes {
		topText := "-"
		if lane.TopSession != "" {
			topText = lane.TopSession
			if lane.TopSummary != "" {
				topText += " " + lane.TopSummary
			}
		}
		fmt.Fprintf(&b, "%-16s %5d %s\n", lane.Name, lane.Count, trimTUI(topText, maxTUI(20, width-24)))
	}
	fmt.Fprintln(&b, "\nSession Queue")
	renderTUISessionRows(&b, report.Rows, minTUI(top, len(report.Rows)), width)
	return b.String()
}

func renderTUISessionRows(b *strings.Builder, rows []tuiSessionRow, limit, width int) {
	fmt.Fprintln(b, "attention session                    run        prio rev budget                         pace          tags")
	for _, row := range rows[:limit] {
		budget := fmt.Sprintf("t=%s tok=%s ctx=%s",
			budgetAxis(row.TurnsLeft), budgetAxis(row.TokensLeft), contextBudgetAxis(row.ContextTokensLeft))
		pace := fmt.Sprintf("max=%d gap=%d", row.MaxTokensPerTurn, row.MinTurnGapMs)
		summary := displayTUITags(row.Tags, 3)
		fmt.Fprintf(b, "%9d %-26s %-10s %4d %-3d %-30s %-13s %s\n",
			row.Attention, trimTUI(row.TraceID, 26), trimTUI(row.Run, 10), row.Priority, row.Rev,
			trimTUI(budget, 30), trimTUI(pace, 13), trimTUI(summary, maxTUI(14, width-95)))
	}
}

func displayTUITags(tags []string, limit int) string {
	if len(tags) == 0 {
		return "-"
	}
	if limit <= 0 || len(tags) <= limit {
		return strings.Join(tags, ",")
	}
	return strings.Join(tags[:limit], ",")
}

func sessionSummary(row tuiSessionRow) string {
	parts := []string{}
	if row.Reason != "" {
		parts = append(parts, row.Reason)
	}
	if row.TurnsLeft >= 0 {
		parts = append(parts, "turns="+budgetAxis(row.TurnsLeft))
	}
	if row.TokensLeft >= 0 {
		parts = append(parts, "tokens="+budgetAxis(row.TokensLeft))
	}
	if row.ContextTokensLeft > 0 {
		parts = append(parts, "context="+contextBudgetAxis(row.ContextTokensLeft))
	}
	if len(parts) == 0 {
		parts = append(parts, strings.Join(row.Tags, ","))
	}
	return strings.Join(parts, " ")
}
