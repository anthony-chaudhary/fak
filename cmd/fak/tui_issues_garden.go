package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
)

func loadTUIGarden(path, workspace string, deep bool, timeout time.Duration) (gardenbundle.Payload, string, error) {
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return gardenbundle.Payload{}, "", err
		}
		payload, err := decodeTUIGarden(b)
		return payload, path, err
	}
	root := workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	commit := gardenbundle.HeadCommit(root)
	if gardenbundle.GardenOff() {
		return gardenbundle.SkippedPayload(root, commit), "live:garden-skipped", nil
	}
	results := gardenbundle.Collect(root, "", timeout, deep)
	return gardenbundle.Fold(results, root, commit), "live:garden-bundle", nil
}

func decodeTUIGarden(b []byte) (gardenbundle.Payload, error) {
	var raw struct {
		Schema     string `json:"schema"`
		OK         bool   `json:"ok"`
		Verdict    string `json:"verdict"`
		Finding    string `json:"finding"`
		Reason     string `json:"reason"`
		NextAction string `json:"next_action"`
		Workspace  string `json:"workspace"`
		Commit     string `json:"commit"`
		Members    []struct {
			Key      string         `json:"key"`
			Label    string         `json:"label"`
			Gates    bool           `json:"gates"`
			ExitCode int            `json:"exit_code"`
			State    string         `json:"state"`
			OK       bool           `json:"ok"`
			Verdict  string         `json:"verdict"`
			Detail   string         `json:"detail"`
			Counts   map[string]int `json:"counts"`
		} `json:"members"`
		MemberCount int      `json:"member_count"`
		Gating      []string `json:"gating"`
		Skipped     bool     `json:"skipped"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return gardenbundle.Payload{}, fmt.Errorf("garden JSON must be a fak garden envelope: %w", err)
	}
	if raw.Schema != "" && raw.Schema != gardenbundle.Schema {
		return gardenbundle.Payload{}, fmt.Errorf("garden JSON schema = %q, want %q", raw.Schema, gardenbundle.Schema)
	}
	members := make([]gardenbundle.MemberResult, 0, len(raw.Members))
	for _, m := range raw.Members {
		members = append(members, gardenbundle.MemberResult{
			Key:      m.Key,
			Label:    m.Label,
			Gates:    m.Gates,
			ExitCode: m.ExitCode,
			State:    m.State,
			OK:       m.OK,
			Verdict:  m.Verdict,
			Detail:   m.Detail,
			Counts:   m.Counts,
		})
	}
	if raw.MemberCount == 0 {
		raw.MemberCount = len(members)
	}
	return gardenbundle.Payload{
		OK:          raw.OK,
		Verdict:     raw.Verdict,
		Finding:     raw.Finding,
		Reason:      raw.Reason,
		NextAction:  raw.NextAction,
		Workspace:   raw.Workspace,
		Commit:      raw.Commit,
		Members:     members,
		MemberCount: raw.MemberCount,
		Gating:      raw.Gating,
		Skipped:     raw.Skipped,
	}, nil
}

func loadTUIGuard(paths []string) ([]tuiGuardArtifact, error) {
	artifacts := make([]tuiGuardArtifact, 0, len(paths))
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		var raw map[string]any
		if err := json.Unmarshal(b, &raw); err != nil {
			return nil, fmt.Errorf("%s: guard JSON must be an object: %w", path, err)
		}
		artifacts = append(artifacts, tuiGuardArtifact{Path: path, Raw: raw})
	}
	return artifacts, nil
}

func loadTUIIssues(path, repo, state string, limit int) ([]tuiIssue, string, error) {
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, "", err
		}
		issues, err := decodeTUIIssues(b)
		return issues, path, err
	}
	args := []string{
		"issue", "list",
		"--state", state,
		"--limit", strconv.Itoa(limit),
		"--json", "number,title,url,state,body,labels,createdAt,updatedAt,author,assignees,milestone,comments",
	}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	cmd := exec.Command("gh", args...)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, "", fmt.Errorf("gh %s: %s", strings.Join(args, " "), msg)
	}
	issues, err := decodeTUIIssues(out)
	if err != nil {
		return nil, "", err
	}
	source := "gh issue list"
	if repo != "" {
		source += " --repo " + repo
	}
	return issues, source, nil
}

func decodeTUIIssues(b []byte) ([]tuiIssue, error) {
	var issues []tuiIssue
	if err := json.Unmarshal(b, &issues); err != nil {
		return nil, fmt.Errorf("issue JSON must be a gh issue list array: %w", err)
	}
	for i := range issues {
		if issues[i].State == "" {
			issues[i].State = "OPEN"
		}
	}
	return issues, nil
}

func buildTUIIssueReport(issues []tuiIssue, source string, asOf time.Time, epic int) tuiIssueReport {
	dups := tuiDuplicateGroups(issues)
	rows := make([]tuiIssueRow, 0, len(issues))
	var epicRow *tuiIssueRow
	for _, issue := range issues {
		row := classifyTUIIssue(issue, asOf, dups)
		if epic > 0 {
			row.Related = issue.Number == epic || tuiIssueReferences(issue, epic)
		}
		if issue.Number == epic {
			cp := row
			epicRow = &cp
		}
		rows = append(rows, row)
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Score != rows[j].Score {
			return rows[i].Score > rows[j].Score
		}
		return rows[i].Number > rows[j].Number
	})
	return tuiIssueReport{
		Schema:  tuiIssuesSchema,
		AsOf:    asOf.Format("2006-01-02"),
		Source:  source,
		Epic:    epicRow,
		Counts:  countTUIIssues(rows),
		Lanes:   buildTUILanes(rows),
		Rows:    rows,
		Actions: buildTUIActions(rows),
	}
}

func classifyTUIIssue(issue tuiIssue, asOf time.Time, dups map[int]int) tuiIssueRow {
	labels := tuiLabelNames(issue)
	labelSet := map[string]bool{}
	for _, label := range labels {
		labelSet[label] = true
	}
	prio := ""
	for _, p := range []string{"priority/P0", "priority/P1", "priority/P2"} {
		if labelSet[p] {
			prio = p
			break
		}
	}
	assigned := len(issue.Assignees) > 0
	inProgress := labelSet["in-progress"]
	ageDays := tuiDaysSince(issue.CreatedAt, asOf)
	idleDays := tuiDaysSince(issue.UpdatedAt, asOf)
	tags := []string{}
	if prio == "" {
		tags = append(tags, "needs-priority")
	}
	if !tuiHasAny(labelSet, tuiKindLabels) {
		tags = append(tags, "needs-kind")
	}
	if !tuiHasAny(labelSet, tuiAreaLabels) {
		tags = append(tags, "needs-area")
	}
	if len(labels) == 0 {
		tags = append(tags, "bare")
	}
	if (prio == "priority/P0" || prio == "priority/P1") && !inProgress && !assigned {
		tags = append(tags, "orphan")
	}
	if idleDays >= 60 && !inProgress {
		tags = append(tags, "stale")
	}
	if labelSet["question"] && idleDays >= 30 {
		tags = append(tags, "dormant-question")
	}
	if _, ok := dups[issue.Number]; ok {
		tags = append(tags, "likely-dup")
	}

	score := tuiPriorityWeights[prio]
	if score == 0 {
		score = 60
	}
	if (prio == "priority/P0" || prio == "priority/P1") && !inProgress && !assigned {
		score += 300
	}
	if labelSet["bug"] {
		score += 40
	}
	if labelSet["documentation"] {
		score -= 20
	}
	if idleDays > 90 {
		score += 90
	} else {
		score += idleDays
	}
	if labelSet["question"] && idleDays < 30 {
		score -= 200
	}

	return tuiIssueRow{
		Number:     issue.Number,
		Title:      issue.Title,
		URL:        issue.URL,
		State:      issue.State,
		Labels:     labels,
		Author:     tuiLogin(issue.Author),
		Assignees:  tuiAssigneeLogins(issue.Assignees),
		Milestone:  tuiMilestoneTitle(issue.Milestone),
		Comments:   int(issue.Comments),
		AgeDays:    ageDays,
		IdleDays:   idleDays,
		Priority:   prio,
		InProgress: inProgress,
		Tags:       tags,
		Score:      score,
	}
}

func tuiLabelNames(issue tuiIssue) []string {
	labels := make([]string, 0, len(issue.Labels))
	seen := map[string]bool{}
	for _, label := range issue.Labels {
		name := strings.TrimSpace(label.Name)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		labels = append(labels, name)
	}
	sort.Strings(labels)
	return labels
}

func tuiHasAny(labels map[string]bool, allowed map[string]bool) bool {
	for label := range labels {
		if allowed[label] {
			return true
		}
	}
	return false
}

func tuiDaysSince(iso string, asOf time.Time) int {
	if strings.TrimSpace(iso) == "" {
		return 0
	}
	t, err := time.Parse(time.RFC3339, iso)
	if err != nil {
		return 0
	}
	days := int(asOf.Sub(t.UTC()).Hours() / 24)
	if days < 0 {
		return 0
	}
	return days
}

func tuiLogin(u *tuiUser) string {
	if u == nil {
		return ""
	}
	return u.Login
}

func tuiAssigneeLogins(users []tuiUser) []string {
	out := make([]string, 0, len(users))
	for _, u := range users {
		if u.Login != "" {
			out = append(out, u.Login)
		}
	}
	sort.Strings(out)
	return out
}

func tuiMilestoneTitle(m *tuiMilestone) string {
	if m == nil {
		return ""
	}
	return m.Title
}

func countTUIIssues(rows []tuiIssueRow) tuiIssueCounts {
	var c tuiIssueCounts
	for _, row := range rows {
		if strings.EqualFold(row.State, "closed") {
			continue
		}
		c.Open++
		switch row.Priority {
		case "priority/P0":
			c.P0++
		case "priority/P1":
			c.P1++
		case "priority/P2":
			c.P2++
		}
		if row.Related {
			c.Related++
		}
		for _, tag := range row.Tags {
			switch tag {
			case "needs-priority":
				c.NeedsPriority++
			case "needs-kind":
				c.NeedsKind++
			case "needs-area":
				c.NeedsArea++
			case "orphan":
				c.Orphan++
			case "stale":
				c.Stale++
			case "dormant-question":
				c.DormantQuestion++
			case "likely-dup":
				c.LikelyDup++
			case "bare":
				c.Bare++
			}
		}
	}
	return c
}

func buildTUILanes(rows []tuiIssueRow) []tuiLane {
	names := []string{"priority/P0", "priority/P1", "priority/P2", "unprioritized"}
	lanes := make([]tuiLane, 0, len(names))
	for _, name := range names {
		lane := tuiLane{Name: name}
		for _, row := range rows {
			if row.Priority != name && !(name == "unprioritized" && row.Priority == "") {
				continue
			}
			lane.Count++
			if tuiHasTag(row, "orphan") {
				lane.Orphan++
			}
			if tuiHasTag(row, "needs-area") {
				lane.NeedsArea++
			}
			if tuiHasTag(row, "needs-kind") {
				lane.NeedsKind++
			}
			if row.IdleDays > lane.MaxIdleDays {
				lane.MaxIdleDays = row.IdleDays
			}
			if lane.TopIssue == 0 {
				lane.TopIssue = row.Number
				lane.TopIssueText = row.Title
			}
		}
		lanes = append(lanes, lane)
	}
	return lanes
}

func buildTUIActions(rows []tuiIssueRow) []tuiIssueAction {
	actions := []tuiIssueAction{}
	for _, row := range rows {
		switch {
		case tuiHasTag(row, "dormant-question"):
			actions = append(actions, tuiIssueAction{
				Number: row.Number,
				Kind:   "close-dormant-question",
				Reason: fmt.Sprintf("question idle %dd", row.IdleDays),
				Command: fmt.Sprintf("gh issue close %d --reason \"not planned\" --comment \"Closing as dormant: question idle %dd. Reopen with new info if it is still live.\"",
					row.Number, row.IdleDays),
			})
		case tuiHasTag(row, "stale") && row.Priority != "priority/P0" && row.Priority != "priority/P1":
			actions = append(actions, tuiIssueAction{
				Number:  row.Number,
				Kind:    "mark-stale",
				Reason:  fmt.Sprintf("idle %dd, not in-progress, not P0/P1", row.IdleDays),
				Command: fmt.Sprintf("gh issue edit %d --add-label \"stale\"", row.Number),
			})
		case len(row.Tags) > 0:
			actions = append(actions, tuiIssueAction{
				Number: row.Number,
				Kind:   "review",
				Reason: strings.Join(row.Tags, ", "),
			})
		}
	}
	return actions
}

func tuiHasTag(row tuiIssueRow, tag string) bool {
	for _, got := range row.Tags {
		if got == tag {
			return true
		}
	}
	return false
}

func tuiIssueReferences(issue tuiIssue, epic int) bool {
	ref := "#" + strconv.Itoa(epic)
	return strings.Contains(issue.Title, ref) || strings.Contains(issue.Body, ref)
}

func tuiDuplicateGroups(issues []tuiIssue) map[int]int {
	type pair struct {
		num int
		tok map[string]bool
	}
	pairs := make([]pair, 0, len(issues))
	for _, issue := range issues {
		pairs = append(pairs, pair{num: issue.Number, tok: tuiTitleTokens(issue.Title)})
	}
	parent := map[int]int{}
	for _, p := range pairs {
		parent[p.num] = p.num
	}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
	}
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ {
			if tuiJaccard(pairs[i].tok, pairs[j].tok) >= 0.60 {
				union(pairs[i].num, pairs[j].num)
			}
		}
	}
	members := map[int][]int{}
	for _, p := range pairs {
		root := find(p.num)
		members[root] = append(members[root], p.num)
	}
	out := map[int]int{}
	gid := 0
	for _, nums := range members {
		if len(nums) < 2 {
			continue
		}
		for _, n := range nums {
			out[n] = gid
		}
		gid++
	}
	return out
}

func tuiTitleTokens(title string) map[string]bool {
	stop := map[string]bool{
		"the": true, "and": true, "for": true, "with": true, "issue": true,
		"feat": true, "fix": true, "add": true, "new": true, "needs": true,
		"work": true, "support": true, "implement": true,
	}
	out := map[string]bool{}
	for _, m := range tuiScopeRE.FindAllStringSubmatch(title, -1) {
		if len(m) == 3 {
			out[strings.ToLower(m[0])] = true
			out[strings.ToLower(m[2])] = true
		}
	}
	for _, word := range tuiWordRE.FindAllString(title, -1) {
		w := strings.ToLower(word)
		if !stop[w] {
			out[w] = true
		}
	}
	return out
}

func tuiJaccard(a, b map[string]bool) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	inter := 0
	for k := range a {
		if b[k] {
			inter++
		}
	}
	union := len(a)
	for k := range b {
		if !a[k] {
			union++
		}
	}
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

func buildTUIGardenReport(payload gardenbundle.Payload, source string, at time.Time, includeGate bool) tuiGardenReport {
	rows := make([]tuiGardenRow, 0, len(payload.Members))
	for _, member := range payload.Members {
		rows = append(rows, classifyTUIGardenMember(member))
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Attention != rows[j].Attention {
			return rows[i].Attention > rows[j].Attention
		}
		return rows[i].Key < rows[j].Key
	})
	counts := countTUIGarden(rows)
	if payload.Skipped {
		counts.Skipped = 1
	}
	gateExit := 0
	gateMessage := ""
	if includeGate {
		gateExit, gateMessage = gardenbundle.CheckGate(payload)
	}
	return tuiGardenReport{
		Schema:      tuiGardenSchema,
		At:          at.UTC().Format(time.RFC3339),
		Source:      source,
		Workspace:   payload.Workspace,
		Commit:      payload.Commit,
		OK:          payload.OK,
		Verdict:     payload.Verdict,
		Finding:     payload.Finding,
		Reason:      payload.Reason,
		NextAction:  payload.NextAction,
		GateExit:    gateExit,
		GateMessage: gateMessage,
		Counts:      counts,
		Rows:        rows,
	}
}

func classifyTUIGardenMember(member gardenbundle.MemberResult) tuiGardenRow {
	row := tuiGardenRow{
		Key:      member.Key,
		Label:    member.Label,
		State:    member.State,
		OK:       member.OK,
		Gates:    member.Gates,
		ExitCode: member.ExitCode,
		Verdict:  member.Verdict,
		Detail:   member.Detail,
		Counts:   member.Counts,
	}
	row.Tags, row.Attention = scoreTUIGardenRow(row)
	return row
}

func scoreTUIGardenRow(row tuiGardenRow) ([]string, int) {
	tags := []string{}
	score := 0
	switch row.State {
	case "errored":
		tags = append(tags, "errored")
		score += 100
	case "red":
		tags = append(tags, "red")
		score += 90
	case "action":
		tags = append(tags, "action")
		score += 55
	case "ok":
		tags = append(tags, "ok")
	default:
		tags = append(tags, "unknown")
		score += 20
	}
	if row.Gates {
		tags = append(tags, "gates")
		score += 20
	}
	if row.ExitCode != 0 {
		tags = append(tags, "nonzero-exit")
		score += 10
	}
	if row.Counts != nil {
		if row.Counts["broken"] > 0 {
			tags = append(tags, "broken-loops")
			score += row.Counts["broken"] * 20
		}
		if row.Counts["action"] > 0 {
			tags = append(tags, "loop-action")
			score += row.Counts["action"] * 10
		}
	}
	return tags, score
}

func countTUIGarden(rows []tuiGardenRow) tuiGardenCounts {
	var c tuiGardenCounts
	for _, row := range rows {
		c.Members++
		if row.Gates {
			c.Gating++
		}
		switch row.State {
		case "ok":
			c.OK++
		case "action":
			c.Action++
		case "red":
			c.Red++
		case "errored":
			c.Errored++
		}
	}
	return c
}

func renderTUIGarden(report tuiGardenReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console garden  at=%s  source=%s\n", report.At, report.Source)
	fmt.Fprintf(&b, "verdict=%s  finding=%s  ok=%v  members=%d  action=%d  red=%d  errored=%d  gating=%d\n",
		report.Verdict, report.Finding, report.OK, report.Counts.Members, report.Counts.Action,
		report.Counts.Red, report.Counts.Errored, report.Counts.Gating)
	if report.GateMessage != "" {
		fmt.Fprintf(&b, "gate=%d  %s\n", report.GateExit, trimTUI(report.GateMessage, width-8))
	}
	if report.Workspace != "" || report.Commit != "" {
		fmt.Fprintf(&b, "workspace=%s  commit=%s\n", report.Workspace, report.Commit)
	}
	if report.Reason != "" {
		fmt.Fprintf(&b, "reason: %s\n", trimTUI(report.Reason, maxTUI(20, width-8)))
	}
	if report.NextAction != "" {
		fmt.Fprintf(&b, "next:   %s\n", trimTUI(report.NextAction, maxTUI(20, width-8)))
	}
	if len(report.Rows) == 0 {
		if report.Counts.Skipped > 0 {
			fmt.Fprintln(&b, "\n(skipped)")
		} else {
			fmt.Fprintln(&b, "\nno garden members")
		}
		return b.String()
	}
	fmt.Fprintln(&b, "\nMembers")
	fmt.Fprintln(&b, "attention member                    state    gate exit verdict tags")
	for _, row := range report.Rows {
		gate := "-"
		if row.Gates {
			gate = "yes"
		}
		tags := displayTUITags(row.Tags, 4)
		detail := row.Detail
		if detail != "" {
			tags += "  " + detail
		}
		fmt.Fprintf(&b, "%9d %-25s %-8s %-4s %-4d %-7s %s\n",
			row.Attention, trimTUI(row.Label, 25), trimTUI(row.State, 8), gate, row.ExitCode,
			trimTUI(row.Verdict, 7), trimTUI(tags, maxTUI(14, width-66)))
	}
	return b.String()
}
