package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/ghexec"
	"github.com/anthony-chaudhary/fak/internal/issuestriage"
	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

func init() {
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "issues",
		Summary: "fold GitHub issues into triage lanes, actions, and an optional epic focus",
		Usage:   "fak console issues [--issues-json FILE] [--repo owner/repo] [--state open|closed|all] [--top N] [--json]",
		Schema:  tuiIssuesSchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "as-of", Label: "As Of", Kind: "input", Flag: "--as-of", Detail: "date for age/idle math"},
			{ID: "epic", Label: "Epic", Kind: "input", Flag: "--epic", Detail: "highlight related issue work"},
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit the typed pane model"},
			{ID: "state", Label: "State", Kind: "flag", Flag: "--state", Default: "open", Options: []string{"open", "closed", "all"}},
			{ID: "top", Label: "Top Rows", Kind: "input", Flag: "--top", Default: "25"},
		},
		Run:      runTUIIssues,
		Overview: tuiOverviewAdapter(buildTUIOverviewIssueCard),
	})
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "garden",
		Summary: "render the read-only garden bundle and gate posture",
		Usage:   "fak console garden [--garden-json FILE] [--json] [--check] [--workspace DIR] [--deep]",
		Schema:  tuiGardenSchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "check", Label: "Check Gate", Kind: "toggle", Flag: "--check", Detail: "include the garden gate decision"},
			{ID: "deep", Label: "Deep", Kind: "toggle", Flag: "--deep", Detail: "run deeper read-only bundle checks"},
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit the typed pane model"},
			{ID: "timeout", Label: "Timeout", Kind: "input", Flag: "--timeout", Default: "20"},
			{ID: "workspace", Label: "Workspace", Kind: "input", Flag: "--workspace"},
		},
		Run:      runTUIGarden,
		Overview: tuiOverviewAdapter(buildTUIOverviewGardenCard),
	})
}

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

const tuiIssueSnapshotMaxAgeSeconds int64 = 15 * 60

type tuiIssueSnapshot struct {
	Issues []tuiIssue
	Census tuiIssueCensus
}

func runTUIIssueGH(args ...string) ([]byte, error) {
	cmd, cancel := ghexec.CommandTimeout(context.Background(), ghexec.DefaultTimeout, args...)
	defer cancel()
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err == nil {
		return out, nil
	}
	msg := strings.TrimSpace(stderr.String())
	if msg == "" {
		msg = err.Error()
	}
	return nil, fmt.Errorf("gh %s: %s", strings.Join(args, " "), msg)
}

func reconcileTUIIssueCensus(scope, state string, fetched, total int, ageSeconds int64, includesPullRequests bool) tuiIssueCensus {
	scope = strings.TrimSpace(scope)
	state = strings.ToLower(strings.TrimSpace(state))
	c := tuiIssueCensus{
		Scope:                scope,
		State:                state,
		FetchedCount:         fetched,
		TotalCount:           total,
		PageComplete:         fetched == total,
		SnapshotAgeSeconds:   ageSeconds,
		IncludesPullRequests: includesPullRequests,
		Reconciliation:       "complete",
	}
	switch {
	case includesPullRequests:
		c.Reconciliation = "pull_requests_included"
	case fetched < 0 || total < 0:
		c.Reconciliation = "count_mismatch"
	case scope != "repository_issues" && scope != "provided_issues" && !strings.HasPrefix(scope, "repository_issues:") && !strings.HasPrefix(scope, "provided_issues:"):
		c.Reconciliation = "scope_mismatch"
	case ageSeconds > tuiIssueSnapshotMaxAgeSeconds:
		c.Reconciliation = "snapshot_stale"
	case fetched < total:
		c.Reconciliation = "pagination_truncated"
	case fetched > total:
		c.Reconciliation = "count_mismatch"
	}
	return c
}

func loadTUIIssueSnapshot(path, repo, state string, limit int) (tuiIssueSnapshot, string, error) {
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return tuiIssueSnapshot{}, "", err
		}
		snapshot, err := decodeTUIIssueSnapshot(b, state, time.Now().UTC())
		return snapshot, path, err
	}
	repoName, err := resolveTUIIssueRepo(repo)
	if err != nil {
		return tuiIssueSnapshot{}, "", err
	}
	totalBefore, err := fetchTUIIssueTotal(repoName, state)
	if err != nil {
		return tuiIssueSnapshot{}, "", err
	}
	fetchLimit := limit
	if totalBefore < fetchLimit {
		fetchLimit = totalBefore
	}
	if fetchLimit < 1 {
		fetchLimit = 1
	}
	args := []string{
		"issue", "list",
		"--state", state,
		"--limit", strconv.Itoa(fetchLimit),
		"--json", "number,title,url,state,body,labels,createdAt,updatedAt,author,assignees,milestone,comments",
	}
	args = append(args, "--repo", repoName)
	out, err := runTUIIssueGH(args...)
	if err != nil {
		return tuiIssueSnapshot{}, "", err
	}
	issues, err := decodeTUIIssues(out)
	if err != nil {
		return tuiIssueSnapshot{}, "", err
	}
	totalAfter, err := fetchTUIIssueTotal(repoName, state)
	if err != nil {
		return tuiIssueSnapshot{}, "", err
	}
	snapshotAt := time.Now().UTC()
	census := reconcileTUIIssueCensus("repository_issues", state, len(issues), totalAfter, 0, false)
	census.SnapshotAt = snapshotAt.Format(time.RFC3339)
	return tuiIssueSnapshot{Issues: issues, Census: census}, "gh issue list --repo " + repoName, nil
}

func loadTUIIssues(path, repo, state string, limit int) ([]tuiIssue, string, error) {
	snapshot, source, err := loadTUIIssueSnapshot(path, repo, state, limit)
	if err != nil {
		return nil, "", err
	}
	if snapshot.Census.Reconciliation != "complete" || !snapshot.Census.PageComplete {
		return nil, "", fmt.Errorf("ranking refused: %s (scope=%s fetched=%d total=%d age=%ds)",
			snapshot.Census.Reconciliation, snapshot.Census.Scope,
			snapshot.Census.FetchedCount, snapshot.Census.TotalCount,
			snapshot.Census.SnapshotAgeSeconds)
	}
	if snapshot.Census.FetchedCount != len(snapshot.Issues) {
		return nil, "", fmt.Errorf("ranking refused: count_mismatch (decoded=%d fetched=%d total=%d)",
			len(snapshot.Issues), snapshot.Census.FetchedCount, snapshot.Census.TotalCount)
	}
	return snapshot.Issues, source, nil
}

func resolveTUIIssueRepo(repo string) (string, error) {
	repo = strings.TrimSpace(repo)
	if repo == "" {
		out, err := runTUIIssueGH("repo", "view", "--json", "nameWithOwner")
		if err != nil {
			return "", err
		}
		var v struct {
			NameWithOwner string `json:"nameWithOwner"`
		}
		if err := json.Unmarshal(out, &v); err != nil {
			return "", fmt.Errorf("gh repo view JSON: %w", err)
		}
		repo = v.NameWithOwner
	}
	parts := strings.Split(repo, "/")
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
		return "", fmt.Errorf("repository must be owner/name, got %q", repo)
	}
	return repo, nil
}

func fetchTUIIssueTotal(repo, state string) (int, error) {
	parts := strings.Split(repo, "/")
	if len(parts) != 2 {
		return 0, fmt.Errorf("repository must be owner/name, got %q", repo)
	}
	stateArg := ""
	switch strings.ToLower(strings.TrimSpace(state)) {
	case "open":
		stateArg = "states:OPEN"
	case "closed":
		stateArg = "states:CLOSED"
	case "all":
	case "":
		return 0, fmt.Errorf("issue state is required")
	default:
		return 0, fmt.Errorf("issue state must be open, closed, or all, got %q", state)
	}
	issueArgs := ""
	if stateArg != "" {
		issueArgs = "(" + stateArg + ")"
	}
	query := fmt.Sprintf("query($owner:String!,$name:String!){repository(owner:$owner,name:$name){issues%s{totalCount}}}", issueArgs)
	out, err := runTUIIssueGH(
		"api", "graphql", "-f", "query="+query,
		"-F", "owner="+parts[0], "-F", "name="+parts[1],
	)
	if err != nil {
		return 0, err
	}
	var v struct {
		Data struct {
			Repository struct {
				Issues struct {
					TotalCount int `json:"totalCount"`
				} `json:"issues"`
			} `json:"repository"`
		} `json:"data"`
	}
	if err := json.Unmarshal(out, &v); err != nil {
		return 0, fmt.Errorf("gh issue total JSON: %w", err)
	}
	return v.Data.Repository.Issues.TotalCount, nil
}

func decodeTUIIssueSnapshot(b []byte, state string, now time.Time) (tuiIssueSnapshot, error) {
	text := strings.TrimSpace(string(b))
	if strings.HasPrefix(text, "[") {
		issues, err := decodeTUIIssues(b)
		if err != nil {
			return tuiIssueSnapshot{}, err
		}
		census := reconcileTUIIssueCensus("provided_issues", state, len(issues), len(issues), 0, false)
		census.SnapshotAt = now.UTC().Format(time.RFC3339)
		return tuiIssueSnapshot{Issues: issues, Census: census}, nil
	}
	var envelope struct {
		Issues []tuiIssue      `json:"issues"`
		Census *tuiIssueCensus `json:"census"`
	}
	if err := json.Unmarshal(b, &envelope); err != nil {
		return tuiIssueSnapshot{}, fmt.Errorf("issue JSON must be an array or {issues,census} envelope: %w", err)
	}
	if envelope.Census == nil {
		return tuiIssueSnapshot{}, fmt.Errorf("issue JSON envelope must carry census metadata")
	}
	for i := range envelope.Issues {
		if envelope.Issues[i].State == "" {
			envelope.Issues[i].State = "OPEN"
		}
	}
	age := envelope.Census.SnapshotAgeSeconds
	if envelope.Census.SnapshotAt != "" {
		at, err := time.Parse(time.RFC3339, envelope.Census.SnapshotAt)
		if err != nil {
			return tuiIssueSnapshot{}, fmt.Errorf("census snapshot_at: %w", err)
		}
		age = int64(now.UTC().Sub(at.UTC()).Seconds())
		if age < 0 {
			age = 0
		}
	}
	census := reconcileTUIIssueCensus(
		envelope.Census.Scope, envelope.Census.State,
		envelope.Census.FetchedCount, envelope.Census.TotalCount,
		age, envelope.Census.IncludesPullRequests,
	)
	census.SnapshotAt = envelope.Census.SnapshotAt
	return tuiIssueSnapshot{Issues: envelope.Issues, Census: census}, nil
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
	census := reconcileTUIIssueCensus("provided_issues", "all", len(issues), len(issues), 0, false)
	report, _ := buildTUIIssueReportWithCensus(issues, source, asOf, epic, census, 50)
	return report
}

func buildTUIIssueReportWithCensus(issues []tuiIssue, source string, asOf time.Time, epic int, census tuiIssueCensus, repairBatchSize int) (tuiIssueReport, error) {
	if census.Reconciliation != "complete" || !census.PageComplete {
		return tuiIssueReport{}, fmt.Errorf("ranking refused: %s (scope=%s fetched=%d total=%d age=%ds)",
			census.Reconciliation, census.Scope, census.FetchedCount, census.TotalCount, census.SnapshotAgeSeconds)
	}
	if census.FetchedCount != len(issues) {
		return tuiIssueReport{}, fmt.Errorf("ranking refused: count_mismatch (decoded=%d fetched=%d total=%d)",
			len(issues), census.FetchedCount, census.TotalCount)
	}
	if repairBatchSize <= 0 {
		return tuiIssueReport{}, fmt.Errorf("repair batch size must be positive")
	}
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
	actions := buildTUIActions(rows)
	counts := countTUIIssues(rows)
	for _, a := range actions {
		if a.NeedsHuman {
			counts.NeedsYou++
		} else {
			counts.AgentClearable++
		}
	}
	return tuiIssueReport{
		Schema:        tuiIssuesSchema,
		AsOf:          asOf.Format("2006-01-02"),
		Source:        source,
		Census:        census,
		Epic:          epicRow,
		Counts:        counts,
		Lanes:         buildTUILanes(rows),
		Rows:          rows,
		Actions:       actions,
		RepairBatches: buildTUIRepairBatches(rows, repairBatchSize),
	}, nil
}

func buildTUIRepairBatches(rows []tuiIssueRow, batchSize int) []tuiIssueRepairBatch {
	axes := []struct {
		name string
		tag  string
	}{
		{name: "priority", tag: "needs-priority"},
		{name: "kind", tag: "needs-kind"},
		{name: "area", tag: "needs-area"},
	}
	var batches []tuiIssueRepairBatch
	for _, axis := range axes {
		var numbers []int
		for _, row := range rows {
			if tuiHasTag(row, axis.tag) {
				numbers = append(numbers, row.Number)
			}
		}
		for start, batch := 0, 1; start < len(numbers); start, batch = start+batchSize, batch+1 {
			end := start + batchSize
			if end > len(numbers) {
				end = len(numbers)
			}
			batches = append(batches, tuiIssueRepairBatch{
				Axis: axis.name, Batch: batch,
				Issues: append([]int(nil), numbers[start:end]...), ReviewOnly: true,
			})
		}
	}
	return batches
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
		score = dispatchtick.PriorityWeightDefault
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
	// Fold every surfaced action through the shared decenter-the-human triage so
	// the pane's "needs you" list is genuine-residual-only. Done as a post-pass
	// over the built actions (not per-branch) so the three action shapes route
	// through the one seam.
	for i := range actions {
		v := triageIssueAction(actions[i])
		actions[i].Disposition = string(v.Disposition)
		actions[i].NeedsHuman = v.NeedsHuman
		actions[i].Resolve = v.Resolve
	}
	return actions
}

// triageIssueAction folds one surfaced issue action through the shared
// internal/issuestriage leaf, so this pane, the garden walk, and the operator
// brief all classify an issue action in one tested place: a close-dormant-question
// / mark-stale action hands over a ready `gh` command -> TAKE_OBVIOUS; a review
// whose reason names an unset priority -> HUMAN_RESIDUAL (the PRIORITY authority a
// person holds); an unlabeled review (needs-kind/needs-area/likely-dup) is knowable
// but not obvious -> FRESH_CONTEXT. Only HUMAN_RESIDUAL sets NeedsHuman.
func triageIssueAction(a tuiIssueAction) choicetriage.Verdict {
	return issuestriage.Triage(issuestriage.Action{
		Number:  a.Number,
		Kind:    a.Kind,
		Reason:  a.Reason,
		Command: a.Command,
	})
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
		fmt.Fprintf(&b, "%9d %s %s %-4s %-4d %s %s\n",
			row.Attention,
			padRightTUI(trimTUI(row.Label, 25), 25),
			padRightTUI(trimTUI(row.State, 8), 8),
			gate, row.ExitCode,
			padRightTUI(trimTUI(row.Verdict, 7), 7),
			trimTUI(tags, maxTUI(14, width-66)))
	}
	return b.String()
}
