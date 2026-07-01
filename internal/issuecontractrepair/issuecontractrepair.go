package issuecontractrepair

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	Schema     = "fak.issue-contract-repair.v1"
	DefaultCap = 500
	MinScore   = 100
)

var reasonKind = map[string]string{
	issuecontract.ReasonNotDispatchLeaf:    "split",
	issuecontract.ReasonOversizedSteps:     "split",
	issuecontract.ReasonScopeIncomplete:    "scope",
	issuecontract.ReasonUnrouted:           "route",
	issuecontract.ReasonLiveUnarmored:      "noise",
	issuecontract.ReasonNoiseIncomplete:    "noise",
	issuecontract.ReasonAgentIncomplete:    "noise",
	issuecontract.ReasonPrivateBoundary:    "private",
	issuecontract.ReasonUnexpandedTemplate: "template",
}

var kindRank = map[string]int{"split": 1, "scope": 2, "route": 3, "noise": 4, "private": 5, "template": 6, "other": 9}

var kindAction = map[string]string{
	"split":    "decompose each non-leaf or oversized row into child issues within the dispatch step budget",
	"scope":    "add the missing parent/current-state/scope/done/witness/closure fields before dispatch",
	"route":    "add a lane or path hints section so the issue maps to one dispatch lane",
	"noise":    "add trigger, batch policy, agent context, and live dedupe/cap evidence before automated sync",
	"private":  "remove private/operator-only evidence or move the work to the private companion repo",
	"template": "dry-run a normalized generated-header repair, review it, then apply explicitly if accepted",
	"other":    "inspect the review reasons and repair the row before dispatch",
}

var fieldQuestion = map[string]string{
	"parent_ref":      "What epic/parent issue (if any) does this belong under?",
	"current_state":   "What is the current state of the code/system before this change?",
	"why_now":         "Why does this need to happen now, not later?",
	"working_spine":   "What is the smallest end-to-end path (the working spine) this change moves?",
	"in_scope":        "What is explicitly in scope for this issue?",
	"out_of_scope":    "What is explicitly out of scope (so a worker doesn't over-reach)?",
	"done_condition":  "What observable state means this issue is done?",
	"acceptance_gate": "What check/command proves the done condition is met?",
	"closure_binding": "What commit-message convention binds the closing commit to this issue (e.g. `#N` in the subject)?",
	"work_unit":       "What is the single unit of work here (one leaf, one commit)?",
	"expected_steps":  "Roughly how many discrete steps should this take (a size estimate)?",
	"assumptions":     "What is being assumed that, if wrong, would change the approach?",
	"confusion_risks": "What could a worker misunderstand or conflate here?",
	"coordination":    "Does this touch a lane/file another worker might also be editing?",
	"trigger":         "What event or condition should cause this to be picked up?",
	"batch_policy":    "Should this run standalone or as part of a batch with related issues?",
	"witness":         "What evidence (log, diff, test) proves this was actually done?",
}

type FieldPrompt struct {
	Field    string `json:"field"`
	Question string `json:"question"`
}

type RouteProposal struct {
	Lane        string
	BlockedLane string
	Confidence  string
}

type RepairRow struct {
	Number          int           `json:"number"`
	Title           string        `json:"title"`
	Score           int           `json:"score"`
	Reasons         []string      `json:"reasons"`
	Kinds           []string      `json:"kinds"`
	Kind            string        `json:"kind"`
	NextAction      string        `json:"next_action"`
	Ready           bool          `json:"ready"`
	ProposedLane    *string       `json:"proposed_lane"`
	RouteConfidence *string       `json:"route_confidence"`
	ProposedHeader  *string       `json:"proposed_header"`
	MissingFields   []FieldPrompt `json:"missing_fields"`
}

type Counts struct {
	CandidatesExamined int            `json:"candidates_examined"`
	NeedsRepair        int            `json:"needs_repair"`
	Ready              int            `json:"ready"`
	ByKind             map[string]int `json:"by_kind"`
}

type Manifest struct {
	Schema    string      `json:"schema"`
	AsOf      string      `json:"as_of"`
	Workspace string      `json:"workspace"`
	Lane      *string     `json:"lane"`
	Limit     int         `json:"limit"`
	Counts    Counts      `json:"counts"`
	Issues    []RepairRow `json:"issues"`
}

type Action struct {
	Number     int     `json:"number"`
	Kind       string  `json:"kind"`
	Ready      bool    `json:"ready"`
	Reason     string  `json:"reason"`
	NextAction string  `json:"next_action"`
	Cmd        *string `json:"cmd"`
}

type Options struct {
	Lane     string
	Limit    int
	AsOf     string
	MinScore int
	Review   func(issuecontract.IssueDraft) issuecontract.Review
	Route    func(issuecontract.IssueDraft, issuecontract.Review) RouteProposal
	Template func(issuecontract.IssueDraft) (issuecontract.TemplateRepairPlan, bool)
}

func RepairKinds(reasons []string) []string {
	var kinds []string
	seen := map[string]bool{}
	for _, reason := range reasons {
		kind := reasonKind[reason]
		if kind == "" {
			kind = "other"
		}
		if !seen[kind] {
			seen[kind] = true
			kinds = append(kinds, kind)
		}
	}
	if len(kinds) == 0 {
		return []string{"other"}
	}
	return kinds
}

func PrimaryKind(kinds []string) string {
	if len(kinds) == 0 {
		return "other"
	}
	best := kinds[0]
	for _, kind := range kinds[1:] {
		if rank(kind) < rank(best) {
			best = kind
		}
	}
	return best
}

func RepairAction(kind string) string {
	if action := kindAction[kind]; action != "" {
		return action
	}
	return kindAction["other"]
}

func FieldScaffold(missing []string) []FieldPrompt {
	out := make([]FieldPrompt, 0, len(missing))
	for _, field := range missing {
		question := fieldQuestion[field]
		if question == "" {
			question = fmt.Sprintf("What is the missing '%s'?", field)
		}
		out = append(out, FieldPrompt{Field: field, Question: question})
	}
	return out
}

func BuildRepairRow(issue issuecontract.IssueDraft, review issuecontract.Review, route RouteProposal, template issuecontract.TemplateRepairPlan, hasTemplate bool, minScore int) *RepairRow {
	if minScore <= 0 {
		minScore = MinScore
	}
	if review.OK && review.Score.Total >= minScore {
		return nil
	}
	reasons := append([]string(nil), review.Reasons...)
	kinds := RepairKinds(reasons)
	kind := PrimaryKind(kinds)
	row := &RepairRow{
		Number:     issue.Number,
		Title:      truncate(strings.TrimSpace(issue.Title), 120),
		Score:      review.Score.Total,
		Reasons:    reasons,
		Kinds:      kinds,
		Kind:       kind,
		NextAction: RepairAction(kind),
		Ready:      false,
	}
	if scaffoldsMissingFields(kind) {
		row.MissingFields = FieldScaffold(review.MissingFields)
	} else {
		row.MissingFields = []FieldPrompt{}
	}
	if contains(kinds, "route") && strings.TrimSpace(route.Lane) != "" {
		lane := strings.TrimSpace(route.Lane)
		row.ProposedLane = &lane
		if strings.TrimSpace(route.Confidence) != "" {
			conf := strings.TrimSpace(route.Confidence)
			row.RouteConfidence = &conf
		}
	}
	if contains(kinds, "template") && hasTemplate && strings.TrimSpace(template.ProposedNormalizedHeader) != "" {
		header := strings.TrimSpace(template.ProposedNormalizedHeader)
		row.Ready = true
		row.ProposedHeader = &header
	}
	return row
}

func BuildManifest(workspace string, issues []issuecontract.IssueDraft, opts Options) Manifest {
	if opts.MinScore <= 0 {
		opts.MinScore = MinScore
	}
	if opts.Review == nil {
		opts.Review = func(issue issuecontract.IssueDraft) issuecontract.Review {
			return issuecontract.ReviewIssueDraft(issue, issuecontract.Options{})
		}
	}
	if opts.Route == nil {
		opts.Route = DefaultRoute
	}
	if opts.Template == nil {
		opts.Template = issuecontract.BuildTemplateRepairPlan
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = len(issues)
	}
	var lanePtr *string
	if strings.TrimSpace(opts.Lane) != "" {
		lane := strings.TrimSpace(opts.Lane)
		lanePtr = &lane
	}
	var rows []RepairRow
	examined := 0
	for _, issue := range issues {
		review := opts.Review(issue)
		route := opts.Route(issue, review)
		if lanePtr != nil && review.Lane != *lanePtr && route.Lane != *lanePtr && route.BlockedLane != *lanePtr {
			continue
		}
		if examined >= limit {
			break
		}
		examined++
		template, hasTemplate := opts.Template(issue)
		row := BuildRepairRow(issue, review, route, template, hasTemplate, opts.MinScore)
		if row != nil {
			rows = append(rows, *row)
		}
	}
	byKind := map[string]int{}
	ready := 0
	for _, row := range rows {
		byKind[row.Kind]++
		if row.Ready {
			ready++
		}
	}
	return Manifest{
		Schema:    Schema,
		AsOf:      opts.AsOf,
		Workspace: workspace,
		Lane:      lanePtr,
		Limit:     limit,
		Counts: Counts{
			CandidatesExamined: examined,
			NeedsRepair:        len(rows),
			Ready:              ready,
			ByKind:             byKind,
		},
		Issues: rows,
	}
}

func BuildActions(manifest Manifest) []Action {
	out := make([]Action, 0, len(manifest.Issues))
	for _, row := range manifest.Issues {
		reason := strings.Join(row.Reasons, ", ")
		if reason == "" {
			reason = "issue contract below spawn floor"
		}
		out = append(out, Action{
			Number: row.Number, Kind: row.Kind, Ready: row.Ready,
			Reason: reason, NextAction: row.NextAction, Cmd: nil,
		})
	}
	return out
}

func RenderMarkdown(manifest Manifest) string {
	lane := "(all lanes)"
	if manifest.Lane != nil {
		lane = *manifest.Lane
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# Issue-contract repairs - %s\n\n", manifest.AsOf)
	fmt.Fprintf(&b, "**Lane:** %s  ·  **examined:** %d  ·  **needs repair:** %d  ·  **template-ready:** %d\n\n",
		lane, manifest.Counts.CandidatesExamined, manifest.Counts.NeedsRepair, manifest.Counts.Ready)
	b.WriteString("> Read-only pass. Never edits, labels, comments on, or closes an issue. `template`-kind rows carry a dry-run-computed header fix; every other kind lists the missing fields as questions for a human/agent to answer -- content is never invented here.\n\n")
	b.WriteString("## Counts by kind\n\n| kind | count |\n|---|---:|\n")
	kinds := make([]string, 0, len(manifest.Counts.ByKind))
	for kind := range manifest.Counts.ByKind {
		kinds = append(kinds, kind)
	}
	sort.Slice(kinds, func(i, j int) bool { return rank(kinds[i]) < rank(kinds[j]) })
	for _, kind := range kinds {
		fmt.Fprintf(&b, "| %s | %d |\n", kind, manifest.Counts.ByKind[kind])
	}
	b.WriteString("\n## Rows\n\n| # | kind | ready | score | title |\n|---|---|---|---:|---|\n")
	for _, row := range manifest.Issues {
		fmt.Fprintf(&b, "| #%d | %s | %t | %d | %s |\n", row.Number, row.Kind, row.Ready, row.Score, row.Title)
	}
	return b.String()
}

func FetchOpenIssues(workspace string, cap int) ([]issuecontract.IssueDraft, error) {
	if cap <= 0 {
		cap = DefaultCap
	}
	cmd := exec.Command("gh", "issue", "list", "--state", "open", "--limit", fmt.Sprint(cap), "--json", "number,title,body,labels,url")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = workspace
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("gh issue list failed (rc=%d): %s", ee.ExitCode(), strings.TrimSpace(string(ee.Stderr)))
		}
		return nil, err
	}
	var issues []issuecontract.IssueDraft
	if err := json.Unmarshal(out, &issues); err != nil {
		return nil, err
	}
	sort.Slice(issues, func(i, j int) bool { return issues[i].Number < issues[j].Number })
	return issues, nil
}

func DefaultRoute(_ issuecontract.IssueDraft, review issuecontract.Review) RouteProposal {
	if strings.TrimSpace(review.Lane) != "" {
		return RouteProposal{Lane: strings.TrimSpace(review.Lane), Confidence: "issue-contract"}
	}
	if len(review.Paths) > 0 {
		return RouteProposal{Confidence: "path-hints"}
	}
	return RouteProposal{}
}

func rank(kind string) int {
	if n, ok := kindRank[kind]; ok {
		return n
	}
	return 9
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func scaffoldsMissingFields(kind string) bool {
	switch kind {
	case "scope", "noise", "split", "private", "other":
		return true
	default:
		return false
	}
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}
