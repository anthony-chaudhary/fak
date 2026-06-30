package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

func cmdIssue(argv []string) { os.Exit(runIssue(os.Stdout, os.Stderr, argv)) }

func runIssue(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		issueUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "contract":
		return runIssueContract(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		issueUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak issue: unknown subcommand %q\n", argv[0])
		issueUsage(stderr)
		return 2
	}
}

type issueContractResult struct {
	Schema       string                     `json:"schema"`
	Mode         string                     `json:"mode"`
	File         string                     `json:"file"`
	OK           bool                       `json:"ok"`
	Counts       issueContractCounts        `json:"counts"`
	RepairQueues []issueContractRepairQueue `json:"repair_queues,omitempty"`
	BatchGroups  []issueContractBatchGroup  `json:"batch_groups,omitempty"`
	Reviews      []issuecontract.Review     `json:"reviews"`
}

type issueContractCounts struct {
	Total                int            `json:"total"`
	Dispatchable         int            `json:"dispatchable"`
	TriageOnly           int            `json:"triage_only"`
	Refused              int            `json:"refused"`
	StepBudget           int            `json:"step_budget"`
	MissingExpectedSteps int            `json:"missing_expected_steps"`
	AgentContextAvg      int            `json:"agent_context_avg"`
	AgentContextFull     int            `json:"agent_context_full"`
	AgentContextMissing  int            `json:"agent_context_missing"`
	ByReason             map[string]int `json:"by_reason"`
	ByLane               map[string]int `json:"by_lane"`
	ByWorkUnit           map[string]int `json:"by_work_unit"`
	ByExpectedStepBucket map[string]int `json:"by_expected_step_bucket"`
}

type issueContractBatchGroup struct {
	Key             string         `json:"key"`
	Lane            string         `json:"lane,omitempty"`
	WorkUnit        string         `json:"work_unit,omitempty"`
	Trigger         string         `json:"trigger,omitempty"`
	BatchPolicy     string         `json:"batch_policy,omitempty"`
	Count           int            `json:"count"`
	StepBudget      int            `json:"step_budget"`
	Dispatchable    int            `json:"dispatchable"`
	TriageOnly      int            `json:"triage_only"`
	Refused         int            `json:"refused"`
	ByReason        map[string]int `json:"by_reason,omitempty"`
	ExampleKeys     []string       `json:"example_keys,omitempty"`
	MissingMetadata []string       `json:"missing_metadata,omitempty"`
}

type issueContractRepairQueue struct {
	Kind             string         `json:"kind"`
	Count            int            `json:"count"`
	StepBudget       int            `json:"step_budget"`
	ChildIssueBudget int            `json:"child_issue_budget,omitempty"`
	NextAction       string         `json:"next_action"`
	ByReason         map[string]int `json:"by_reason,omitempty"`
	MissingFields    map[string]int `json:"missing_fields,omitempty"`
	ExampleKeys      []string       `json:"example_keys,omitempty"`
}

func runIssueContract(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue contract", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "issue candidate JSON file")
	fromPlan := fs.String("from-plan", "", "issue-plan JSON file containing one candidate, candidate, candidates, or items")
	fromIssues := fs.String("from-issues", "", "GitHub issue JSON file from gh issue list --json number,title,body,labels")
	live := fs.Bool("live", false, "review as an armed live/scheduled producer")
	dedupeChecked := fs.Bool("dedupe-checked", false, "producer proved marker dedupe against existing issues")
	dedupeCap := fs.Int("dedupe-cap", 0, "bounded issue scan cap proven before live sync")
	asJSON := fs.Bool("json", false, "emit machine-readable review/result")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	selected := 0
	for _, value := range []string{*file, *fromPlan, *fromIssues} {
		if value != "" {
			selected++
		}
	}
	if fs.NArg() != 0 || selected != 1 {
		fmt.Fprintln(stderr, "fak issue contract: pass exactly one of --file CANDIDATE.json, --from-plan PLAN.json, or --from-issues ISSUES.json")
		return 2
	}

	pathArg := *file
	mode := "candidate"
	if *fromPlan != "" {
		pathArg = *fromPlan
		mode = "plan"
	}
	if *fromIssues != "" {
		pathArg = *fromIssues
		mode = "issues"
	}
	path, err := filepath.Abs(pathArg)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue contract: %v\n", err)
		return 2
	}
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue contract: %v\n", err)
		return 2
	}
	result := issueContractResult{
		Schema: "fak.issue-contract-result.v1",
		Mode:   mode,
		File:   path,
		OK:     true,
	}
	opts := issuecontract.Options{
		Live:          *live,
		DedupeChecked: *dedupeChecked,
		DedupeCap:     *dedupeCap,
	}
	if mode == "issues" {
		issues, err := decodeIssueContractIssues(b)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue contract: %v\n", err)
			return 2
		}
		result.Reviews = make([]issuecontract.Review, 0, len(issues))
		for _, issue := range issues {
			review := issuecontract.ReviewIssueDraft(issue, opts)
			if !review.OK {
				result.OK = false
			}
			result.Reviews = append(result.Reviews, review)
		}
	} else {
		candidates, err := decodeIssueContractCandidates(b)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue contract: %v\n", err)
			return 2
		}
		result.Reviews = make([]issuecontract.Review, 0, len(candidates))
		for _, c := range candidates {
			review := issuecontract.ReviewCandidate(c, opts)
			if !review.OK {
				result.OK = false
			}
			result.Reviews = append(result.Reviews, review)
		}
	}
	result.Counts, result.BatchGroups = summarizeIssueContractReviews(result.Reviews)
	result.RepairQueues = issueContractRepairQueues(result.Reviews)

	if *asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak issue contract: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, renderIssueContract(result))
	}
	if !result.OK {
		return 3
	}
	return 0
}

func summarizeIssueContractReviews(reviews []issuecontract.Review) (issueContractCounts, []issueContractBatchGroup) {
	counts := issueContractCounts{
		Total:                len(reviews),
		ByReason:             map[string]int{},
		ByLane:               map[string]int{},
		ByWorkUnit:           map[string]int{},
		ByExpectedStepBucket: map[string]int{},
	}
	batches := map[string]*issueContractBatchGroup{}
	agentContextSum := 0
	for _, review := range reviews {
		switch review.Dispatchability {
		case issuecontract.Dispatchable:
			counts.Dispatchable++
		case issuecontract.TriageOnly:
			counts.TriageOnly++
		case issuecontract.Refused:
			counts.Refused++
		}
		if review.AgentContext.Total >= 100 {
			counts.AgentContextFull++
		} else {
			counts.AgentContextMissing++
		}
		stepBudget := issueContractReviewStepBudget(review)
		counts.StepBudget += stepBudget
		if review.ExpectedSteps <= 0 {
			counts.MissingExpectedSteps++
		}
		counts.ByLane[issueContractBucketValue(review.Lane, "(unrouted)")]++
		counts.ByWorkUnit[issueContractBucketValue(review.WorkUnit, "(missing)")]++
		counts.ByExpectedStepBucket[issueContractStepBucket(review.ExpectedSteps)]++
		agentContextSum += review.AgentContext.Total
		for _, reason := range review.Reasons {
			counts.ByReason[reason]++
		}
		key := issueContractBatchKey(review)
		group := batches[key]
		if group == nil {
			group = &issueContractBatchGroup{
				Key:         key,
				Lane:        strings.TrimSpace(review.Lane),
				WorkUnit:    strings.TrimSpace(review.WorkUnit),
				Trigger:     strings.TrimSpace(review.Trigger),
				BatchPolicy: strings.TrimSpace(review.BatchPolicy),
				ByReason:    map[string]int{},
			}
			group.MissingMetadata = issueContractBatchMissingMetadata(review)
			batches[key] = group
		}
		group.Count++
		group.StepBudget += stepBudget
		switch review.Dispatchability {
		case issuecontract.Dispatchable:
			group.Dispatchable++
		case issuecontract.TriageOnly:
			group.TriageOnly++
		case issuecontract.Refused:
			group.Refused++
		}
		for _, reason := range review.Reasons {
			group.ByReason[reason]++
		}
		if review.Key != "" && len(group.ExampleKeys) < 5 {
			group.ExampleKeys = append(group.ExampleKeys, review.Key)
		}
	}
	if len(reviews) > 0 {
		counts.AgentContextAvg = (agentContextSum + len(reviews)/2) / len(reviews)
	}
	groups := make([]issueContractBatchGroup, 0, len(batches))
	for _, group := range batches {
		if len(group.ByReason) == 0 {
			group.ByReason = nil
		}
		groups = append(groups, *group)
	}
	sort.Slice(groups, func(i, j int) bool {
		if groups[i].Count != groups[j].Count {
			return groups[i].Count > groups[j].Count
		}
		if groups[i].StepBudget != groups[j].StepBudget {
			return groups[i].StepBudget > groups[j].StepBudget
		}
		return groups[i].Key < groups[j].Key
	})
	return counts, groups
}

func issueContractReviewStepBudget(review issuecontract.Review) int {
	if review.ExpectedSteps > 0 {
		return review.ExpectedSteps
	}
	return 1
}

func issueContractStepBucket(steps int) string {
	switch {
	case steps <= 0:
		return "(missing)"
	case steps == 1:
		return "1"
	case steps <= 3:
		return "2-3"
	case steps <= issuecontract.MaxDispatchExpectedSteps:
		return "4-8"
	default:
		return "over-8"
	}
}

func issueContractBucketValue(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}

func issueContractBatchKey(review issuecontract.Review) string {
	lane := issueContractBucketValue(review.Lane, "unrouted")
	workUnit := issueContractBucketValue(review.WorkUnit, "missing-work-unit")
	trigger := issueContractBucketValue(review.Trigger, "missing-trigger")
	batchPolicy := issueContractBucketValue(review.BatchPolicy, "missing-batch-policy")
	return lane + "|" + workUnit + "|" + trigger + "|" + batchPolicy
}

func issueContractBatchMissingMetadata(review issuecontract.Review) []string {
	var missing []string
	if strings.TrimSpace(review.Lane) == "" {
		missing = append(missing, "lane")
	}
	if strings.TrimSpace(review.WorkUnit) == "" {
		missing = append(missing, "work_unit")
	}
	if review.ExpectedSteps <= 0 {
		missing = append(missing, "expected_steps")
	}
	if strings.TrimSpace(review.Trigger) == "" {
		missing = append(missing, "trigger")
	}
	if strings.TrimSpace(review.BatchPolicy) == "" {
		missing = append(missing, "batch_policy")
	}
	return missing
}

func issueContractRepairQueues(reviews []issuecontract.Review) []issueContractRepairQueue {
	queues := map[string]*issueContractRepairQueue{}
	for _, review := range reviews {
		kinds := issueContractRepairKinds(review)
		for _, kind := range kinds {
			queue := queues[kind]
			if queue == nil {
				queue = &issueContractRepairQueue{
					Kind:          kind,
					NextAction:    issueContractRepairAction(kind),
					ByReason:      map[string]int{},
					MissingFields: map[string]int{},
				}
				queues[kind] = queue
			}
			queue.Count++
			queue.StepBudget += issueContractReviewStepBudget(review)
			queue.ChildIssueBudget += issueContractReviewChildIssueBudget(review, kind)
			for _, reason := range review.Reasons {
				queue.ByReason[reason]++
			}
			for _, missing := range review.MissingFields {
				queue.MissingFields[missing]++
			}
			if review.Key != "" && len(queue.ExampleKeys) < 8 {
				queue.ExampleKeys = append(queue.ExampleKeys, review.Key)
			}
		}
	}
	out := make([]issueContractRepairQueue, 0, len(queues))
	for _, queue := range queues {
		if len(queue.ByReason) == 0 {
			queue.ByReason = nil
		}
		if len(queue.MissingFields) == 0 {
			queue.MissingFields = nil
		}
		out = append(out, *queue)
	}
	sort.Slice(out, func(i, j int) bool {
		ri, rj := issueContractRepairRank(out[i].Kind), issueContractRepairRank(out[j].Kind)
		if ri != rj {
			return ri < rj
		}
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Kind < out[j].Kind
	})
	return out
}

func issueContractReviewChildIssueBudget(review issuecontract.Review, kind string) int {
	if kind != "split" {
		return 0
	}
	if review.ExpectedSteps <= 0 {
		return 1
	}
	return (review.ExpectedSteps + issuecontract.MaxDispatchExpectedSteps - 1) / issuecontract.MaxDispatchExpectedSteps
}

func issueContractRepairKinds(review issuecontract.Review) []string {
	if review.OK && review.Dispatchability == issuecontract.Dispatchable {
		return []string{"dispatch"}
	}
	var kinds []string
	add := func(kind string) {
		for _, existing := range kinds {
			if existing == kind {
				return
			}
		}
		kinds = append(kinds, kind)
	}
	for _, reason := range review.Reasons {
		switch reason {
		case issuecontract.ReasonNotDispatchLeaf, issuecontract.ReasonOversizedSteps:
			add("split")
		case issuecontract.ReasonScopeIncomplete:
			add("scope")
		case issuecontract.ReasonUnrouted:
			add("route")
		case issuecontract.ReasonLiveUnarmored, issuecontract.ReasonNoiseIncomplete, issuecontract.ReasonAgentIncomplete:
			add("noise")
		case issuecontract.ReasonPrivateBoundary:
			add("private")
		default:
			add("other")
		}
	}
	if len(kinds) == 0 {
		kinds = append(kinds, "other")
	}
	return kinds
}

func issueContractRepairRank(kind string) int {
	switch kind {
	case "dispatch":
		return 0
	case "split":
		return 1
	case "scope":
		return 2
	case "route":
		return 3
	case "noise":
		return 4
	case "private":
		return 5
	default:
		return 9
	}
}

func issueContractRepairAction(kind string) string {
	switch kind {
	case "dispatch":
		return "send these scoped leaves to dispatch lanes, oldest/highest-priority first"
	case "split":
		return fmt.Sprintf("decompose each non-leaf or oversized row into child issues with <= %d expected steps", issuecontract.MaxDispatchExpectedSteps)
	case "scope":
		return "add the missing parent/current-state/scope/done/witness/closure fields before dispatch"
	case "route":
		return "add a lane or path hints section so the issue maps to one dispatch lane"
	case "noise":
		return "add trigger, batch policy, agent context, and live dedupe/cap evidence before automated sync"
	case "private":
		return "remove private/operator-only evidence or move the work to the private companion repo"
	default:
		return "inspect the review reasons and repair the row before dispatch"
	}
}

func decodeIssueContractCandidates(b []byte) ([]issuecontract.Candidate, error) {
	var arr []issuecontract.Candidate
	if err := json.Unmarshal(b, &arr); err == nil {
		if len(arr) == 0 {
			return nil, fmt.Errorf("candidate list is empty")
		}
		return arr, nil
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil {
		return nil, fmt.Errorf("parse candidate JSON: %w", err)
	}
	for _, key := range []string{"candidates", "items"} {
		if raw, ok := obj[key]; ok {
			if err := json.Unmarshal(raw, &arr); err != nil {
				return nil, fmt.Errorf("%s must be an issue-candidate array: %w", key, err)
			}
			if len(arr) == 0 {
				return nil, fmt.Errorf("%s is empty", key)
			}
			return arr, nil
		}
	}
	if raw, ok := obj["candidate"]; ok {
		var c issuecontract.Candidate
		if err := json.Unmarshal(raw, &c); err != nil {
			return nil, fmt.Errorf("candidate must be an issue-candidate object: %w", err)
		}
		return []issuecontract.Candidate{c}, nil
	}
	var c issuecontract.Candidate
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("candidate must be an issue-candidate object: %w", err)
	}
	return []issuecontract.Candidate{c}, nil
}

func decodeIssueContractIssues(b []byte) ([]issuecontract.IssueDraft, error) {
	var arr []issuecontract.IssueDraft
	if err := json.Unmarshal(b, &arr); err == nil {
		if len(arr) == 0 {
			return nil, fmt.Errorf("issue list is empty")
		}
		return arr, nil
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil {
		return nil, fmt.Errorf("parse issue JSON: %w", err)
	}
	for _, key := range []string{"issues", "items"} {
		if raw, ok := obj[key]; ok {
			if err := json.Unmarshal(raw, &arr); err != nil {
				return nil, fmt.Errorf("%s must be a GitHub issue array: %w", key, err)
			}
			if len(arr) == 0 {
				return nil, fmt.Errorf("%s is empty", key)
			}
			return arr, nil
		}
	}
	var issue issuecontract.IssueDraft
	if err := json.Unmarshal(b, &issue); err != nil {
		return nil, fmt.Errorf("issue must be a GitHub issue object: %w", err)
	}
	return []issuecontract.IssueDraft{issue}, nil
}

func renderIssueContract(r issueContractResult) string {
	lines := []string{
		fmt.Sprintf("issue-contract: %s  ok=%t  candidate_count=%d", r.Mode, r.OK, len(r.Reviews)),
		fmt.Sprintf("  file: %s", r.File),
		fmt.Sprintf("  counts: dispatchable=%d triage_only=%d refused=%d steps=%d missing_steps=%d agent_context_avg=%d full=%d missing=%d",
			r.Counts.Dispatchable, r.Counts.TriageOnly, r.Counts.Refused,
			r.Counts.StepBudget, r.Counts.MissingExpectedSteps,
			r.Counts.AgentContextAvg, r.Counts.AgentContextFull, r.Counts.AgentContextMissing),
	}
	if len(r.Counts.ByReason) > 0 {
		lines = append(lines, "  reasons: "+renderIssueContractReasonCounts(r.Counts.ByReason))
	}
	if len(r.Counts.ByLane) > 0 {
		lines = append(lines, "  lanes: "+renderIssueContractReasonCounts(r.Counts.ByLane))
	}
	if len(r.Counts.ByWorkUnit) > 0 {
		lines = append(lines, "  work_units: "+renderIssueContractReasonCounts(r.Counts.ByWorkUnit))
	}
	if len(r.Counts.ByExpectedStepBucket) > 0 {
		lines = append(lines, "  step_buckets: "+renderIssueContractReasonCounts(r.Counts.ByExpectedStepBucket))
	}
	for i, group := range r.BatchGroups {
		if i >= 8 {
			lines = append(lines, fmt.Sprintf("  batch_groups: ... %d more", len(r.BatchGroups)-i))
			break
		}
		lines = append(lines, fmt.Sprintf("  batch_group[%d]: count=%d steps=%d lane=%s work_unit=%s key=%s",
			i, group.Count, group.StepBudget,
			issueContractBucketValue(group.Lane, "(unrouted)"),
			issueContractBucketValue(group.WorkUnit, "(missing)"),
			group.Key))
		if len(group.MissingMetadata) > 0 {
			lines = append(lines, "    missing_batch_metadata: "+strings.Join(group.MissingMetadata, ", "))
		}
	}
	for _, queue := range r.RepairQueues {
		line := fmt.Sprintf("  repair_queue[%s]: count=%d steps=%d",
			queue.Kind, queue.Count, queue.StepBudget)
		if queue.ChildIssueBudget > 0 {
			line += fmt.Sprintf(" child_issues=%d", queue.ChildIssueBudget)
		}
		line += fmt.Sprintf(" next=%s", queue.NextAction)
		lines = append(lines, line)
		if len(queue.ByReason) > 0 {
			lines = append(lines, "    reasons: "+renderIssueContractReasonCounts(queue.ByReason))
		}
		if len(queue.MissingFields) > 0 {
			lines = append(lines, "    missing_fields: "+renderIssueContractReasonCounts(queue.MissingFields))
		}
	}
	for _, review := range r.Reviews {
		key := review.Key
		if strings.TrimSpace(key) == "" {
			key = "(missing-key)"
		}
		lines = append(lines, fmt.Sprintf("  [%s] %s dispatchability=%s score=%d spine_priority=%d",
			review.Verdict, key, review.Dispatchability, review.Score.Total, review.SpinePriority.Total))
		for _, reason := range review.Reasons {
			lines = append(lines, "    refuses: "+reason)
		}
		for _, missing := range review.MissingFields {
			lines = append(lines, "    missing: "+missing)
		}
	}
	return strings.Join(lines, "\n")
}

func renderIssueContractReasonCounts(counts map[string]int) string {
	reasons := make([]string, 0, len(counts))
	for reason := range counts {
		reasons = append(reasons, reason)
	}
	sort.Strings(reasons)
	parts := make([]string, 0, len(reasons))
	for _, reason := range reasons {
		parts = append(parts, fmt.Sprintf("%s=%d", reason, counts[reason]))
	}
	return strings.Join(parts, ", ")
}

func issueUsage(w io.Writer) {
	fmt.Fprint(w, `fak issue - generated-issue gates

  fak issue contract --file CANDIDATE.json [--json]
  fak issue contract --from-plan PLAN.json [--json]
  fak issue contract --from-issues ISSUES.json [--json]
                     [--live --dedupe-checked --dedupe-cap N]

The contract command reviews machine-created GitHub issue candidates before a
producer syncs them. Exit 0 means dispatchable; exit 3 means the candidate is
triage-only or refused with closed reasons such as ISSUE_SCOPE_INCOMPLETE,
ISSUE_UNROUTED, ISSUE_NOT_DISPATCH_LEAF, ISSUE_OVERSIZED_EXPECTED_STEPS,
ISSUE_NOISE_CONTROL_INCOMPLETE, ISSUE_AGENT_CONTEXT_INCOMPLETE,
ISSUE_PRIVATE_BOUNDARY, or ISSUE_LIVE_UNARMORED.
`)
}
