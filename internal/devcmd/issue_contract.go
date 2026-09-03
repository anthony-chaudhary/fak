package devcmd

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func RunIssue(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		issueUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "audit":
		return runIssueAudit(stdout, stderr, argv[1:])
	case "audit-loop":
		return runIssueAuditLoop(stdout, stderr, argv[1:])
	case "contract":
		return runIssueContract(stdout, stderr, argv[1:])
	case "reconcile":
		return runIssueReconcile(stdout, stderr, argv[1:])
	case "cohort":
		return runIssueCohort(stdout, stderr, argv[1:])
	case "fanout":
		return runIssueFanout(stdout, stderr, argv[1:])
	case "create":
		return runIssueCreate(stdout, stderr, argv[1:])
	case "edit":
		return runIssueEdit(stdout, stderr, argv[1:])
	case "repair":
		return runIssueRepair(stdout, stderr, argv[1:])
	case "decompose":
		return runIssueDecompose(stdout, stderr, argv[1:])
	case "dedup":
		return runIssueDedup(stdout, stderr, argv[1:])
	case "finding":
		return runIssueFinding(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		issueUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak-dev issue: unknown subcommand %q\n", argv[0])
		issueUsage(stderr)
		return 2
	}
}

type issueContractResult struct {
	Schema              string                           `json:"schema"`
	Mode                string                           `json:"mode"`
	File                string                           `json:"file"`
	OK                  bool                             `json:"ok"`
	Counts              issueContractCounts              `json:"counts"`
	RepairQueues        []issueContractRepairQueue       `json:"repair_queues,omitempty"`
	BatchGroups         []issueContractBatchGroup        `json:"batch_groups,omitempty"`
	DuplicateKeyGroups  []issueContractDuplicateGroup    `json:"duplicate_key_groups,omitempty"`
	AssumptionGroups    []issueContractAgentNoteGroup    `json:"assumption_groups,omitempty"`
	ConfusionGroups     []issueContractAgentNoteGroup    `json:"confusion_groups,omitempty"`
	CoordinationGroups  []issueContractAgentNoteGroup    `json:"coordination_groups,omitempty"`
	TemplateRepairPlans []issuepolicy.TemplateRepairPlan `json:"template_repair_plans,omitempty"`
	Reviews             []issuepolicy.Review             `json:"reviews"`
	ReadinessSchema     string                           `json:"readiness_schema"`
	ProblemFrameSchema  string                           `json:"problem_frame_schema"`
}

type issueContractCounts struct {
	Total                 int            `json:"total"`
	Dispatchable          int            `json:"dispatchable"`
	TriageOnly            int            `json:"triage_only"`
	Refused               int            `json:"refused"`
	StepBudget            int            `json:"step_budget"`
	MissingExpectedSteps  int            `json:"missing_expected_steps"`
	AgentContextAvg       int            `json:"agent_context_avg"`
	AgentContextFull      int            `json:"agent_context_full"`
	AgentContextMissing   int            `json:"agent_context_missing"`
	GenerationFitAvg      int            `json:"generation_fit_avg,omitempty"`
	GenerationFitMeasured int            `json:"generation_fit_measured,omitempty"`
	GenerationMismatches  int            `json:"generation_mismatches,omitempty"`
	ModelTierTagged       int            `json:"model_tier_tagged,omitempty"`
	ModelTierFlagged      int            `json:"model_tier_flagged,omitempty"`
	BornRouted            int            `json:"born_routed,omitempty"`
	BornRoutedFlagged     int            `json:"born_routed_flagged,omitempty"`
	ScaleFlagged          int            `json:"scale_flagged,omitempty"`
	ByReason              map[string]int `json:"by_reason"`
	ByLane                map[string]int `json:"by_lane"`
	ByWorkUnit            map[string]int `json:"by_work_unit"`
	ByExpectedStepBucket  map[string]int `json:"by_expected_step_bucket"`
	ByGeneration          map[string]int `json:"by_generation,omitempty"`
	ByRequiredTier        map[string]int `json:"by_required_tier,omitempty"`
	ByScale               map[string]int `json:"by_scale,omitempty"`
}

type issueContractBatchGroup struct {
	Key              string         `json:"key"`
	Lane             string         `json:"lane,omitempty"`
	WorkUnit         string         `json:"work_unit,omitempty"`
	Trigger          string         `json:"trigger,omitempty"`
	BatchPolicy      string         `json:"batch_policy,omitempty"`
	Count            int            `json:"count"`
	StepBudget       int            `json:"step_budget"`
	ChildIssueBudget int            `json:"child_issue_budget,omitempty"`
	DeclaredCap      int            `json:"declared_cap,omitempty"`
	OverCap          int            `json:"over_cap,omitempty"`
	Dispatchable     int            `json:"dispatchable"`
	TriageOnly       int            `json:"triage_only"`
	Refused          int            `json:"refused"`
	ByReason         map[string]int `json:"by_reason,omitempty"`
	ExampleKeys      []string       `json:"example_keys,omitempty"`
	MissingMetadata  []string       `json:"missing_metadata,omitempty"`
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

type issueContractAgentNoteGroup struct {
	Key              string         `json:"key"`
	Count            int            `json:"count"`
	StepBudget       int            `json:"step_budget"`
	ChildIssueBudget int            `json:"child_issue_budget,omitempty"`
	Dispatchable     int            `json:"dispatchable"`
	TriageOnly       int            `json:"triage_only"`
	Refused          int            `json:"refused"`
	ByLane           map[string]int `json:"by_lane,omitempty"`
	ByWorkUnit       map[string]int `json:"by_work_unit,omitempty"`
	ByReason         map[string]int `json:"by_reason,omitempty"`
	ExampleKeys      []string       `json:"example_keys,omitempty"`
}

type issueContractDuplicateGroup struct {
	Key              string         `json:"key"`
	Count            int            `json:"count"`
	StepBudget       int            `json:"step_budget"`
	ChildIssueBudget int            `json:"child_issue_budget,omitempty"`
	Issues           []int          `json:"issues,omitempty"`
	Dispatchable     int            `json:"dispatchable"`
	TriageOnly       int            `json:"triage_only"`
	Refused          int            `json:"refused"`
	ByLane           map[string]int `json:"by_lane,omitempty"`
	ByReason         map[string]int `json:"by_reason,omitempty"`
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
	strictModelTier := fs.Bool("strict-model-tier", false, "hold issues with missing/invalid/contradictory model-tier metadata triage-only")
	strictScale := fs.Bool("strict-scale", false, "hold issues with an undeclared work size or a witness smaller than the work triage-only")
	strictWitness := fs.Bool("strict-witness", false, "hold issues whose done-condition witness is missing, weak, or forgeable triage-only")
	strictProjectWork := fs.Bool("strict-project-work", false, "hold issues missing/invalid effort, parent contribution, or completion-standard metadata triage-only")
	strictBornRouted := fs.Bool("strict-born-routed", false, "hold issues missing a lane, class label, or priority label triage-only")
	asJSON := fs.Bool("json", false, "emit machine-readable review/result")
	if !parseFlags(fs, argv) {
		return 2
	}
	selected := 0
	for _, value := range []string{*file, *fromPlan, *fromIssues} {
		if value != "" {
			selected++
		}
	}
	if fs.NArg() != 0 || selected != 1 {
		fmt.Fprintln(stderr, "fak-dev issue contract: pass exactly one of --file CANDIDATE.json, --from-plan PLAN.json, or --from-issues ISSUES.json")
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
		fmt.Fprintf(stderr, "fak-dev issue contract: %v\n", err)
		return 2
	}
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue contract: %v\n", err)
		return 2
	}
	result := issueContractResult{
		Schema:             "fak.issue-contract-result.v1",
		Mode:               mode,
		File:               path,
		ReadinessSchema:    issuepolicy.TaskBriefSchema,
		ProblemFrameSchema: issuepolicy.ProblemFrameSchema,
		OK:                 true,
	}
	opts := issuepolicy.Options{
		Live:              *live,
		DedupeChecked:     *dedupeChecked,
		DedupeCap:         *dedupeCap,
		StrictModelTier:   *strictModelTier,
		StrictScale:       *strictScale,
		StrictWitness:     *strictWitness,
		StrictBornRouted:  *strictBornRouted,
		StrictProjectWork: *strictProjectWork,
	}
	if mode == "issues" {
		issues, err := decodeIssueContractIssues(b)
		if err != nil {
			return issueContractDecodeRefusal(stderr, err)
		}
		result.Reviews = make([]issuepolicy.Review, 0, len(issues))
		for _, issue := range issues {
			review := issuepolicy.ReviewIssueDraft(issue, opts)
			if !review.OK {
				result.OK = false
			}
			result.Reviews = append(result.Reviews, review)
			if plan, ok := issuepolicy.BuildTemplateRepairPlan(issue); ok {
				result.TemplateRepairPlans = append(result.TemplateRepairPlans, plan)
			}
		}
	} else {
		candidates, err := decodeIssueContractCandidates(b)
		if err != nil {
			return issueContractDecodeRefusal(stderr, err)
		}
		result.Reviews = make([]issuepolicy.Review, 0, len(candidates))
		for _, c := range candidates {
			review := issuepolicy.ReviewCandidate(c, opts)
			if !review.OK {
				result.OK = false
			}
			result.Reviews = append(result.Reviews, review)
		}
	}
	result.Counts, result.BatchGroups, result.DuplicateKeyGroups, result.AssumptionGroups, result.ConfusionGroups, result.CoordinationGroups = summarizeIssueContractReviews(result.Reviews)
	result.RepairQueues = issueContractRepairQueues(result.Reviews)

	if *asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak-dev issue contract: encode json: %v\n", err)
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

// issueContractDecodeRefusal reports input this command cannot even parse. Both accepted
// input shapes -- issue drafts and dedupe candidates -- refuse with the same wording and the
// same exit 2 (usage, not a failed review), so an operator who fed the wrong file gets one
// answer regardless of which mode they asked for.
func issueContractDecodeRefusal(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "fak-dev issue contract: %v\n", err)
	return 2
}

func summarizeIssueContractReviews(reviews []issuepolicy.Review) (issueContractCounts, []issueContractBatchGroup, []issueContractDuplicateGroup, []issueContractAgentNoteGroup, []issueContractAgentNoteGroup, []issueContractAgentNoteGroup) {
	counts := issueContractCounts{
		Total:                len(reviews),
		ByReason:             map[string]int{},
		ByLane:               map[string]int{},
		ByWorkUnit:           map[string]int{},
		ByExpectedStepBucket: map[string]int{},
		ByGeneration:         map[string]int{},
		ByRequiredTier:       map[string]int{},
		ByScale:              map[string]int{},
	}
	batches := map[string]*issueContractBatchGroup{}
	duplicateGroups := map[string]*issueContractDuplicateGroup{}
	assumptionGroups := map[string]*issueContractAgentNoteGroup{}
	confusionGroups := map[string]*issueContractAgentNoteGroup{}
	coordinationGroups := map[string]*issueContractAgentNoteGroup{}
	agentContextSum := 0
	generationFitSum := 0
	for _, review := range reviews {
		switch review.Dispatchability {
		case issuepolicy.Dispatchable:
			counts.Dispatchable++
		case issuepolicy.TriageOnly:
			counts.TriageOnly++
		case issuepolicy.Refused:
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
		if issueContractReviewHasGenerationFit(review) {
			counts.GenerationFitMeasured++
			generationFitSum += review.GenerationFit.Total
			if len(review.GenerationFit.Flags) > 0 {
				counts.GenerationMismatches++
			}
			counts.ByGeneration[issueContractBucketValue(review.GenerationFit.Stream, "(unclassified)")]++
		}
		if issueContractReviewHasModelTier(review) {
			counts.ModelTierTagged++
			if review.ModelTier.Required != "" {
				counts.ByRequiredTier[review.ModelTier.Required]++
			}
		}
		if len(review.ModelTier.Flags) > 0 {
			counts.ModelTierFlagged++
		}
		if review.BornRouted.Lane != "" && review.BornRouted.ClassLabel != "" && review.BornRouted.PriorityLabel != "" {
			counts.BornRouted++
		}
		if len(review.BornRouted.Flags) > 0 {
			counts.BornRoutedFlagged++
		}
		counts.ByScale[issueContractBucketValue(string(review.Scale.Effective), "(undeclared)")]++
		if len(review.Scale.Flags) > 0 {
			counts.ScaleFlagged++
		}
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
				DeclaredCap: issueContractBatchPolicyCap(review.BatchPolicy),
				ByReason:    map[string]int{},
			}
			group.MissingMetadata = issueContractBatchMissingMetadata(review)
			batches[key] = group
		}
		group.Count++
		group.StepBudget += stepBudget
		group.ChildIssueBudget += issueContractReviewSplitChildIssueBudget(review)
		if group.DeclaredCap > 0 && group.Count > group.DeclaredCap {
			group.OverCap = group.Count - group.DeclaredCap
		}
		switch review.Dispatchability {
		case issuepolicy.Dispatchable:
			group.Dispatchable++
		case issuepolicy.TriageOnly:
			group.TriageOnly++
		case issuepolicy.Refused:
			group.Refused++
		}
		for _, reason := range review.Reasons {
			group.ByReason[reason]++
		}
		if review.Key != "" && len(group.ExampleKeys) < 5 {
			group.ExampleKeys = append(group.ExampleKeys, review.Key)
		}
		issueContractAddDuplicateGroup(duplicateGroups, review, stepBudget)
		issueContractAddAgentNoteGroups(assumptionGroups, review.Assumptions, review, stepBudget)
		issueContractAddAgentNoteGroups(confusionGroups, review.ConfusionRisks, review, stepBudget)
		issueContractAddAgentNoteGroups(coordinationGroups, review.Coordination, review, stepBudget)
	}
	if len(reviews) > 0 {
		counts.AgentContextAvg = (agentContextSum + len(reviews)/2) / len(reviews)
	}
	if counts.GenerationFitMeasured > 0 {
		counts.GenerationFitAvg = (generationFitSum + counts.GenerationFitMeasured/2) / counts.GenerationFitMeasured
	} else {
		counts.ByGeneration = nil
	}
	if counts.ModelTierTagged == 0 {
		counts.ByRequiredTier = nil
	}
	groups := make([]issueContractBatchGroup, 0, len(batches))
	for _, group := range batches {
		if len(group.ByReason) == 0 {
			group.ByReason = nil
		}
		groups = append(groups, *group)
	}
	sort.Slice(groups, func(i, j int) bool {
		if (groups[i].OverCap > 0) != (groups[j].OverCap > 0) {
			return groups[i].OverCap > 0
		}
		if groups[i].OverCap != groups[j].OverCap {
			return groups[i].OverCap > groups[j].OverCap
		}
		if groups[i].Count != groups[j].Count {
			return groups[i].Count > groups[j].Count
		}
		if groups[i].StepBudget != groups[j].StepBudget {
			return groups[i].StepBudget > groups[j].StepBudget
		}
		if groups[i].ChildIssueBudget != groups[j].ChildIssueBudget {
			return groups[i].ChildIssueBudget > groups[j].ChildIssueBudget
		}
		return groups[i].Key < groups[j].Key
	})
	duplicates := issueContractSortedDuplicateGroups(duplicateGroups)
	assumptions := issueContractSortedAgentNoteGroups(assumptionGroups)
	confusions := issueContractSortedAgentNoteGroups(confusionGroups)
	coordination := issueContractSortedAgentNoteGroups(coordinationGroups)
	return counts, groups, duplicates, assumptions, confusions, coordination
}

func issueContractReviewStepBudget(review issuepolicy.Review) int {
	if review.ExpectedSteps > 0 {
		return review.ExpectedSteps
	}
	return 1
}

func issueContractReviewHasGenerationFit(review issuepolicy.Review) bool {
	return strings.TrimSpace(review.GenerationFit.Stream) != "" ||
		strings.TrimSpace(review.GenerationFit.LabelStream) != "" ||
		strings.TrimSpace(review.GenerationFit.BodyStream) != "" ||
		len(review.GenerationFit.Flags) > 0
}

// issueContractReviewHasModelTier reports whether the review resolved at least
// one model tier (required or optimal). A review that only carries missing-tier
// flags is NOT counted as tagged — the flag count is the separate signal.
func issueContractReviewHasModelTier(review issuepolicy.Review) bool {
	return strings.TrimSpace(review.ModelTier.Required) != "" ||
		strings.TrimSpace(review.ModelTier.Optimal) != ""
}

func issueContractAddDuplicateGroup(groups map[string]*issueContractDuplicateGroup, review issuepolicy.Review, stepBudget int) {
	key := strings.TrimSpace(review.Key)
	if key == "" {
		return
	}
	group := groups[key]
	if group == nil {
		group = &issueContractDuplicateGroup{
			Key:      key,
			ByLane:   map[string]int{},
			ByReason: map[string]int{},
		}
		groups[key] = group
	}
	group.Count++
	group.StepBudget += stepBudget
	group.ChildIssueBudget += issueContractReviewSplitChildIssueBudget(review)
	switch review.Dispatchability {
	case issuepolicy.Dispatchable:
		group.Dispatchable++
	case issuepolicy.TriageOnly:
		group.TriageOnly++
	case issuepolicy.Refused:
		group.Refused++
	}
	group.ByLane[issueContractBucketValue(review.Lane, "(unrouted)")]++
	for _, reason := range review.Reasons {
		group.ByReason[reason]++
	}
	if review.IssueNumber > 0 && len(group.Issues) < 12 {
		group.Issues = append(group.Issues, review.IssueNumber)
	}
}

func issueContractSortedDuplicateGroups(groups map[string]*issueContractDuplicateGroup) []issueContractDuplicateGroup {
	out := make([]issueContractDuplicateGroup, 0, len(groups))
	for _, group := range groups {
		if group.Count < 2 {
			continue
		}
		if len(group.ByLane) == 0 {
			group.ByLane = nil
		}
		if len(group.ByReason) == 0 {
			group.ByReason = nil
		}
		sort.Ints(group.Issues)
		out = append(out, *group)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		if out[i].StepBudget != out[j].StepBudget {
			return out[i].StepBudget > out[j].StepBudget
		}
		return out[i].Key < out[j].Key
	})
	return out
}

func issueContractAddAgentNoteGroups(groups map[string]*issueContractAgentNoteGroup, notes []string, review issuepolicy.Review, stepBudget int) {
	for _, key := range issueContractAgentNoteKeys(notes) {
		group := groups[key]
		if group == nil {
			group = &issueContractAgentNoteGroup{
				Key:        key,
				ByLane:     map[string]int{},
				ByWorkUnit: map[string]int{},
				ByReason:   map[string]int{},
			}
			groups[key] = group
		}
		group.Count++
		group.StepBudget += stepBudget
		group.ChildIssueBudget += issueContractReviewSplitChildIssueBudget(review)
		switch review.Dispatchability {
		case issuepolicy.Dispatchable:
			group.Dispatchable++
		case issuepolicy.TriageOnly:
			group.TriageOnly++
		case issuepolicy.Refused:
			group.Refused++
		}
		group.ByLane[issueContractBucketValue(review.Lane, "(unrouted)")]++
		group.ByWorkUnit[issueContractBucketValue(review.WorkUnit, "(missing)")]++
		for _, reason := range review.Reasons {
			group.ByReason[reason]++
		}
		if review.Key != "" && len(group.ExampleKeys) < 5 {
			group.ExampleKeys = append(group.ExampleKeys, review.Key)
		}
	}
}

func issueContractAgentNoteKeys(notes []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, note := range notes {
		key := issueContractAgentNoteKey(note)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func issueContractAgentNoteKey(note string) string {
	note = strings.TrimSpace(note)
	note = strings.TrimLeft(note, "-* \t")
	note = strings.TrimSpace(note)
	note = strings.Trim(note, "`")
	return note
}

func issueContractSortedAgentNoteGroups(groups map[string]*issueContractAgentNoteGroup) []issueContractAgentNoteGroup {
	out := make([]issueContractAgentNoteGroup, 0, len(groups))
	for _, group := range groups {
		if len(group.ByLane) == 0 {
			group.ByLane = nil
		}
		if len(group.ByWorkUnit) == 0 {
			group.ByWorkUnit = nil
		}
		if len(group.ByReason) == 0 {
			group.ByReason = nil
		}
		out = append(out, *group)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		if out[i].StepBudget != out[j].StepBudget {
			return out[i].StepBudget > out[j].StepBudget
		}
		if out[i].ChildIssueBudget != out[j].ChildIssueBudget {
			return out[i].ChildIssueBudget > out[j].ChildIssueBudget
		}
		return out[i].Key < out[j].Key
	})
	return out
}

func issueContractStepBucket(steps int) string {
	switch {
	case steps <= 0:
		return "(missing)"
	case steps == 1:
		return "1"
	case steps <= 3:
		return "2-3"
	case steps <= issuepolicy.MaxDispatchExpectedSteps:
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

func issueContractBatchPolicyCap(policy string) int {
	tokens := issueContractPolicyTokens(policy)
	for i, tok := range tokens {
		switch tok {
		case "cap", "capped", "limit", "limited", "max", "maximum":
			if n := firstIssueContractPolicyNumber(tokens[i+1:], 4); n > 0 {
				return n
			}
		case "at":
			if i+1 < len(tokens) && tokens[i+1] == "most" {
				if n := firstIssueContractPolicyNumber(tokens[i+2:], 4); n > 0 {
					return n
				}
			}
		case "no":
			if i+2 < len(tokens) && tokens[i+1] == "more" && tokens[i+2] == "than" {
				if n := firstIssueContractPolicyNumber(tokens[i+3:], 4); n > 0 {
					return n
				}
			}
		}
	}
	return 0
}

func issueContractPolicyTokens(policy string) []string {
	return strings.FieldsFunc(strings.ToLower(policy), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	})
}

func firstIssueContractPolicyNumber(tokens []string, limit int) int {
	if limit > len(tokens) {
		limit = len(tokens)
	}
	for i := 0; i < limit; i++ {
		if n := issueContractPolicyNumber(tokens[i]); n > 0 {
			return n
		}
	}
	return 0
}

func issueContractPolicyNumber(token string) int {
	if n, err := strconv.Atoi(token); err == nil {
		return n
	}
	switch token {
	case "one":
		return 1
	case "two":
		return 2
	case "three":
		return 3
	case "four":
		return 4
	case "five":
		return 5
	case "six":
		return 6
	case "seven":
		return 7
	case "eight":
		return 8
	case "nine":
		return 9
	case "ten":
		return 10
	case "twenty":
		return 20
	default:
		return 0
	}
}

func issueContractBatchKey(review issuepolicy.Review) string {
	lane := issueContractBucketValue(review.Lane, "unrouted")
	workUnit := issueContractBucketValue(review.WorkUnit, "missing-work-unit")
	trigger := issueContractBucketValue(review.Trigger, "missing-trigger")
	batchPolicy := issueContractBucketValue(review.BatchPolicy, "missing-batch-policy")
	return lane + "|" + workUnit + "|" + trigger + "|" + batchPolicy
}

func issueContractBatchMissingMetadata(review issuepolicy.Review) []string {
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

func issueContractRepairQueues(reviews []issuepolicy.Review) []issueContractRepairQueue {
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

func issueContractReviewChildIssueBudget(review issuepolicy.Review, kind string) int {
	if kind != "split" {
		return 0
	}
	if review.ExpectedSteps <= 0 {
		return 1
	}
	return (review.ExpectedSteps + issuepolicy.MaxDispatchExpectedSteps - 1) / issuepolicy.MaxDispatchExpectedSteps
}

func issueContractReviewSplitChildIssueBudget(review issuepolicy.Review) int {
	for _, kind := range issueContractRepairKinds(review) {
		if kind == "split" {
			return issueContractReviewChildIssueBudget(review, kind)
		}
	}
	return 0
}

func issueContractRepairKinds(review issuepolicy.Review) []string {
	if review.OK && review.Dispatchability == issuepolicy.Dispatchable {
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
		case issuepolicy.ReasonNotDispatchLeaf, issuepolicy.ReasonOversizedSteps:
			add("split")
		case issuepolicy.ReasonScopeIncomplete:
			add("scope")
		case issuepolicy.ReasonUnrouted:
			add("route")
		case issuepolicy.ReasonLiveUnarmored, issuepolicy.ReasonNoiseIncomplete, issuepolicy.ReasonAgentIncomplete:
			add("noise")
		case issuepolicy.ReasonPrivateBoundary:
			add("private")
		case issuepolicy.ReasonUnexpandedTemplate:
			add("template")
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
	case "template":
		return 6
	default:
		return 9
	}
}

func issueContractRepairAction(kind string) string {
	switch kind {
	case "dispatch":
		return "send these scoped leaves to dispatch lanes, oldest/highest-priority first"
	case "split":
		return fmt.Sprintf("decompose each non-leaf or oversized row into child issues with <= %d expected steps", issuepolicy.MaxDispatchExpectedSteps)
	case "scope":
		return "add the missing parent/current-state/scope/done/witness/closure fields before dispatch"
	case "route":
		return "add a lane or path hints section so the issue maps to one dispatch lane"
	case "noise":
		return "add trigger, batch policy, agent context, and live dedupe/cap evidence before automated sync"
	case "private":
		return "remove private/operator-only evidence or move the work to the private companion repo"
	case "template":
		return "dry-run a normalized generated-header repair, review it, then apply explicitly if accepted"
	default:
		return "inspect the review reasons and repair the row before dispatch"
	}
}

func decodeIssueContractJSONArray[T any](b []byte, emptyErr string, invalidErr func(error) error) ([]T, bool, error) {
	var arr []T
	if err := json.Unmarshal(b, &arr); err != nil {
		if invalidErr == nil {
			return nil, false, nil
		}
		return nil, true, invalidErr(err)
	}
	if len(arr) == 0 {
		return nil, true, errors.New(emptyErr)
	}
	return arr, true, nil
}

func decodeIssueContractCandidates(b []byte) ([]issuepolicy.Candidate, error) {
	if arr, decoded, err := decodeIssueContractJSONArray[issuepolicy.Candidate](b, "candidate list is empty", nil); decoded || err != nil {
		return arr, err
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil {
		return nil, fmt.Errorf("parse candidate JSON: %w", err)
	}
	for _, key := range []string{"candidates", "items"} {
		if raw, ok := obj[key]; ok {
			arr, _, err := decodeIssueContractJSONArray[issuepolicy.Candidate](raw, key+" is empty", func(err error) error {
				return fmt.Errorf("%s must be an issue-candidate array: %w", key, err)
			})
			if err != nil {
				return nil, err
			}
			return arr, nil
		}
	}
	if raw, ok := obj["candidate"]; ok {
		var c issuepolicy.Candidate
		if err := json.Unmarshal(raw, &c); err != nil {
			return nil, fmt.Errorf("candidate must be an issue-candidate object: %w", err)
		}
		return []issuepolicy.Candidate{c}, nil
	}
	var c issuepolicy.Candidate
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("candidate must be an issue-candidate object: %w", err)
	}
	return []issuepolicy.Candidate{c}, nil
}

func decodeIssueContractIssues(b []byte) ([]issuepolicy.IssueDraft, error) {
	if arr, decoded, err := decodeIssueContractJSONArray[issuepolicy.IssueDraft](b, "issue list is empty", nil); decoded || err != nil {
		return arr, err
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil {
		return nil, fmt.Errorf("parse issue JSON: %w", err)
	}
	for _, key := range []string{"issues", "items"} {
		if raw, ok := obj[key]; ok {
			arr, _, err := decodeIssueContractJSONArray[issuepolicy.IssueDraft](raw, key+" is empty", func(err error) error {
				return fmt.Errorf("%s must be a GitHub issue array: %w", key, err)
			})
			if err != nil {
				return nil, err
			}
			return arr, nil
		}
	}
	var issue issuepolicy.IssueDraft
	if err := json.Unmarshal(b, &issue); err != nil {
		return nil, fmt.Errorf("issue must be a GitHub issue object: %w", err)
	}
	return []issuepolicy.IssueDraft{issue}, nil
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
	if r.Counts.GenerationFitMeasured > 0 {
		lines = append(lines, fmt.Sprintf("  generation_fit: measured=%d avg=%d mismatches=%d",
			r.Counts.GenerationFitMeasured, r.Counts.GenerationFitAvg, r.Counts.GenerationMismatches))
	}
	if r.Counts.ModelTierTagged > 0 || r.Counts.ModelTierFlagged > 0 {
		lines = append(lines, fmt.Sprintf("  model_tier: tagged=%d flagged=%d",
			r.Counts.ModelTierTagged, r.Counts.ModelTierFlagged))
	}
	if r.Counts.BornRouted > 0 || r.Counts.BornRoutedFlagged > 0 {
		lines = append(lines, fmt.Sprintf("  born_routed: complete=%d flagged=%d",
			r.Counts.BornRouted, r.Counts.BornRoutedFlagged))
	}
	if r.Counts.ScaleFlagged > 0 {
		lines = append(lines, fmt.Sprintf("  scale: flagged=%d", r.Counts.ScaleFlagged))
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
	if len(r.Counts.ByGeneration) > 0 {
		lines = append(lines, "  generations: "+renderIssueContractReasonCounts(r.Counts.ByGeneration))
	}
	if len(r.Counts.ByRequiredTier) > 0 {
		lines = append(lines, "  required_tiers: "+renderIssueContractReasonCounts(r.Counts.ByRequiredTier))
	}
	if len(r.Counts.ByScale) > 0 {
		lines = append(lines, "  scales: "+renderIssueContractReasonCounts(r.Counts.ByScale))
	}
	for i, group := range r.BatchGroups {
		if i >= 8 {
			lines = append(lines, fmt.Sprintf("  batch_groups: ... %d more", len(r.BatchGroups)-i))
			break
		}
		line := fmt.Sprintf("  batch_group[%d]: count=%d steps=%d",
			i, group.Count, group.StepBudget,
		)
		if group.ChildIssueBudget > 0 {
			line += fmt.Sprintf(" child_issues=%d", group.ChildIssueBudget)
		}
		if group.DeclaredCap > 0 {
			line += fmt.Sprintf(" cap=%d", group.DeclaredCap)
		}
		if group.OverCap > 0 {
			line += fmt.Sprintf(" over_cap=%d", group.OverCap)
		}
		line += fmt.Sprintf(" lane=%s work_unit=%s key=%s",
			issueContractBucketValue(group.Lane, "(unrouted)"),
			issueContractBucketValue(group.WorkUnit, "(missing)"),
			group.Key)
		lines = append(lines, line)
		if len(group.MissingMetadata) > 0 {
			lines = append(lines, "    missing_batch_metadata: "+strings.Join(group.MissingMetadata, ", "))
		}
	}
	for i, group := range r.DuplicateKeyGroups {
		if i >= 8 {
			lines = append(lines, fmt.Sprintf("  duplicate_key_groups: ... %d more", len(r.DuplicateKeyGroups)-i))
			break
		}
		line := fmt.Sprintf("  duplicate_key_group[%d]: count=%d steps=%d",
			i, group.Count, group.StepBudget)
		if group.ChildIssueBudget > 0 {
			line += fmt.Sprintf(" child_issues=%d", group.ChildIssueBudget)
		}
		if len(group.Issues) > 0 {
			line += fmt.Sprintf(" issues=%s", intList(group.Issues))
		}
		line += fmt.Sprintf(" key=%s", group.Key)
		lines = append(lines, line)
		if len(group.ByLane) > 0 {
			lines = append(lines, "    lanes: "+renderIssueContractReasonCounts(group.ByLane))
		}
		if len(group.ByReason) > 0 {
			lines = append(lines, "    reasons: "+renderIssueContractReasonCounts(group.ByReason))
		}
	}
	lines = renderIssueContractAgentNoteGroups(lines, "assumption", r.AssumptionGroups)
	lines = renderIssueContractAgentNoteGroups(lines, "confusion", r.ConfusionGroups)
	lines = renderIssueContractAgentNoteGroups(lines, "coordination", r.CoordinationGroups)
	for i, plan := range r.TemplateRepairPlans {
		if i >= 8 {
			lines = append(lines, fmt.Sprintf("  template_repair_plans: ... %d more", len(r.TemplateRepairPlans)-i))
			break
		}
		line := fmt.Sprintf("  template_repair_plan[%d]: issue=#%d key=%s detected=%q dry_run=%t",
			i,
			plan.IssueNumber,
			issueContractBucketValue(plan.Key, "(missing-key)"),
			plan.DetectedMarker,
			plan.DryRunOnly)
		lines = append(lines, line)
		for _, marker := range plan.DetectedMarkers {
			lines = append(lines, "    marker: "+marker)
		}
		if strings.TrimSpace(plan.ProposedNormalizedHeader) != "" {
			lines = append(lines, "    proposed_header:")
			lines = append(lines, indentIssueContractBlock(plan.ProposedNormalizedHeader, "      ")...)
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
	return renderIssueContractReviews(lines, r.Reviews)
}

func renderIssueContractReviews(lines []string, reviews []issuepolicy.Review) string {
	for _, review := range reviews {
		key := review.Key
		if strings.TrimSpace(key) == "" {
			key = "(missing-key)"
		}
		line := fmt.Sprintf("  [%s] %s dispatchability=%s score=%d spine_priority=%d",
			review.Verdict, key, review.Dispatchability, review.Score.Total, review.SpinePriority.Total)
		if issueContractReviewHasGenerationFit(review) {
			line += fmt.Sprintf(" generation=%s generation_fit=%d",
				issueContractBucketValue(review.GenerationFit.Stream, "(unclassified)"),
				review.GenerationFit.Total)
		}
		if issueContractReviewHasModelTier(review) {
			line += fmt.Sprintf(" model_tier=%s/%s",
				issueContractBucketValue(review.ModelTier.Required, "?"),
				issueContractBucketValue(review.ModelTier.Optimal, "?"))
		}
		if review.BornRouted.Lane != "" || len(review.BornRouted.Flags) > 0 {
			line += fmt.Sprintf(" born_routed=%s/%s/%s",
				issueContractBucketValue(review.BornRouted.Lane, "?"),
				issueContractBucketValue(review.BornRouted.ClassLabel, "?"),
				issueContractBucketValue(review.BornRouted.PriorityLabel, "?"))
		}
		if review.Scale.Effective != "" {
			line += fmt.Sprintf(" scale=%s(%s)",
				string(review.Scale.Effective), issueContractBucketValue(review.Scale.Source, "?"))
		}
		if review.ProblemFrame.Enforced || review.ProblemFrame.Centrality != issuepolicy.CentralityUnclassified {
			line += " centrality=" + issueContractBucketValue(review.ProblemFrame.Centrality, issuepolicy.CentralityUnclassified)
			if review.ProblemFrame.CentralityTarget != "" {
				line += fmt.Sprintf("(%s)", review.ProblemFrame.CentralityTarget)
			}
		}
		if review.OperatingEnvelope.Status != "" && review.OperatingEnvelope.Status != issuepolicy.EnvelopeUndeclared {
			line += fmt.Sprintf(" envelope=%s target=%d witnessed=%d gaps=%d",
				review.OperatingEnvelope.Status, len(review.OperatingEnvelope.Target),
				len(review.OperatingEnvelope.Witnessed), len(review.OperatingEnvelope.Gaps))
		}
		if len(review.ScaleEvidence.Records) > 0 || len(review.ScaleEvidence.RequiredStages) > 0 {
			line += fmt.Sprintf(" scale_evidence=%d required_stages=%d missing_stages=%d",
				len(review.ScaleEvidence.Records), len(review.ScaleEvidence.RequiredStages), len(review.ScaleEvidence.MissingStages))
		}
		if review.ProjectWork.Status != issuepolicy.ProjectWorkUndeclared {
			line += fmt.Sprintf(" project_work=%s estimate=%g contribution=%g/%g completion=%s",
				review.ProjectWork.Status, review.ProjectWork.EstimatePoints, review.ProjectWork.Contribution,
				review.ProjectWork.ParentBaseline, issueContractBucketValue(review.ProjectWork.CompletionStandard, "?"))
		}
		if review.Closure.Status != issuepolicy.ClosureNotRequested {
			line += fmt.Sprintf(" closure=%s claim=%s witnessed=%s production_credit=%t",
				review.Closure.Status, issueContractBucketValue(review.Closure.ClaimedStandard, "?"),
				issueContractBucketValue(review.Closure.WitnessedStandard, "?"), review.Closure.ProductionCredit)
		}
		lines = append(lines, line)
		for _, reason := range review.Reasons {
			lines = append(lines, "    refuses: "+reason)
		}
		for _, reason := range review.ProblemFrame.Reasons {
			lines = append(lines, "    problem_frame: "+reason)
		}
		for _, repair := range review.ProblemFrame.RepairActions {
			lines = append(lines, "    problem_frame_repair: "+repair)
		}
		for _, missing := range review.MissingFields {
			lines = append(lines, "    missing: "+missing)
		}
		for _, flag := range review.GenerationFit.Flags {
			lines = append(lines, "    generation_flag: "+flag)
		}
		for _, flag := range review.ModelTier.Flags {
			lines = append(lines, "    model_tier_flag: "+flag)
		}
		for _, flag := range review.Scale.Flags {
			lines = append(lines, "    scale_flag: "+flag)
		}
		for _, invalid := range review.OperatingEnvelope.Invalid {
			lines = append(lines, "    envelope_invalid: "+invalid)
		}
		for _, gap := range review.OperatingEnvelope.Gaps {
			lines = append(lines, fmt.Sprintf("    envelope_gap: %s: %s", gap.Dimension, gap.Reason))
		}
		for _, invalid := range review.ScaleEvidence.Invalid {
			lines = append(lines, "    scale_evidence_invalid: "+invalid)
		}
		for _, stage := range review.ScaleEvidence.MissingStages {
			lines = append(lines, "    scale_evidence_missing_stage: "+stage)
		}
		for _, invalid := range review.ProjectWork.Invalid {
			lines = append(lines, "    project_work_invalid: "+invalid)
		}
		for _, repair := range review.ProjectWork.Repair {
			lines = append(lines, "    project_work_repair: "+repair)
		}
		for _, reason := range review.Closure.Reasons {
			lines = append(lines, "    closure_refuses: "+reason)
		}
		for _, repair := range review.Closure.Repair {
			lines = append(lines, "    closure_repair: "+repair)
		}
		for _, flag := range review.BornRouted.Flags {
			lines = append(lines, "    born_routed_flag: "+flag)
		}
	}
	return strings.Join(lines, "\n")
}

func indentIssueContractBlock(block, prefix string) []string {
	raw := strings.Split(strings.ReplaceAll(block, "\r\n", "\n"), "\n")
	out := make([]string, 0, len(raw))
	for _, line := range raw {
		out = append(out, prefix+line)
	}
	return out
}

func renderIssueContractAgentNoteGroups(lines []string, label string, groups []issueContractAgentNoteGroup) []string {
	for i, group := range groups {
		if i >= 8 {
			lines = append(lines, fmt.Sprintf("  %s_groups: ... %d more", label, len(groups)-i))
			break
		}
		line := fmt.Sprintf("  %s_group[%d]: count=%d steps=%d",
			label, i, group.Count, group.StepBudget)
		if group.ChildIssueBudget > 0 {
			line += fmt.Sprintf(" child_issues=%d", group.ChildIssueBudget)
		}
		line += fmt.Sprintf(" key=%s", group.Key)
		lines = append(lines, line)
		if len(group.ByLane) > 0 {
			lines = append(lines, "    lanes: "+renderIssueContractReasonCounts(group.ByLane))
		}
		if len(group.ByReason) > 0 {
			lines = append(lines, "    reasons: "+renderIssueContractReasonCounts(group.ByReason))
		}
	}
	return lines
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
	fmt.Fprint(w, `fak-dev issue - generated-issue gates

  fak-dev issue audit    --issue N --author-manifest A
                     --auditor PROVIDER/FAMILY/MODEL --identity-roster R [--json]
  fak-dev issue audit-loop --snapshot SUBJECTS.json [--ledger P] [--cursor P]
                     [--scan-cap N] [--batch-cap N] [--replay N,M] [--json]
  fak-dev issue audit-loop --status [--ledger P] [--cursor P] [--json]
  fak-dev issue contract --file CANDIDATE.json [--json]
  fak-dev issue contract --from-plan PLAN.json [--json]
  fak-dev issue contract --from-issues ISSUES.json [--json]
                     [--live --dedupe-checked --dedupe-cap N]
                     [--strict-model-tier] [--strict-scale]
                     JSON reviews include brief_readiness: ready/enforced plus beneficiary,
                     problem, alternative, advantage, outcome, scope, dependencies,
                     acceptance, witness, and placement fields;
                     each field is present, unknown(reason), or missing with repair_action.
  fak-dev issue cohort   --from-plan PLAN.json [--json]
  fak-dev issue cohort   --from-issues ISSUES.json [--json]
                     [--live --dedupe-checked --dedupe-cap N] [--max-wave N]
  fak-dev issue fanout   --title T --leaf L --spine REF [--parent REF]
                     [--paths p1,p2] [--areas a1,a2] [--max N] [--json]
  fak-dev issue create   --title T (--body B | --body-file F) [--labels l1,l2] [--category C --layer L]
                     [--repo owner/name] [--dry-run] [--json]
  fak-dev issue edit     --issue N [--title T] [--body B | --body-file F]
                     [--add-label l1,l2] [--remove-label l1,l2]
                     [--repo owner/name] [--dry-run] [--json]
  fak-dev issue repair   [--live] [--kind k1,k2] [--issue N,M] [--limit N]
                     [--max-apply N] [--from-issues ISSUES.json]
                     [--repo owner/name] [--json | --markdown]
  fak-dev issue decompose [--live] [--from-plan PLAN.json] [--issue N,M]
                     [--from-issues ISSUES.json] [--allow-stubs]
                     [--max-create N] [--repo owner/name] [--json]
  fak-dev issue dedup    [--json] [--limit N] [--threshold F] [--topk N]
  fak-dev issue dedup    --from-issues ISSUES.json|- [--json]
  fak-dev issue finding  (--ledger LEDGER.jsonl | --receipts RECEIPTS.json)
                     [--from-issues ISSUES.json] [--lane L] [--json]
                     [--live --dedupe-cap N --max-apply N] [--repo owner/name]

The dedup command is the retrospective backlog duplicate census — the read-only
complement of the write-time near-duplicate gate. It builds a body-aware simhash
index over the open backlog (title and title+body axes), clusters near-twins the
title-only Jaccard census was blind to, and emits a ranked merge/close proposal
report where every proposal names its evidence (per-pair similarity on both axes,
shared labels/paths, matched excerpts). It never writes to GitHub — the
confirm-before-closing-as-dup discipline stands. Default reads the live backlog
via gh; --from-issues reads a cached array (offline-safe). Exit 0 report; exit 2
bad flags/input; exit 1 gh/encode failure.

The finding command is the live adapter over the pure cross-audit finding
planner (#3857): it reads a batch of verified audit receipts (--ledger, the
hash-chained receipt ledger, or --receipts, an offline array) plus the findings
already filed (--from-issues), and turns each receipt into one bounded action —
CREATE a new finding issue for a fresh REFUTE, UPDATE/COMMENT one that recurs
with new evidence, REOPEN a closed finding whose subject a REFUTE recurs on, or
ESCALATE an INCONCLUSIVE/UNAVAILABLE receipt for human review instead of
asserting a corruption finding. Repeated identical receipts dedupe onto one
finding. Every generated CREATE is held to the strict armed issue contract, so a
candidate the contract would not admit fails the run (exit 3) rather than being
filed. It DEFAULTS to a dry-run plan that never touches GitHub; --live arms
bounded mutations through the same governed gh atoms and requires a proven
--dedupe-cap, refusing the whole run if planned mutations exceed --max-apply.
Exit 0 plan/all-applied; exit 2 bad flags/unarmed/blast-radius refusal; exit 3 a
generated candidate is not dispatchable; exit 1 a live gh mutation failed.

The create command files one GitHub issue directly, shelling to gh issue
create from the trusted fak binary instead of the agent proposing raw gh
issue create via Bash — the sanctioned smooth path so routine issue filing
does not trip the reversibility/ESCALATE preview-confirm gate (which is
correct to keep escalating an agent's own raw gh/git calls). --dry-run
renders the issue and the exact gh argv without calling gh. Exit 0 created
(or dry-run ok); exit 2 bad flags; exit 1 gh failure.

The audit-loop command runs one deterministic producer tick of the durable
cross-model issue-audit background loop (#3856): it plans a bounded batch from a
discovery snapshot, reconciles the at-most-once receipt ledger and the durable
scheduler cursor, and reports the typed next decision — ADVANCING (a subject
reached a terminal PASS/REFUTE this tick), WAIT (eligible work is only
transiently blocked or everything is settled), STALLED (all pending work has
exhausted its retry budget), or DARK (no eligible subjects — a potential blind
spot). It defaults to a dry-run plan that never leases, audits, or mutates the
ledger; UNAVAILABLE/INCONCLUSIVE subjects never advance the cursor so no
unavailable row is lost. Exit 0 report; exit 2 bad flags/snapshot. --status is
the read-only twin: it verifies the ledger hash-chain and folds the cursor to
report settled coverage, the dead-letter queue, and provider cooldowns without a
snapshot.

The contract command reviews machine-created GitHub issue candidates before a
producer syncs them. Exit 0 means dispatchable; exit 3 means the candidate is
triage-only or refused with closed reasons such as ISSUE_SCOPE_INCOMPLETE,
ISSUE_UNROUTED, ISSUE_NOT_DISPATCH_LEAF, ISSUE_OVERSIZED_EXPECTED_STEPS,
ISSUE_NOISE_CONTROL_INCOMPLETE, ISSUE_AGENT_CONTEXT_INCOMPLETE,
ISSUE_PRIVATE_BOUNDARY, or ISSUE_LIVE_UNARMORED. Every review also reports a
model_tier readout (required/optimal tier, source, and missing/invalid/
contradictory flags); --strict-model-tier turns a flagged issue triage-only
with ISSUE_MODEL_TIER_INCOMPLETE instead of leaving it advisory.

Every review also reports a scale readout on the S0..S4 work-size ladder
(step/leaf/feature/epic/program): the declared size, the size derived from the
step budget and work-unit shape, the effective size, and whether its witness
KIND matches (a feature/epic "done" witnessed only by a commit/test flags
witness_under_scale). A feature-or-larger scale (S2+) is always held off dispatch
as ISSUE_NOT_DISPATCH_LEAF — it must decompose first. --strict-scale additionally
turns an undeclared size or an under-scale witness triage-only with
ISSUE_SCALE_UNDECLARED / ISSUE_WITNESS_SCALE_MISMATCH instead of leaving it
advisory.

The cohort command plans a whole BATCH (1..1000) of candidates at creation time:
it partitions the dispatchable leaves into concurrency-safe waves (the same
disjoint-tree rule dos arbitrate uses), pulls oversized/non-leaf rows into a
split-first queue with a child-issue budget, buckets the rest into triage, and
reports duplicate marker keys. It is a planner, not a gate: exit 0 on a valid
plan, exit 2 on bad input.

The edit command is the governed mutation twin of create: it shells to gh issue
edit N from the trusted fak binary (title/body/add-label/remove-label) so an
apply step never proposes raw gh issue edit via Bash. It is the single write atom
every repair path routes through. --dry-run renders the gh argv without calling
gh. Exit 0 edited (or dry-run ok); exit 2 bad flags; exit 1 gh failure.

The repair command consumes the read-only issue-contract-repair manifest and
turns each repairable row into a disposition: auto-apply (template rows whose
generated-header merge is provably lossless), propose-only (scope/noise/route/
other — the fix is computed or scaffolded but a human/agent applies it via issue
edit), refuse (private-boundary), or defer-to-phase3 (split). It DEFAULTS to a
dry-run plan that never touches GitHub; --live arms writes but writes ONLY
auto-apply template rows, and refuses the whole run if that count exceeds
--max-apply. Exit 0 plan/all-applied; exit 2 bad flags/fetch/blast-radius
refusal; exit 1 a live gh edit failed.
`)
}
