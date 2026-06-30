package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

func TestIssueContractReviewsDispatchableCandidate(t *testing.T) {
	path := writeIssueContractJSON(t, completeIssueCandidate())
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--file", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		OK     bool `json:"ok"`
		Counts struct {
			Total                int            `json:"total"`
			Dispatchable         int            `json:"dispatchable"`
			StepBudget           int            `json:"step_budget"`
			MissingExpectedSteps int            `json:"missing_expected_steps"`
			AgentContextAvg      int            `json:"agent_context_avg"`
			AgentContextFull     int            `json:"agent_context_full"`
			ByReason             map[string]int `json:"by_reason"`
			ByLane               map[string]int `json:"by_lane"`
			ByWorkUnit           map[string]int `json:"by_work_unit"`
			ByExpectedStepBucket map[string]int `json:"by_expected_step_bucket"`
		} `json:"counts"`
		BatchGroups []struct {
			Key         string   `json:"key"`
			Lane        string   `json:"lane"`
			WorkUnit    string   `json:"work_unit"`
			Count       int      `json:"count"`
			StepBudget  int      `json:"step_budget"`
			ExampleKeys []string `json:"example_keys"`
		} `json:"batch_groups"`
		RepairQueues []repairQueueAssertion `json:"repair_queues"`
		Reviews      []struct {
			OK              bool   `json:"ok"`
			Key             string `json:"key"`
			Dispatchability string `json:"dispatchability"`
			WorkUnit        string `json:"work_unit"`
			ExpectedSteps   int    `json:"expected_steps"`
			Trigger         string `json:"trigger"`
			BatchPolicy     string `json:"batch_policy"`
			Score           struct {
				Total int `json:"total"`
			} `json:"score"`
			SpinePriority struct {
				Total int `json:"total"`
			} `json:"spine_priority"`
		} `json:"reviews"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if !got.OK || len(got.Reviews) != 1 || !got.Reviews[0].OK {
		t.Fatalf("review = %+v, want one OK review", got)
	}
	if got.Counts.Total != 1 || got.Counts.Dispatchable != 1 ||
		got.Counts.StepBudget != 3 || got.Counts.MissingExpectedSteps != 0 ||
		got.Counts.AgentContextAvg != 100 || got.Counts.AgentContextFull != 1 ||
		len(got.Counts.ByReason) != 0 {
		t.Fatalf("counts = %+v, want one full-context dispatchable review", got.Counts)
	}
	if got.Counts.ByLane["taskmgr"] != 1 ||
		got.Counts.ByWorkUnit["leaf"] != 1 ||
		got.Counts.ByExpectedStepBucket["2-3"] != 1 {
		t.Fatalf("organization buckets = lane=%+v work_unit=%+v steps=%+v",
			got.Counts.ByLane, got.Counts.ByWorkUnit, got.Counts.ByExpectedStepBucket)
	}
	if len(got.BatchGroups) != 1 || got.BatchGroups[0].Lane != "taskmgr" ||
		got.BatchGroups[0].WorkUnit != "leaf" || got.BatchGroups[0].Count != 1 ||
		got.BatchGroups[0].StepBudget != 3 || len(got.BatchGroups[0].ExampleKeys) != 1 {
		t.Fatalf("batch groups = %+v, want one taskmgr leaf group", got.BatchGroups)
	}
	if len(got.RepairQueues) != 1 || got.RepairQueues[0].Kind != "dispatch" ||
		got.RepairQueues[0].Count != 1 || got.RepairQueues[0].StepBudget != 3 ||
		!strings.Contains(got.RepairQueues[0].NextAction, "dispatch") {
		t.Fatalf("repair queues = %+v, want one dispatch queue", got.RepairQueues)
	}
	if got.Reviews[0].Key != "task_push_next/strict-scope" ||
		got.Reviews[0].Dispatchability != issuecontract.Dispatchable ||
		got.Reviews[0].WorkUnit != "leaf" ||
		got.Reviews[0].ExpectedSteps != 3 ||
		got.Reviews[0].Trigger == "" ||
		got.Reviews[0].BatchPolicy == "" ||
		got.Reviews[0].Score.Total != 100 ||
		got.Reviews[0].SpinePriority.Total != 100 {
		t.Fatalf("review identity = %+v", got.Reviews[0])
	}
}

func TestIssueContractRefusesVagueCandidate(t *testing.T) {
	c := completeIssueCandidate()
	c.OutOfScope = ""
	c.DoneCondition = ""
	c.Lane = ""
	c.Paths = nil
	path := writeIssueContractJSON(t, c)
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--file", path})
	if code != 3 {
		t.Fatalf("exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	rendered := out.String()
	for _, want := range []string{
		"counts: dispatchable=0 triage_only=1 refused=0",
		"reasons: ISSUE_SCOPE_INCOMPLETE=1, ISSUE_UNROUTED=1",
		"lanes: (unrouted)=1",
		"work_units: leaf=1",
		"step_buckets: 2-3=1",
		"batch_group[0]: count=1 steps=3 lane=(unrouted) work_unit=leaf",
		"coordination_group[0]: count=1 steps=3 key=Do not dispatch concurrently",
		"repair_queue[scope]: count=1 steps=3",
		"repair_queue[route]: count=1 steps=3",
		"ISSUE_SCOPE_INCOMPLETE",
		"ISSUE_UNROUTED",
		"missing: out_of_scope",
		"missing: done_condition",
	} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("rendered review missing %q:\n%s", want, rendered)
		}
	}
}

func TestIssueContractLiveRequiresDedupeArmor(t *testing.T) {
	path := writeIssueContractJSON(t, completeIssueCandidate())
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--file", path, "--live", "--json"})
	if code != 3 {
		t.Fatalf("unarmored live exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), issuecontract.ReasonLiveUnarmored) {
		t.Fatalf("unarmored live output missing %s:\n%s", issuecontract.ReasonLiveUnarmored, out.String())
	}

	out.Reset()
	errb.Reset()
	code = runIssue(&out, &errb, []string{
		"contract", "--file", path, "--live", "--dedupe-checked", "--dedupe-cap", "300", "--json",
	})
	if code != 0 {
		t.Fatalf("armed live exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
}

func TestIssueContractFromPlanReviewsCandidatesArray(t *testing.T) {
	path := filepath.Join(t.TempDir(), "plan.json")
	body := map[string]any{"candidates": []issuecontract.Candidate{completeIssueCandidate()}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--from-plan", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), `"mode": "plan"`) {
		t.Fatalf("plan mode missing:\n%s", out.String())
	}
}

func TestIssueContractFromIssuesReviewsGitHubRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuecontract.IssueDraft{{
		Number: 1450,
		Title:  "guardrsi: require block reasons",
		Body:   completeIssueDraftBody(),
		Labels: []issuecontract.IssueLabel{{Name: "guardrsi"}},
	}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--from-issues", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), `"mode": "issues"`) ||
		!strings.Contains(out.String(), `"key": "issue/1450"`) ||
		!strings.Contains(out.String(), `"dispatchability": "dispatchable"`) {
		t.Fatalf("issue review missing expected fields:\n%s", out.String())
	}
}

func TestIssueContractSummarizesMixedIssueAuditCounts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuecontract.IssueDraft{
		{
			Number: 1450,
			Title:  "guardrsi: require block reasons",
			Body:   completeIssueDraftBody(),
			Labels: []issuecontract.IssueLabel{{Name: "guardrsi"}},
		},
		{
			Number: 1451,
			Title:  "make it better",
			Body:   "### Current state\nExists.\n",
		},
		{
			Number: 1452,
			Title:  "guardrsi: split oversized block-reason work",
			Body:   completeIssueDraftBodyWithSteps("12"),
			Labels: []issuecontract.IssueLabel{{Name: "guardrsi"}},
		},
	}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--from-issues", path, "--json"})
	if code != 3 {
		t.Fatalf("exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	var got struct {
		OK     bool `json:"ok"`
		Counts struct {
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
		} `json:"counts"`
		BatchGroups []struct {
			Key              string   `json:"key"`
			Count            int      `json:"count"`
			StepBudget       int      `json:"step_budget"`
			ChildIssueBudget int      `json:"child_issue_budget"`
			MissingMetadata  []string `json:"missing_metadata"`
		} `json:"batch_groups"`
		CoordinationGroups []struct {
			Key              string         `json:"key"`
			Count            int            `json:"count"`
			StepBudget       int            `json:"step_budget"`
			ChildIssueBudget int            `json:"child_issue_budget"`
			ByLane           map[string]int `json:"by_lane"`
			ByReason         map[string]int `json:"by_reason"`
			ExampleKeys      []string       `json:"example_keys"`
		} `json:"coordination_groups"`
		RepairQueues []repairQueueAssertion `json:"repair_queues"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.OK {
		t.Fatalf("ok = true, want mixed audit to fail")
	}
	if got.Counts.Total != 3 || got.Counts.Dispatchable != 1 || got.Counts.TriageOnly != 2 || got.Counts.Refused != 0 {
		t.Fatalf("dispatch counts = %+v, want one dispatchable and two triage-only", got.Counts)
	}
	if got.Counts.StepBudget != 16 || got.Counts.MissingExpectedSteps != 1 {
		t.Fatalf("step counts = %+v, want fallback step budget 16 and one missing expected step", got.Counts)
	}
	if got.Counts.AgentContextAvg != 67 || got.Counts.AgentContextFull != 2 || got.Counts.AgentContextMissing != 1 {
		t.Fatalf("agent context counts = %+v, want two full and one missing", got.Counts)
	}
	if got.Counts.ByReason[issuecontract.ReasonScopeIncomplete] != 1 ||
		got.Counts.ByReason[issuecontract.ReasonUnrouted] != 1 ||
		got.Counts.ByReason[issuecontract.ReasonOversizedSteps] != 1 {
		t.Fatalf("reason counts = %+v, want scope, unrouted, and oversized refusals", got.Counts.ByReason)
	}
	if got.Counts.ByLane["guardrsi"] != 2 || got.Counts.ByLane["(unrouted)"] != 1 {
		t.Fatalf("lane buckets = %+v, want guardrsi and unrouted", got.Counts.ByLane)
	}
	if got.Counts.ByWorkUnit["leaf"] != 2 || got.Counts.ByWorkUnit["(missing)"] != 1 {
		t.Fatalf("work-unit buckets = %+v, want leaf and missing", got.Counts.ByWorkUnit)
	}
	if got.Counts.ByExpectedStepBucket["2-3"] != 1 ||
		got.Counts.ByExpectedStepBucket["(missing)"] != 1 ||
		got.Counts.ByExpectedStepBucket["over-8"] != 1 {
		t.Fatalf("step buckets = %+v, want 2-3 and missing", got.Counts.ByExpectedStepBucket)
	}
	if len(got.BatchGroups) != 2 || got.BatchGroups[0].Count != 2 || got.BatchGroups[0].StepBudget != 15 ||
		got.BatchGroups[0].ChildIssueBudget != 2 {
		t.Fatalf("batch groups = %+v, want guardrsi rows grouped under shared trigger/batch with two child issues", got.BatchGroups)
	}
	if len(got.CoordinationGroups) != 1 ||
		got.CoordinationGroups[0].Count != 2 ||
		got.CoordinationGroups[0].StepBudget != 15 ||
		got.CoordinationGroups[0].ChildIssueBudget != 2 ||
		got.CoordinationGroups[0].ByLane["guardrsi"] != 2 ||
		got.CoordinationGroups[0].ByReason[issuecontract.ReasonOversizedSteps] != 1 ||
		len(got.CoordinationGroups[0].ExampleKeys) != 2 ||
		!strings.Contains(got.CoordinationGroups[0].Key, "Avoid concurrent edits") {
		t.Fatalf("coordination groups = %+v, want shared guardrsi coordination group with split budget", got.CoordinationGroups)
	}
	assertRepairQueue(t, got.RepairQueues, "dispatch", 1, 3, nil)
	assertRepairQueue(t, got.RepairQueues, "split", 1, 12, map[string]int{issuecontract.ReasonOversizedSteps: 1}, 2)
	assertRepairQueue(t, got.RepairQueues, "scope", 1, 1, map[string]int{issuecontract.ReasonScopeIncomplete: 1})
	assertRepairQueue(t, got.RepairQueues, "route", 1, 1, map[string]int{issuecontract.ReasonUnrouted: 1})
	scopeQueue := repairQueueByKind(got.RepairQueues, "scope")
	if scopeQueue.MissingFields["parent_ref"] != 1 || scopeQueue.MissingFields["done_condition"] != 1 {
		t.Fatalf("scope missing fields = %+v, want parent_ref and done_condition", scopeQueue.MissingFields)
	}
}

func TestIssueContractFromIssuesRefusesVagueRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuecontract.IssueDraft{{Number: 1451, Title: "make it better", Body: "### Current state\nExists.\n"}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--from-issues", path})
	if code != 3 {
		t.Fatalf("exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	for _, want := range []string{"issue/1451", issuecontract.ReasonScopeIncomplete, issuecontract.ReasonUnrouted} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("rendered review missing %q:\n%s", want, out.String())
		}
	}
}

func completeIssueDraftBody() string {
	return completeIssueDraftBodyWithSteps("3")
}

func completeIssueDraftBodyWithSteps(expectedSteps string) string {
	return strings.Join([]string{
		"### Parent context",
		"guard-verdict-rsi",
		"### Current state",
		"A guard verdict can reach the journal without a closed reason.",
		"### Why this is next",
		"Reasonless blocks weaken the guard before any tuning work.",
		"### Working spine",
		"Every blocked guard verdict records one closed-vocabulary reason.",
		"### Priority context",
		"Working path: guard preflight to closed reason.",
		"Current blocker: reasonless guard blocks hide the failing gate.",
		"Unblocks: guard tuning depends on reason buckets.",
		"Not polish: fix the smallest guard hole before threshold optimization.",
		"### Work unit",
		"leaf",
		"### Expected steps",
		expectedSteps,
		"### Assumptions",
		"- The guard journal fixture can reproduce the blank reason.",
		"### Confusion risks",
		"- Reason labels and threshold tuning are adjacent but separate.",
		"### Coordination notes",
		"- Avoid concurrent edits to the guard reason taxonomy.",
		"### Trigger",
		"Guard journal emits a denied verdict with no reason.",
		"### Batch policy",
		"One issue per repeated reason class; update existing marker on rerun.",
		"### In scope",
		"Add the missing classification and one regression fixture.",
		"### Out of scope",
		"Do not retune guard thresholds.",
		"### Done condition",
		"The fixture no longer emits a blank reason.",
		"### Witness",
		"go test ./internal/guardrsi",
		"### Acceptance gate",
		"go test ./internal/guardrsi ./internal/guardroute",
		"### Lane",
		"guardrsi",
		"### Path hints",
		"- `internal/guardrsi/**`",
		"### Boundary notes",
		"- Public issue only.",
		"### Closure binding",
		"Resolving commit cites #N and carries `(fak guardrsi)`.",
	}, "\n")
}

type repairQueueAssertion struct {
	Kind             string         `json:"kind"`
	Count            int            `json:"count"`
	StepBudget       int            `json:"step_budget"`
	ChildIssueBudget int            `json:"child_issue_budget"`
	NextAction       string         `json:"next_action"`
	ByReason         map[string]int `json:"by_reason"`
	MissingFields    map[string]int `json:"missing_fields"`
	ExampleKeys      []string       `json:"example_keys"`
}

func assertRepairQueue(t *testing.T, queues []repairQueueAssertion, kind string, count, steps int, reasons map[string]int, childIssueBudget ...int) {
	t.Helper()
	queue := repairQueueByKind(queues, kind)
	if queue.Kind == "" {
		t.Fatalf("repair queue %q missing from %+v", kind, queues)
	}
	if queue.Count != count || queue.StepBudget != steps || queue.NextAction == "" || len(queue.ExampleKeys) == 0 {
		t.Fatalf("repair queue %q = %+v, want count=%d steps=%d action/examples", kind, queue, count, steps)
	}
	if len(childIssueBudget) > 0 && queue.ChildIssueBudget != childIssueBudget[0] {
		t.Fatalf("repair queue %q child issue budget = %d, want %d", kind, queue.ChildIssueBudget, childIssueBudget[0])
	}
	for reason, want := range reasons {
		if queue.ByReason[reason] != want {
			t.Fatalf("repair queue %q reasons = %+v, want %s=%d", kind, queue.ByReason, reason, want)
		}
	}
}

func repairQueueByKind(queues []repairQueueAssertion, kind string) repairQueueAssertion {
	for _, queue := range queues {
		if queue.Kind == kind {
			return queue
		}
	}
	return repairQueueAssertion{}
}

func completeIssueCandidate() issuecontract.Candidate {
	return issuecontract.Candidate{
		Schema:          issuecontract.Schema,
		Key:             "task_push_next/strict-scope",
		Title:           "taskmgr: enforce strict handoff scope",
		ParentRef:       "task_push_next",
		CurrentState:    "Task handoff can already create stable follow-up issues.",
		WhyNow:          "Generated issues are the next weak point before dispatch.",
		WorkingSpine:    "A verified task completion creates one scoped follow-up issue.",
		PriorityContext: "Working path: clean Stop handoff -> scoped issue -> dispatch. Current blocker: vague follow-ups waste dispatch cycles. Unblocks: guard live handoff. Not polish: enforce the smallest leaf before optimization.",
		WorkUnit:        "leaf",
		ExpectedSteps:   3,
		Assumptions:     []string{"The handoff producer can derive the candidate before syncing."},
		ConfusionRisks:  []string{"A broad follow-up can be mistaken for an epic unless scoped."},
		Coordination:    []string{"Do not dispatch concurrently with taskmgr handoff body edits."},
		Trigger:         "A verified completion handoff proposes this next leaf.",
		BatchPolicy:     "At most two follow-up issues per handoff; update by marker key on rerun.",
		InScope:         "Review the next-step candidate and render scoped sections.",
		OutOfScope:      "Do not optimize issue routing or add new scorecards.",
		DoneCondition:   "Legacy handoffs pass by default; strict handoffs refuse vague next steps.",
		Witness:         "go test ./internal/taskmgr",
		AcceptanceGate:  "go test ./cmd/fak -run TestIssueContract",
		Lane:            "taskmgr",
		Paths:           []string{"internal/taskmgr/handoff.go"},
		BoundaryNotes:   []string{"Public issue only; no private lab evidence."},
		ClosureBinding:  "Resolving commit cites #N and carries a matching (fak <leaf>) trailer.",
	}
}

func writeIssueContractJSON(t *testing.T, c issuecontract.Candidate) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "candidate.json")
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
