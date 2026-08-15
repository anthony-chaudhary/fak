package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func TestIssueContractReviewsDispatchableCandidate(t *testing.T) {
	path := writeIssueContractJSON(t, completeIssueCandidate())
	var out, errb bytes.Buffer
	code := RunIssue(&out, &errb, []string{"contract", "--file", path, "--json"})
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
			DeclaredCap int      `json:"declared_cap"`
			OverCap     int      `json:"over_cap"`
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
		got.BatchGroups[0].StepBudget != 3 || got.BatchGroups[0].DeclaredCap != 2 ||
		got.BatchGroups[0].OverCap != 0 || len(got.BatchGroups[0].ExampleKeys) != 1 {
		t.Fatalf("batch groups = %+v, want one taskmgr leaf group", got.BatchGroups)
	}
	if len(got.RepairQueues) != 1 || got.RepairQueues[0].Kind != "dispatch" ||
		got.RepairQueues[0].Count != 1 || got.RepairQueues[0].StepBudget != 3 ||
		!strings.Contains(got.RepairQueues[0].NextAction, "dispatch") {
		t.Fatalf("repair queues = %+v, want one dispatch queue", got.RepairQueues)
	}
	if got.Reviews[0].Key != "task_push_next/strict-scope" ||
		got.Reviews[0].Dispatchability != issuepolicy.Dispatchable ||
		got.Reviews[0].WorkUnit != "leaf" ||
		got.Reviews[0].ExpectedSteps != 3 ||
		got.Reviews[0].Trigger == "" ||
		got.Reviews[0].BatchPolicy == "" ||
		got.Reviews[0].Score.Total != 100 ||
		got.Reviews[0].SpinePriority.Total != 100 {
		t.Fatalf("review identity = %+v", got.Reviews[0])
	}
}

func TestIssueContractJSONIncludesShiftLeftReadinessSchema(t *testing.T) {
	body := "## Value\n\n- For: maintainers reviewing issue contracts\n- Problem: frame decisions can disappear before dispatch\n- Today: reviewers reconstruct them manually\n- Better because: typed intake catches omissions early\n- Centrality: Enabling (reliable dispatch contracts)\n- P1: preserved - no runtime context is duplicated\n- P2: advanced - review rework is removed\n- P3: preserved - qualitative decisions remain revisable\n- P4: advanced - the real contract path exposes the frame\n\n## Outcome\n\ntyped readiness\n\n## Scope / tree\n\ninternal/issuepolicy/**\n\n## Dependencies\n\nnone\n\n## Acceptance\n\nJSON fields are typed\n\n## Witness / proof\n\nfocused test\n\n## Placement\n\ngen/now, P1, issuecontract lane\n"
	payload, _ := json.Marshal([]issuepolicy.IssueDraft{{Number: 6421, Title: "validate briefs", Body: body}})
	path := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	// This intentionally minimal issue is not a complete legacy dispatch contract,
	// so the command exits 3 while still returning the readiness read-back.
	if code := RunIssue(&stdout, &stderr, []string{"contract", "--from-issues", path, "--json"}); code != 3 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got issueContractResult
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ReadinessSchema != issuepolicy.TaskBriefSchema || got.ProblemFrameSchema != issuepolicy.ProblemFrameSchema || !got.Reviews[0].BriefReadiness.Ready || !got.Reviews[0].BriefReadiness.Enforced || !got.Reviews[0].ProblemFrame.Ready {
		t.Fatalf("result=%+v", got)
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
	code := RunIssue(&out, &errb, []string{"contract", "--file", path})
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
		"batch_group[0]: count=1 steps=3 cap=2 lane=(unrouted) work_unit=leaf",
		"assumption_group[0]: count=1 steps=3 key=The handoff producer can derive",
		"confusion_group[0]: count=1 steps=3 key=A broad follow-up can be mistaken",
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
	code := RunIssue(&out, &errb, []string{"contract", "--file", path, "--live", "--json"})
	if code != 3 {
		t.Fatalf("unarmored live exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), issuepolicy.ReasonLiveUnarmored) {
		t.Fatalf("unarmored live output missing %s:\n%s", issuepolicy.ReasonLiveUnarmored, out.String())
	}

	out.Reset()
	errb.Reset()
	code = RunIssue(&out, &errb, []string{
		"contract", "--file", path, "--live", "--dedupe-checked", "--dedupe-cap", "300", "--json",
	})
	if code != 0 {
		t.Fatalf("armed live exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
}

func TestIssueContractFromPlanReviewsCandidatesArray(t *testing.T) {
	path := filepath.Join(t.TempDir(), "plan.json")
	body := map[string]any{"candidates": []issuepolicy.Candidate{completeIssueCandidate()}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := RunIssue(&out, &errb, []string{"contract", "--from-plan", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), `"mode": "plan"`) {
		t.Fatalf("plan mode missing:\n%s", out.String())
	}
}

func TestIssueContractFromIssuesReviewsGitHubRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuepolicy.IssueDraft{{
		Number: 1450,
		Title:  "guardrsi: require block reasons",
		Body:   completeIssueDraftBody(),
		Labels: []issuepolicy.IssueLabel{{Name: "guardrsi"}},
	}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := RunIssue(&out, &errb, []string{"contract", "--from-issues", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), `"mode": "issues"`) ||
		!strings.Contains(out.String(), `"key": "issue/1450"`) ||
		!strings.Contains(out.String(), `"dispatchability": "dispatchable"`) {
		t.Fatalf("issue review missing expected fields:\n%s", out.String())
	}
}

func TestIssueContractFromIssuesEmitsTemplateRepairPlan(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuepolicy.IssueDraft{{
		Number: 1727,
		Title:  "generation(second-next): build the multi-generation portfolio optimizer",
		Body:   corruptGenerationIssueBody(),
		Labels: []issuepolicy.IssueLabel{{Name: "generation"}, {Name: "gen/second-next"}},
	}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := RunIssue(&out, &errb, []string{"contract", "--from-issues", path, "--json"})
	if code != 3 {
		t.Fatalf("exit = %d, want 3 for corrupt generated body\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	var got struct {
		OK                  bool `json:"ok"`
		TemplateRepairPlans []struct {
			IssueNumber              int      `json:"issue_number"`
			DetectedMarker           string   `json:"detected_marker"`
			DetectedMarkers          []string `json:"detected_markers"`
			ProposedNormalizedHeader string   `json:"proposed_normalized_header"`
			DryRunOnly               bool     `json:"dry_run_only"`
		} `json:"template_repair_plans"`
		RepairQueues []repairQueueAssertion `json:"repair_queues"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.OK || len(got.TemplateRepairPlans) != 1 {
		t.Fatalf("result = %+v, want one failed dry-run repair plan", got)
	}
	plan := got.TemplateRepairPlans[0]
	if plan.IssueNumber != 1727 || !plan.DryRunOnly ||
		plan.DetectedMarker != "$(@{gen=second-next; title=...; labels=...; why=...; scope=...}.gen)" {
		t.Fatalf("plan = %+v, want issue #1727 dry-run with first marker", plan)
	}
	if !hasString(plan.DetectedMarkers, "$(System.Collections.Hashtable.title)") ||
		!hasString(plan.DetectedMarkers, "- Source: $source, Phase 2") {
		t.Fatalf("markers = %+v, want 1727-style marker set", plan.DetectedMarkers)
	}
	if !strings.Contains(plan.ProposedNormalizedHeader, "- Generation: gen/second-next") ||
		!strings.Contains(plan.ProposedNormalizedHeader, "- Parent: #1625") ||
		strings.Contains(plan.ProposedNormalizedHeader, "$(") {
		t.Fatalf("proposed header = %q, want normalized generation header", plan.ProposedNormalizedHeader)
	}
	assertRepairQueue(t, got.RepairQueues, "template", 1, 1, map[string]int{issuepolicy.ReasonUnexpandedTemplate: 1})

	out.Reset()
	errb.Reset()
	code = RunIssue(&out, &errb, []string{"contract", "--from-issues", path})
	if code != 3 {
		t.Fatalf("text exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	rendered := out.String()
	for _, want := range []string{
		"template_repair_plan[0]: issue=#1727",
		"detected=\"$(@{gen=second-next; title=...; labels=...; why=...; scope=...}.gen)\" dry_run=true",
		"marker: $(System.Collections.Hashtable.title)",
		"marker: - Source: $source, Phase 2",
		"proposed_header:",
		"- Generation: gen/second-next",
	} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("rendered output missing %q:\n%s", want, rendered)
		}
	}
}

func TestIssueContractReportsGenerationFit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "generation-candidates.json")
	clean := completeIssueCandidate()
	clean.Generation = "gen/next"
	clean.Title = "generation(next): add branchless feature gating"
	clean.Labels = []string{"generation", "gen/next"}
	clean.WhyNow = "Next gen near-term foundation work needs a gate, handoff, and operator visibility before it is agent-runnable."
	clean.InScope = "Add the generation checklist with promotion evidence, demotion evidence, and runtime feature gate boundaries."
	clean.OutOfScope = "Do not create a branch per generation; priority, shared trunk, and runtime feature gates remain orthogonal."
	clean.DoneCondition = "The issue names promotion evidence, demotion/retirement evidence, and an invalidating assumption."
	clean.Witness = "Captured command witness from fak-dev issue contract."
	clean.Assumptions = []string{"Invalidating assumption: generation labels stay available during issue grooming."}

	mismatch := clean
	mismatch.Key = "generation/mismatch"
	mismatch.Labels = []string{"generation", "gen/future"}
	body := []issuepolicy.Candidate{clean, mismatch}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := RunIssue(&out, &errb, []string{"contract", "--file", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	var got struct {
		Counts struct {
			GenerationFitMeasured int            `json:"generation_fit_measured"`
			GenerationMismatches  int            `json:"generation_mismatches"`
			ByGeneration          map[string]int `json:"by_generation"`
		} `json:"counts"`
		Reviews []struct {
			GenerationFit struct {
				Stream string   `json:"stream"`
				Total  int      `json:"total"`
				Flags  []string `json:"flags"`
			} `json:"generation_fit"`
		} `json:"reviews"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.Counts.GenerationFitMeasured != 2 || got.Counts.GenerationMismatches != 1 ||
		got.Counts.ByGeneration["gen/next"] != 1 || got.Counts.ByGeneration["gen/future"] != 1 {
		t.Fatalf("generation counts = %+v, want two measured rows with one mismatch", got.Counts)
	}
	if len(got.Reviews) != 2 || got.Reviews[0].GenerationFit.Total != 100 ||
		got.Reviews[1].GenerationFit.Stream != "gen/future" ||
		!hasString(got.Reviews[1].GenerationFit.Flags, "generation_body_mismatch") {
		t.Fatalf("generation reviews = %+v, want clean first row and mismatched second row", got.Reviews)
	}
}

func TestIssueContractModelTierReadout(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuepolicy.IssueDraft{{
		Number: 3041,
		Title:  "modeltier(C4): issue contract parses required and optimal model-tier tags",
		Body:   completeIssueDraftBody(),
		Labels: []issuepolicy.IssueLabel{
			{Name: "guardrsi"},
			{Name: "tier/T1-required"},
			{Name: "tier/T1-optimal"},
			{Name: "priority/P1"},
		},
	}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := RunIssue(&out, &errb, []string{"contract", "--from-issues", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	var got struct {
		Counts struct {
			ModelTierTagged  int            `json:"model_tier_tagged"`
			ModelTierFlagged int            `json:"model_tier_flagged"`
			ByRequiredTier   map[string]int `json:"by_required_tier"`
		} `json:"counts"`
		Reviews []struct {
			ModelTier struct {
				Required       string   `json:"required"`
				Optimal        string   `json:"optimal"`
				RequiredSource string   `json:"required_source"`
				OptimalSource  string   `json:"optimal_source"`
				Flags          []string `json:"flags"`
			} `json:"model_tier"`
		} `json:"reviews"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.Counts.ModelTierTagged != 1 || got.Counts.ModelTierFlagged != 0 || got.Counts.ByRequiredTier["T1"] != 1 {
		t.Fatalf("model tier counts = %+v, want one T1-tagged, unflagged review", got.Counts)
	}
	if len(got.Reviews) != 1 || got.Reviews[0].ModelTier.Required != "T1" ||
		got.Reviews[0].ModelTier.Optimal != "T1" ||
		got.Reviews[0].ModelTier.RequiredSource != "label" ||
		got.Reviews[0].ModelTier.OptimalSource != "label" ||
		len(got.Reviews[0].ModelTier.Flags) != 0 {
		t.Fatalf("model tier review = %+v, want T1/T1 from labels with no flags", got.Reviews)
	}

	// Strict mode on an untagged (but otherwise dispatchable) issue holds it
	// triage-only with the closed reason; default mode leaves it dispatchable.
	untagged := []issuepolicy.IssueDraft{{
		Number: 3042,
		Title:  "guardrsi: untagged block reasons",
		Body:   completeIssueDraftBody(),
		Labels: []issuepolicy.IssueLabel{{Name: "guardrsi"}},
	}}
	ub, err := json.Marshal(untagged)
	if err != nil {
		t.Fatal(err)
	}
	upath := filepath.Join(t.TempDir(), "untagged.json")
	if err := os.WriteFile(upath, ub, 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	if code := RunIssue(&out, &errb, []string{"contract", "--from-issues", upath}); code != 0 {
		t.Fatalf("default untagged exit = %d, want 0 (advisory)\nstdout:\n%s", code, out.String())
	}
	out.Reset()
	errb.Reset()
	code = RunIssue(&out, &errb, []string{"contract", "--from-issues", upath, "--strict-model-tier"})
	if code != 3 {
		t.Fatalf("strict untagged exit = %d, want 3\nstdout:\n%s", code, out.String())
	}
	if !strings.Contains(out.String(), issuepolicy.ReasonModelTierIncomplete) {
		t.Fatalf("strict untagged output missing %s:\n%s", issuepolicy.ReasonModelTierIncomplete, out.String())
	}
}

func TestIssueContractScaleReadout(t *testing.T) {
	// A dispatchable leaf carries an S1 scale readout derived from its step
	// budget, with a matching test/gate witness and no scale flags.
	path := writeIssueContractJSON(t, completeIssueCandidate())
	var out, errb bytes.Buffer
	if code := RunIssue(&out, &errb, []string{"contract", "--file", path, "--json"}); code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		Counts struct {
			ScaleFlagged int            `json:"scale_flagged"`
			ByScale      map[string]int `json:"by_scale"`
		} `json:"counts"`
		Reviews []struct {
			Scale struct {
				Effective    string   `json:"effective"`
				Source       string   `json:"source"`
				Dispatchable bool     `json:"dispatchable"`
				WitnessScale string   `json:"witness_scale"`
				Flags        []string `json:"flags"`
			} `json:"scale"`
		} `json:"reviews"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.Counts.ScaleFlagged != 0 || got.Counts.ByScale["S1"] != 1 {
		t.Fatalf("scale counts = %+v, want one unflagged S1", got.Counts)
	}
	if len(got.Reviews) != 1 {
		t.Fatalf("reviews = %d, want 1", len(got.Reviews))
	}
	s := got.Reviews[0].Scale
	if s.Effective != "S1" || s.Source != "steps" || !s.Dispatchable ||
		s.WitnessScale != "S1" || len(s.Flags) != 0 {
		t.Fatalf("scale readout = %+v, want dispatchable S1 from steps, S1 witness, no flags", s)
	}

	// A feature-shaped unit with a leaf-sized step budget is S2 by shape: the
	// small budget must not shrink it below its shape. S2+ is always held off
	// dispatch (ISSUE_NOT_DISPATCH_LEAF), and its commit/test witness is smaller
	// than the work, so witness_under_scale flags even without strict mode.
	feature := completeIssueCandidate()
	feature.WorkUnit = "feature"
	feature.ExpectedSteps = 4
	fpath := writeIssueContractJSON(t, feature)
	out.Reset()
	errb.Reset()
	code := RunIssue(&out, &errb, []string{"contract", "--file", fpath})
	if code != 3 {
		t.Fatalf("feature default exit = %d, want 3\nstdout:\n%s", code, out.String())
	}
	if !strings.Contains(out.String(), issuepolicy.ReasonNotDispatchLeaf) {
		t.Fatalf("feature output missing %s:\n%s", issuepolicy.ReasonNotDispatchLeaf, out.String())
	}
	if !strings.Contains(out.String(), "scale=S2") || !strings.Contains(out.String(), "witness_under_scale") {
		t.Fatalf("feature output missing S2 scale readout / under-scale flag:\n%s", out.String())
	}

	// --strict-scale additionally holds the under-scale witness with its own
	// closed reason.
	out.Reset()
	errb.Reset()
	if code := RunIssue(&out, &errb, []string{"contract", "--file", fpath, "--strict-scale"}); code != 3 {
		t.Fatalf("feature strict-scale exit = %d, want 3\nstdout:\n%s", code, out.String())
	}
	if !strings.Contains(out.String(), issuepolicy.ReasonWitnessScaleMismatch) {
		t.Fatalf("strict-scale output missing %s:\n%s", issuepolicy.ReasonWitnessScaleMismatch, out.String())
	}
}

func TestIssueContractFlagsBatchGroupsOverDeclaredCap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "candidates.json")
	candidates := make([]issuepolicy.Candidate, 0, 3)
	for _, key := range []string{"cap-batch/one", "cap-batch/two", "cap-batch/three"} {
		c := completeIssueCandidate()
		c.Key = key
		c.BatchPolicy = "At most 2 creates per live wave; update by marker key on rerun."
		candidates = append(candidates, c)
	}
	b, err := json.Marshal(candidates)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := RunIssue(&out, &errb, []string{"contract", "--file", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	var got struct {
		BatchGroups []struct {
			Count       int `json:"count"`
			StepBudget  int `json:"step_budget"`
			DeclaredCap int `json:"declared_cap"`
			OverCap     int `json:"over_cap"`
		} `json:"batch_groups"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if len(got.BatchGroups) != 1 ||
		got.BatchGroups[0].Count != 3 ||
		got.BatchGroups[0].StepBudget != 9 ||
		got.BatchGroups[0].DeclaredCap != 2 ||
		got.BatchGroups[0].OverCap != 1 {
		t.Fatalf("batch groups = %+v, want count=3 steps=9 cap=2 over_cap=1", got.BatchGroups)
	}
}

func TestIssueContractGroupsDuplicateGeneratedMarkers(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuepolicy.IssueDraft{
		{
			Number: 1450,
			Title:  "taskmgr: follow up",
			Body:   "<!-- fak-task-handoff-key: task_push_next/issue-sync -->\n" + completeIssueDraftBody(),
			Labels: []issuepolicy.IssueLabel{{Name: "guardrsi"}},
		},
		{
			Number: 1453,
			Title:  "taskmgr: duplicate follow up",
			Body:   "<!-- fak-task-handoff-key: task_push_next/issue-sync -->\n" + completeIssueDraftBody(),
			Labels: []issuepolicy.IssueLabel{{Name: "guardrsi"}},
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
	code := RunIssue(&out, &errb, []string{"contract", "--from-issues", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	var got struct {
		DuplicateKeyGroups []struct {
			Key        string         `json:"key"`
			Count      int            `json:"count"`
			StepBudget int            `json:"step_budget"`
			Issues     []int          `json:"issues"`
			ByLane     map[string]int `json:"by_lane"`
		} `json:"duplicate_key_groups"`
		Reviews []struct {
			Key         string `json:"key"`
			IssueNumber int    `json:"issue_number"`
		} `json:"reviews"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if len(got.DuplicateKeyGroups) != 1 ||
		got.DuplicateKeyGroups[0].Key != "task_push_next/issue-sync" ||
		got.DuplicateKeyGroups[0].Count != 2 ||
		got.DuplicateKeyGroups[0].StepBudget != 6 ||
		got.DuplicateKeyGroups[0].ByLane["guardrsi"] != 2 ||
		len(got.DuplicateKeyGroups[0].Issues) != 2 ||
		got.DuplicateKeyGroups[0].Issues[0] != 1450 ||
		got.DuplicateKeyGroups[0].Issues[1] != 1453 {
		t.Fatalf("duplicate groups = %+v, want duplicated task handoff marker with issue numbers", got.DuplicateKeyGroups)
	}
	if len(got.Reviews) != 2 || got.Reviews[0].Key != "task_push_next/issue-sync" || got.Reviews[0].IssueNumber != 1450 {
		t.Fatalf("reviews = %+v, want marker key and issue numbers", got.Reviews)
	}
}

func TestIssueContractSummarizesMixedIssueAuditCounts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuepolicy.IssueDraft{
		{
			Number: 1450,
			Title:  "guardrsi: require block reasons",
			Body:   completeIssueDraftBody(),
			Labels: []issuepolicy.IssueLabel{{Name: "guardrsi"}},
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
			Labels: []issuepolicy.IssueLabel{{Name: "guardrsi"}},
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
	code := RunIssue(&out, &errb, []string{"contract", "--from-issues", path, "--json"})
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
		AssumptionGroups []struct {
			Key              string         `json:"key"`
			Count            int            `json:"count"`
			StepBudget       int            `json:"step_budget"`
			ChildIssueBudget int            `json:"child_issue_budget"`
			ByLane           map[string]int `json:"by_lane"`
			ByReason         map[string]int `json:"by_reason"`
			ExampleKeys      []string       `json:"example_keys"`
		} `json:"assumption_groups"`
		ConfusionGroups []struct {
			Key              string         `json:"key"`
			Count            int            `json:"count"`
			StepBudget       int            `json:"step_budget"`
			ChildIssueBudget int            `json:"child_issue_budget"`
			ByLane           map[string]int `json:"by_lane"`
			ByReason         map[string]int `json:"by_reason"`
			ExampleKeys      []string       `json:"example_keys"`
		} `json:"confusion_groups"`
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
	if got.Counts.ByReason[issuepolicy.ReasonScopeIncomplete] != 1 ||
		got.Counts.ByReason[issuepolicy.ReasonUnrouted] != 1 ||
		got.Counts.ByReason[issuepolicy.ReasonOversizedSteps] != 1 {
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
	if len(got.AssumptionGroups) != 1 ||
		got.AssumptionGroups[0].Count != 2 ||
		got.AssumptionGroups[0].StepBudget != 15 ||
		got.AssumptionGroups[0].ChildIssueBudget != 2 ||
		got.AssumptionGroups[0].ByLane["guardrsi"] != 2 ||
		got.AssumptionGroups[0].ByReason[issuepolicy.ReasonOversizedSteps] != 1 ||
		len(got.AssumptionGroups[0].ExampleKeys) != 2 ||
		!strings.Contains(got.AssumptionGroups[0].Key, "guard journal fixture") {
		t.Fatalf("assumption groups = %+v, want shared guardrsi assumption group with split budget", got.AssumptionGroups)
	}
	if len(got.ConfusionGroups) != 1 ||
		got.ConfusionGroups[0].Count != 2 ||
		got.ConfusionGroups[0].StepBudget != 15 ||
		got.ConfusionGroups[0].ChildIssueBudget != 2 ||
		got.ConfusionGroups[0].ByLane["guardrsi"] != 2 ||
		got.ConfusionGroups[0].ByReason[issuepolicy.ReasonOversizedSteps] != 1 ||
		len(got.ConfusionGroups[0].ExampleKeys) != 2 ||
		!strings.Contains(got.ConfusionGroups[0].Key, "threshold tuning") {
		t.Fatalf("confusion groups = %+v, want shared guardrsi confusion group with split budget", got.ConfusionGroups)
	}
	if len(got.CoordinationGroups) != 1 ||
		got.CoordinationGroups[0].Count != 2 ||
		got.CoordinationGroups[0].StepBudget != 15 ||
		got.CoordinationGroups[0].ChildIssueBudget != 2 ||
		got.CoordinationGroups[0].ByLane["guardrsi"] != 2 ||
		got.CoordinationGroups[0].ByReason[issuepolicy.ReasonOversizedSteps] != 1 ||
		len(got.CoordinationGroups[0].ExampleKeys) != 2 ||
		!strings.Contains(got.CoordinationGroups[0].Key, "Avoid concurrent edits") {
		t.Fatalf("coordination groups = %+v, want shared guardrsi coordination group with split budget", got.CoordinationGroups)
	}
	assertRepairQueue(t, got.RepairQueues, "dispatch", 1, 3, nil)
	assertRepairQueue(t, got.RepairQueues, "split", 1, 12, map[string]int{issuepolicy.ReasonOversizedSteps: 1}, 2)
	assertRepairQueue(t, got.RepairQueues, "scope", 1, 1, map[string]int{issuepolicy.ReasonScopeIncomplete: 1})
	assertRepairQueue(t, got.RepairQueues, "route", 1, 1, map[string]int{issuepolicy.ReasonUnrouted: 1})
	scopeQueue := repairQueueByKind(got.RepairQueues, "scope")
	if scopeQueue.MissingFields["parent_ref"] != 1 || scopeQueue.MissingFields["done_condition"] != 1 {
		t.Fatalf("scope missing fields = %+v, want parent_ref and done_condition", scopeQueue.MissingFields)
	}
}

func TestIssueContractFromIssuesRefusesVagueRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuepolicy.IssueDraft{{Number: 1451, Title: "make it better", Body: "### Current state\nExists.\n"}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := RunIssue(&out, &errb, []string{"contract", "--from-issues", path})
	if code != 3 {
		t.Fatalf("exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	for _, want := range []string{"issue/1451", issuepolicy.ReasonScopeIncomplete, issuepolicy.ReasonUnrouted} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("rendered review missing %q:\n%s", want, out.String())
		}
	}
}

func completeIssueDraftBody() string {
	return completeIssueDraftBodyWithSteps("3")
}

func corruptGenerationIssueBody() string {
	return strings.Join([]string{
		"## Generation stream",
		"- Generation: $(@{gen=second-next; title=...; labels=...; why=...; scope=...}.gen)",
		"- Milestone: $(System.Collections.Hashtable.title)",
		"- Parent: #1625",
		"- Source: $source, Phase 2",
		"",
		"## Why",
		"The generated body below the corrupt header is intact.",
		"",
		"## Initial scope",
		"Repair only the generated metadata header.",
		"",
		"## Witness",
		"Captured dry-run output lists affected issue, marker, and replacement header.",
	}, "\n")
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

func hasString(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

func TestIssueContractStrictWitnessFlagHoldsForgeableCandidate(t *testing.T) {
	c := completeIssueCandidate()
	c.Witness = "agent reports that it completed the task"
	path := writeIssueContractJSON(t, c)
	var out, errb bytes.Buffer
	code := RunIssue(&out, &errb, []string{"contract", "--file", path, "--strict-witness", "--json"})
	if code != 3 {
		t.Fatalf("exit = %d, want 3\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
	}
	var got issueContractResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Reviews) != 1 || got.Reviews[0].WitnessGrade.Grade != issuepolicy.WitnessGradeForgeable {
		t.Fatalf("review = %+v", got.Reviews)
	}
	if !containsIssueString(got.Reviews[0].Reasons, issuepolicy.ReasonWitnessForgeable) {
		t.Fatalf("reasons = %+v", got.Reviews[0].Reasons)
	}
}

func completeIssueCandidate() issuepolicy.Candidate {
	return issuepolicy.Candidate{
		Schema:          issuepolicy.Schema,
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

func writeIssueContractJSON(t *testing.T, c issuepolicy.Candidate) string {
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

func TestRenderIssueContractClosureGate(t *testing.T) {
	review := issuepolicy.ReviewCandidate(issuepolicy.Candidate{
		Schema: issuepolicy.Schema, Key: "closure/4641", Title: "demo closure", ParentRef: "#4636",
		CurrentState: "toy passes", WhyNow: "closure requested", WorkingSpine: "contract -> close gate",
		WorkUnit: "leaf", ExpectedSteps: 3, InScope: "closure verification", OutOfScope: "load generator",
		DoneCondition: "demo may close", Witness: "go test ./internal/issuecontract passes", AcceptanceGate: "go test ./internal/issuecontract",
		Lane: "issuecontract", Paths: []string{"internal/issuecontract/**"}, ClosureBinding: "commit cites #4641",
		WorkEstimate: "Estimate: 3 points", ScopeContribution: "Contribution: 3/34 points", CompletionStandard: "demo",
		ClosureClaim: "complete", ClosureWitnessStandard: "demo",
	}, issuepolicy.Options{})
	got := renderIssueContract(issueContractResult{Schema: "test", Reviews: []issuepolicy.Review{review}})
	for _, want := range []string{"closure=refused", "claim=production", "production_credit=false", "closure_refuses: ISSUE_CLOSURE_WITNESS_MISMATCH", "closure_repair:"} {
		if !strings.Contains(got, want) {
			t.Fatalf("render missing %q:\n%s", want, got)
		}
	}
}

func TestRenderIssueContractProblemFrame(t *testing.T) {
	frame := issuepolicy.AssessProblemFrame(issuepolicy.IssueDraft{Body: "## Value\n- Centrality: Enabling\n- P1: advanced\n"})
	got := renderIssueContract(issueContractResult{Schema: "test", Reviews: []issuepolicy.Review{{
		Key: "issuepolicy/problem-frame", Verdict: "needs_problem_frame", Dispatchability: issuepolicy.TriageOnly,
		ProblemFrame: frame,
	}}})
	for _, want := range []string{
		"centrality=enabling",
		"problem_frame: problem_centrality_target_missing",
		"problem_frame: problem_check_p1_ceremonial",
		"problem_frame_repair: centrality: name the Core outcome",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("render missing %q:\n%s", want, got)
		}
	}
}

func containsIssueString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
