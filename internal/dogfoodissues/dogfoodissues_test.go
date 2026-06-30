package dogfoodissues

import (
	"encoding/json"
	"strings"
	"testing"
)

// fixtureReport mirrors the Python test's fixture_report(): a recent-feature
// dogfood report with a code-slop ACTION probe and a dogfood-coverage probe.
func fixtureReport(t *testing.T) map[string]any {
	t.Helper()
	const raw = `{
		"schema": "fak.recent-feature-dogfood.v1",
		"ok": true,
		"out_dir": ".fak/recent-feature-dogfood/fixture",
		"probes": [
			{
				"key": "code-slop-scorecard",
				"ok": true,
				"payload": {
					"schema": "fleet-code-slop-scorecard/1",
					"ok": false,
					"verdict": "ACTION",
					"finding": "code_slop",
					"corpus": {"score": 71.5, "grade": "C", "slop_debt": 12},
					"next_action": "retire slop-debt worst-first; re-run to prove the drop"
				}
			},
			{
				"key": "dogfood-coverage-scorecard",
				"ok": true,
				"payload": {
					"schema": "dogfood-coverage/1",
					"coverage": 88.9,
					"grade": "B",
					"dogfood_debt": 0,
					"audit_rows": 0,
					"worst_first": ["audit_journal_evidence"]
				}
			}
		]
	}`
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		t.Fatalf("fixture parse: %v", err)
	}
	return m
}

func byKey(items []ActionItem) map[string]ActionItem {
	out := map[string]ActionItem{}
	for _, it := range items {
		out[it.Key] = it
	}
	return out
}

func TestExtractsCodeSlopAndDogfoodCoverageItems(t *testing.T) {
	items := ExtractActionItems(fixtureReport(t), "report.json")
	bk := byKey(items)
	slop, ok := bk["recent-feature-dogfood/code-slop-scorecard/code_slop"]
	if !ok {
		t.Fatalf("missing code-slop item; got keys %v", keysOf(bk))
	}
	cov, ok := bk["recent-feature-dogfood/dogfood-coverage-scorecard/dogfood_coverage"]
	if !ok {
		t.Fatalf("missing dogfood-coverage item; got keys %v", keysOf(bk))
	}
	if slop.DebtCount != 12 {
		t.Errorf("slop debt = %d, want 12", slop.DebtCount)
	}
	if slop.Score != "71.5" {
		t.Errorf("slop score = %q, want \"71.5\"", slop.Score)
	}
	if slop.Grade != "C" {
		t.Errorf("slop grade = %q, want \"C\"", slop.Grade)
	}
	if cov.DebtCount != 1 {
		t.Errorf("coverage debt = %d, want 1", cov.DebtCount)
	}
}

func keysOf(m map[string]ActionItem) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func TestPlanUpdatesExistingStableKeyInsteadOfDuplicate(t *testing.T) {
	items := ExtractActionItems(fixtureReport(t), "report.json")
	existing := []Issue{{
		Number: 123,
		State:  "OPEN",
		Title:  "old",
		Body:   "<!-- fak-dogfood-action-key: recent-feature-dogfood/code-slop-scorecard/code_slop -->",
	}}
	plan := BuildPlan(items, existing)
	bk := map[string]PlanRow{}
	for _, row := range plan {
		bk[row.Key] = row
	}
	slop := bk["recent-feature-dogfood/code-slop-scorecard/code_slop"]
	if slop.Action != "update" {
		t.Errorf("slop action = %q, want update", slop.Action)
	}
	if slop.Number == nil || *slop.Number != 123 {
		t.Errorf("slop number = %v, want 123", slop.Number)
	}
	cov := bk["recent-feature-dogfood/dogfood-coverage-scorecard/dogfood_coverage"]
	if cov.Action != "create" {
		t.Errorf("coverage action = %q, want create", cov.Action)
	}
}

func TestReviewedPlanSkipsAggregateRowsWithoutScope(t *testing.T) {
	items := ExtractActionItems(fixtureReport(t), "report.json")
	plan, skipped := BuildPlanWithOptions(items, nil, BuildOptions{})
	if len(plan) != 0 {
		t.Fatalf("reviewed plan len = %d, want 0 dispatchable aggregate rows", len(plan))
	}
	if len(skipped) != 2 {
		t.Fatalf("skipped len = %d, want 2; rows=%+v", len(skipped), skipped)
	}
	for _, row := range skipped {
		if row.Dispatchability != "triage_only" {
			t.Fatalf("skipped dispatchability = %q, want triage_only", row.Dispatchability)
		}
		if !strings.Contains(row.Reason, "ISSUE_SCOPE_INCOMPLETE") || !strings.Contains(row.Reason, "ISSUE_UNROUTED") {
			t.Fatalf("skip reason = %q, want scope+route reasons", row.Reason)
		}
	}
}

func TestReviewedPlanAcceptsScopedColonKeyItem(t *testing.T) {
	item := scopedGuardActionItem()
	existing := []Issue{{
		Number: 44,
		State:  "OPEN",
		Body:   "<!-- fak-dogfood-action-key: guard-rsi-route/guard-journal:blank_reason_on_deny -->",
	}}
	plan, skipped := BuildPlanWithOptions([]ActionItem{item}, existing, BuildOptions{
		Live:          true,
		DedupeChecked: true,
		DedupeCap:     300,
	})
	if len(skipped) != 0 {
		t.Fatalf("skipped = %+v, want none", skipped)
	}
	if len(plan) != 1 {
		t.Fatalf("plan len = %d, want 1", len(plan))
	}
	row := plan[0]
	if row.Action != "update" || row.Number == nil || *row.Number != 44 {
		t.Fatalf("row = %+v, want update #44", row)
	}
	if !row.Review.OK || row.Review.Dispatchability != "dispatchable" || row.Review.Score.Total != 100 {
		t.Fatalf("review = %+v, want dispatchable full score", row.Review)
	}
	if row.Lane != "guardrsi" || len(row.Paths) != 1 || row.Paths[0] != "internal/guardrsi/**" {
		t.Fatalf("route = lane %q paths %+v", row.Lane, row.Paths)
	}
	if len(row.Labels) != 1 || row.Labels[0] != "guardrsi" {
		t.Fatalf("labels = %+v, want guardrsi", row.Labels)
	}
	for _, want := range []string{"Working spine", "Priority context", "Work unit", "Expected steps", "Trigger", "Batch policy", "In scope", "Acceptance gate", "Path hints"} {
		if !strings.Contains(row.Body, want) {
			t.Fatalf("body missing %q\n---\n%s", want, row.Body)
		}
	}
}

func TestBodyContainsRequiredActionFields(t *testing.T) {
	items := ExtractActionItems(fixtureReport(t), "report.json")
	if len(items) == 0 {
		t.Fatal("no items extracted")
	}
	body := IssueBody(items[0])
	for _, want := range []string{
		"Stable key",
		"slop_score",
		"grade",
		"slop_debt",
		"Evidence path",
		"Suggested next action",
		"retire slop-debt",
		"Work unit",
		"Expected steps",
		"Trigger",
		"Batch policy",
		"Scorecard probe `code-slop-scorecard` emitted ACTION finding `code_slop`.",
		"One issue per stable dogfood action key; reruns update the existing marker",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("body missing %q\n---\n%s", want, body)
		}
	}
}

func TestMarkerKeyRoundTrips(t *testing.T) {
	items := ExtractActionItems(fixtureReport(t), "report.json")
	body := IssueBody(items[0])
	got := MarkerKey(body)
	if got != items[0].Key {
		t.Errorf("MarkerKey = %q, want %q", got, items[0].Key)
	}
	if MarkerKey("no marker here") != "" {
		t.Errorf("MarkerKey on bodiless = %q, want \"\"", MarkerKey("no marker here"))
	}
}

func TestRenderReportsSkippedContractRows(t *testing.T) {
	body := Render(Result{
		Schema: Schema,
		Mode:   "dry-run",
		Report: "report.json",
		Skipped: []SkippedRow{{
			Key:    "recent-feature-dogfood/code-slop-scorecard/code_slop",
			Reason: "ISSUE_SCOPE_INCOMPLETE,ISSUE_UNROUTED",
		}},
	})
	if !strings.Contains(body, "skipped-contract: 1 item(s)") {
		t.Fatalf("render missing skipped count:\n%s", body)
	}
	if !strings.Contains(body, "no dispatchable scorecard ACTION items found") {
		t.Fatalf("render missing no-dispatchable line:\n%s", body)
	}
}

func scopedGuardActionItem() ActionItem {
	return ActionItem{
		Key:            "guard-rsi-route/guard-journal:blank_reason_on_deny",
		Title:          "guardrsi: close guard-unexplained-block honesty hole",
		SourceProbe:    "guard-verdict-rsi",
		ScoreName:      "severity",
		Score:          "P1",
		Grade:          "P1",
		DebtName:       "guard_honesty_hole",
		DebtCount:      1,
		EvidencePath:   "guard-audit.jsonl",
		NextAction:     "require a closed-vocabulary reason on every block",
		Finding:        "guard-journal:blank_reason_on_deny",
		ParentRef:      "fak guard-verdict-rsi route",
		WorkingSpine:   "Make the specific guard journal honesty hole impossible before tuning thresholds.",
		InScope:        "Add a closed-vocabulary classification and a regression fixture for this cause key.",
		OutOfScope:     "Do not broaden the route queue or refactor unrelated guard journals.",
		DoneCondition:  "The regression fixture no longer routes this cause as an unexplained honesty hole.",
		Witness:        "go test ./internal/guardrsi ./internal/guardroute",
		AcceptanceGate: "go test ./internal/guardrsi ./internal/guardroute",
		Lane:           "guardrsi",
		Paths:          []string{"internal/guardrsi/**"},
		Labels:         []string{"guardrsi"},
		BoundaryNotes:  []string{"Public guard-journal defect only."},
	}
}

func TestSyncUsesPlanLabelsAndGlobalLabels(t *testing.T) {
	plan, skipped := BuildPlanWithOptions([]ActionItem{scopedGuardActionItem()}, nil, BuildOptions{})
	if len(skipped) != 0 || len(plan) != 1 {
		t.Fatalf("plan=%+v skipped=%+v, want one plan row", plan, skipped)
	}
	var calls [][]string
	rows := Sync(plan, "owner/repo", []string{"guardrsi", "backlog"}, func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "https://example/issues/2", "", true
	})
	if len(rows) != 1 || !rows[0].OK {
		t.Fatalf("sync rows = %+v, want one ok", rows)
	}
	var labels []string
	for i := 0; i < len(calls[0])-1; i++ {
		if calls[0][i] == "--label" {
			labels = append(labels, calls[0][i+1])
		}
	}
	want := []string{"guardrsi", "backlog"}
	if strings.Join(labels, ",") != strings.Join(want, ",") {
		t.Fatalf("labels = %+v, want %+v; args=%+v", labels, want, calls[0])
	}
}

// TestSyncUsesInjectedRunnerWithoutGh proves the effectful surface is testable
// without a real gh: the plan rows route to gh issue create/edit via the runner.
func TestSyncUsesInjectedRunnerWithoutGh(t *testing.T) {
	items := ExtractActionItems(fixtureReport(t), "report.json")
	existing := []Issue{{
		Number: 7,
		State:  "OPEN",
		Body:   "<!-- fak-dogfood-action-key: recent-feature-dogfood/code-slop-scorecard/code_slop -->",
	}}
	plan := BuildPlan(items, existing)
	var calls [][]string
	runner := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "https://example/issues/1", "", true
	}
	rows := Sync(plan, "owner/repo", []string{"backlog"}, runner)
	if len(rows) != len(plan) {
		t.Fatalf("got %d sync rows, want %d", len(rows), len(plan))
	}
	for _, r := range rows {
		if !r.OK {
			t.Errorf("sync row %s not ok", r.Key)
		}
	}
	var sawEdit, sawCreate, sawLabel bool
	for _, args := range calls {
		joined := strings.Join(args, " ")
		if strings.Contains(joined, "issue edit 7") {
			sawEdit = true
		}
		if strings.Contains(joined, "issue create") {
			sawCreate = true
			if strings.Contains(joined, "--label backlog") {
				sawLabel = true
			}
		}
		if !strings.Contains(joined, "--repo owner/repo") {
			t.Errorf("missing repo arg: %v", args)
		}
	}
	if !sawEdit {
		t.Error("no edit call for the existing-keyed item")
	}
	if !sawCreate {
		t.Error("no create call for the new item")
	}
	if !sawLabel {
		t.Error("label not passed on create")
	}
}
