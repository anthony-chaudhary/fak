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
