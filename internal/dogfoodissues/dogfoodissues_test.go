package dogfoodissues

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
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

func TestReportFreshnessForFileDetectsFreshAndStale(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.json")
	if err := os.WriteFile(path, []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	mtime := now.Add(-2 * time.Hour)
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatal(err)
	}
	fresh, err := ReportFreshnessForFile(path, now, 3*time.Hour, false)
	if err != nil {
		t.Fatal(err)
	}
	if fresh.Stale || fresh.AgeSeconds != 7200 || fresh.MaxAgeSeconds != 10800 {
		t.Fatalf("freshness = %+v, want fresh 2h under 3h", fresh)
	}
	stale, err := ReportFreshnessForFile(path, now, time.Hour, true)
	if err != nil {
		t.Fatal(err)
	}
	if !stale.Stale || !stale.StaleAllowed || stale.Age != "2h" || stale.MaxAge != "1h" {
		t.Fatalf("stale freshness = %+v, want allowed stale 2h over 1h", stale)
	}
	if stale.Timestamp != "2026-07-01T10:00:00Z" {
		t.Fatalf("timestamp = %q, want RFC3339 mtime", stale.Timestamp)
	}
}

func TestRenderReportsReportFreshnessAndStaleWarning(t *testing.T) {
	freshness := ReportFreshness{
		Timestamp:     "2026-07-01T10:00:00Z",
		Source:        "mtime",
		Age:           "2h",
		MaxAge:        "1h",
		AgeSeconds:    7200,
		MaxAgeSeconds: 3600,
		Stale:         true,
	}
	body := Render(Result{
		Schema:          Schema,
		Mode:            "dry-run",
		Report:          "report.json",
		ReportFreshness: &freshness,
	})
	for _, want := range []string{"report timestamp: 2026-07-01T10:00:00Z", "report age: 2h  max=1h  stale=yes", "STALE report:"} {
		if !strings.Contains(body, want) {
			t.Fatalf("render missing %q:\n%s", want, body)
		}
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
	createdBody := ""
	rows := Sync(plan, "owner/repo", []string{"guardrsi", "backlog"}, func(args []string) (string, string, bool) {
		calls = append(calls, args)
		if len(args) >= 2 && args[0] == "issue" && args[1] == "view" {
			return dogfoodIssueViewJSON(2, createdBody), "", true
		}
		createdBody = dogfoodArgAfter(args, "--body")
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
	viewBodies := map[string]string{}
	runner := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		if len(args) >= 2 && args[0] == "issue" && args[1] == "view" {
			return dogfoodIssueViewJSON(toIntForTest(args[2]), viewBodies[args[2]]), "", true
		}
		if len(args) >= 3 && args[0] == "issue" && args[1] == "edit" {
			viewBodies[args[2]] = dogfoodArgAfter(args, "--body")
			return "", "", true
		}
		if len(args) >= 2 && args[0] == "issue" && args[1] == "create" {
			viewBodies["1"] = dogfoodArgAfter(args, "--body")
			return "https://example/issues/1", "", true
		}
		return "", "unexpected call", false
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

func TestSyncRecordsPartialProgressAndRerunUpdatesVerifiedCreate(t *testing.T) {
	first := scopedGuardActionItem()
	second := scopedGuardActionItem()
	second.Key = "guard-rsi-route/guard-journal:second_blank_reason"
	second.Title = "guardrsi: close second blank-reason hole"
	second.Finding = "guard-journal:second_blank_reason"
	plan := []PlanRow{planRow(first), planRow(second)}

	var calls [][]string
	runner := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		if len(args) >= 2 && args[0] == "issue" && args[1] == "view" {
			if args[2] != "42" {
				return "", "not found", false
			}
			return dogfoodIssueViewJSON(42, IssueBody(first)), "", true
		}
		body := dogfoodArgAfter(args, "--body")
		if strings.Contains(body, first.Key) {
			return "https://github.com/owner/repo/issues/42\n", "", true
		}
		return "", "network down after first create", false
	}
	rows := SyncWithOptions(plan, "owner/repo", []string{"backlog"}, runner, SyncOptions{Timeout: time.Second})
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2", len(rows))
	}
	if !rows[0].OK || !rows[0].Verified || rows[0].Number == nil || *rows[0].Number != 42 {
		t.Fatalf("first row should record verified partial progress: %+v", rows[0])
	}
	if rows[1].OK || !strings.Contains(rows[1].Stderr, "network down") {
		t.Fatalf("second row should record failure and continue: %+v", rows[1])
	}
	if len(calls) < 3 {
		t.Fatalf("expected create, verify, second create calls; got %+v", calls)
	}

	rerun := BuildPlan([]ActionItem{first, second}, []Issue{{
		Number: 42,
		State:  "OPEN",
		Body:   IssueBody(first),
	}})
	if rerun[0].Action != "update" || rerun[0].Number == nil || *rerun[0].Number != 42 {
		t.Fatalf("rerun should update the verified created issue instead of duplicating: %+v", rerun[0])
	}
	if rerun[1].Action != "create" {
		t.Fatalf("rerun should still create the failed row, got %+v", rerun[1])
	}
}

func TestSyncRejectsCreatedIssueWhenMarkerVerificationFails(t *testing.T) {
	item := scopedGuardActionItem()
	rows := SyncWithOptions([]PlanRow{planRow(item)}, "", nil, func(args []string) (string, string, bool) {
		if len(args) >= 2 && args[0] == "issue" && args[1] == "view" {
			return dogfoodIssueViewJSON(77, "body without marker"), "", true
		}
		return "https://github.com/owner/repo/issues/77", "", true
	}, SyncOptions{Timeout: time.Second})
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	if rows[0].OK || rows[0].Verified {
		t.Fatalf("unverified create should not be OK: %+v", rows[0])
	}
	if rows[0].Number == nil || *rows[0].Number != 77 {
		t.Fatalf("failed verification should still record created issue number: %+v", rows[0])
	}
	if !strings.Contains(rows[0].Stderr, "marker key") {
		t.Fatalf("verification failure should explain marker mismatch: %+v", rows[0])
	}
}

func TestSyncBoundsHungRunner(t *testing.T) {
	item := scopedGuardActionItem()
	start := time.Now()
	rows := SyncWithOptions([]PlanRow{planRow(item)}, "", nil, func(args []string) (string, string, bool) {
		select {}
	}, SyncOptions{Timeout: 10 * time.Millisecond})
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("hung runner was not bounded, elapsed=%s rows=%+v", elapsed, rows)
	}
	if len(rows) != 1 || rows[0].OK || !strings.Contains(rows[0].Stderr, "timed out") {
		t.Fatalf("timeout row = %+v, want failed timeout", rows)
	}
}

func dogfoodArgAfter(args []string, flag string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

func dogfoodIssueViewJSON(number int, body string) string {
	b, _ := json.Marshal(Issue{
		Number: number,
		Title:  "dogfood issue",
		Body:   body,
		State:  "OPEN",
		URL:    "https://github.com/owner/repo/issues/" + strconv.Itoa(number),
	})
	return string(b)
}

func toIntForTest(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}
