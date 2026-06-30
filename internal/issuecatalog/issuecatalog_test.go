package issuecatalog

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

func completeRow() Row {
	return Row{
		Key:            "perf/kv-cache/cross-agent-reuse-default-on",
		Title:          "default-on cross-agent KV prefix reuse for the dispatch fleet",
		ParentRef:      "Milestone #2 KV cache 2x; child of #1463",
		CurrentState:   "internal/cachemeta ships the tier ladder but cross-agent reuse is opt-in.",
		WhyNow:         "The realized reuse ratio is the headline KV value and it is left off by default.",
		WorkingSpine:   "internal/cachemeta placement policy + internal/gateway L3 share scope.",
		WorkUnit:       "leaf",
		ExpectedSteps:  5,
		Assumptions:    []string{"The gateway can observe cross-agent reuse in the same fleet run."},
		ConfusionRisks: []string{"Do not fold eviction-policy redesign into the default flip."},
		Coordination:   []string{"Coordinate with gateway L3 share-scope changes before live dispatch."},
		Trigger:        "A performance-enablement catalog row names a default-off cache-value gap.",
		BatchPolicy:    "Catalog sync creates in capped waves; reruns update by marker key.",
		InScope:        "Flip the default and gate it behind a measured floor.",
		OutOfScope:     "The eviction policy rewrite (its own row).",
		DoneCondition:  "A fresh fleet run reuses prefixes across agents with no flag set.",
		Witness:        "go test ./internal/cachevalueledger -run TestScoreLedger; fak nightrun score --json reuse>=0.5.",
		AcceptanceGate: "make ci; go test ./... green.",
		ClosureBinding: "Resolving commit must cite #N and carry a matching (fak <leaf>) trailer.",
		Lane:           "cachemeta",
		Paths:          []string{"internal/cachemeta/pool.go", "internal/gateway/gateway.go"},
		Labels:         []string{"performance", "track/B-performance", "priority/P1"},
		Milestone:      "The KV cache value is owned, observed & 2x",
		Lens:           "default-off",
		Priority:       "P1",
	}
}

// TestIssueBodyRoundTripsToDispatchable is the load-bearing guarantee: a complete
// catalog row renders to a body whose headings re-parse, through the SAME contract
// used to audit open issues, back to a dispatchable verdict. If this breaks, every
// generated issue would re-audit as triage debt.
func TestIssueBodyRoundTripsToDispatchable(t *testing.T) {
	row := completeRow()
	body := IssueBody(row)

	draft := issuecontract.IssueDraft{Number: 42, Title: row.Title, Body: body}
	rev := issuecontract.ReviewIssueDraft(draft, issuecontract.Options{})
	if !rev.OK {
		t.Fatalf("round-tripped body should be dispatchable, got verdict=%q reasons=%v missing=%v",
			rev.Verdict, rev.Reasons, rev.MissingFields)
	}
	if rev.Score.Total != 100 {
		t.Errorf("complete row should score 100, got %d (%+v)", rev.Score.Total, rev.Score)
	}
	if rev.AgentContext.Total != 100 {
		t.Errorf("complete row should carry full agent context, got %+v", rev.AgentContext)
	}
}

func TestReviewRejectsIncompleteRow(t *testing.T) {
	row := completeRow()
	row.Witness = ""
	rev := Review(row, Options{})
	if rev.OK {
		t.Fatalf("row with no witness must not be OK")
	}
	if !containsStr(rev.MissingFields, "witness") {
		t.Errorf("missing fields should name witness, got %v", rev.MissingFields)
	}
}

func TestReviewRejectsUnrouted(t *testing.T) {
	row := completeRow()
	row.Lane = ""
	row.Paths = nil
	rev := Review(row, Options{})
	if rev.OK {
		t.Fatalf("row with no lane and no paths must not be OK")
	}
	if !containsStr(rev.Reasons, issuecontract.ReasonUnrouted) {
		t.Errorf("reasons should include %s, got %v", issuecontract.ReasonUnrouted, rev.Reasons)
	}
}

func TestMarkerKeyRoundTrip(t *testing.T) {
	row := completeRow()
	body := IssueBody(row)
	if got := MarkerKey(body); got != row.Key {
		t.Errorf("MarkerKey = %q, want %q", got, row.Key)
	}
	if !strings.HasPrefix(body, "<!-- "+MarkerName+":") {
		t.Errorf("body must start with the marker, got %q", body[:40])
	}
	if MarkerKey("# no marker here") != "" {
		t.Errorf("MarkerKey on a body without a marker must be empty")
	}
}

func TestBuildPlanCreateVsUpdate(t *testing.T) {
	row := completeRow()
	existing := []Issue{{
		Number: 777,
		Title:  "stale title",
		Body:   "<!-- " + MarkerName + ": " + row.Key + " -->\nold body",
		State:  "open",
	}}
	plan, skipped := BuildPlan([]Row{row}, existing, Options{})
	if len(skipped) != 0 {
		t.Fatalf("complete row should not be skipped, got %v", skipped)
	}
	if len(plan) != 1 || plan[0].Action != "update" || plan[0].Number == nil || *plan[0].Number != 777 {
		t.Fatalf("expected update of #777, got %+v", plan)
	}

	plan, _ = BuildPlan([]Row{row}, nil, Options{})
	if len(plan) != 1 || plan[0].Action != "create" || plan[0].Number != nil {
		t.Fatalf("expected create, got %+v", plan)
	}
	if plan[0].Milestone != row.Milestone {
		t.Errorf("plan should carry milestone, got %q", plan[0].Milestone)
	}
}

func TestBuildPlanSkipsIncompleteAndDuplicate(t *testing.T) {
	good := completeRow()
	bad := completeRow()
	bad.Key = "perf/kv-cache/incomplete"
	bad.DoneCondition = ""
	dup := completeRow() // same key as good
	dup.Title = "duplicate key row"

	plan, skipped := BuildPlan([]Row{good, bad, dup}, nil, Options{})
	if len(plan) != 1 {
		t.Fatalf("only the good row should plan, got %d", len(plan))
	}
	var sawIncomplete, sawDup bool
	for _, s := range skipped {
		if s.Key == bad.Key {
			sawIncomplete = true
		}
		if s.Key == dup.Key && s.Reason == "DUPLICATE_KEY_IN_CATALOG" {
			sawDup = true
		}
	}
	if !sawIncomplete || !sawDup {
		t.Fatalf("expected incomplete + duplicate skips, got %+v", skipped)
	}
}

func TestSyncCreateAndUpdateArgs(t *testing.T) {
	row := completeRow()
	plan, _ := BuildPlan([]Row{row}, nil, Options{})

	var captured [][]string
	runner := func(args []string) (string, string, bool) {
		captured = append(captured, args)
		return "https://github.com/x/y/issues/1", "", true
	}
	res := Sync(plan, "owner/repo", runner)
	if len(res) != 1 || !res[0].OK {
		t.Fatalf("sync should report ok, got %+v", res)
	}
	if len(captured) != 1 {
		t.Fatalf("expected one gh call, got %d", len(captured))
	}
	args := strings.Join(captured[0], " ")
	for _, want := range []string{"issue create", "--body-file", "--label performance",
		"--milestone The KV cache value is owned, observed & 2x", "--repo owner/repo"} {
		if !strings.Contains(args, want) {
			t.Errorf("create args missing %q in: %s", want, args)
		}
	}

	// Update path uses `issue edit <n>` + --add-label.
	n := 5
	updPlan := []PlanRow{{Action: "update", Number: &n, Key: row.Key, Title: row.Title,
		Body: IssueBody(row), Milestone: row.Milestone, Labels: row.Labels}}
	captured = nil
	Sync(updPlan, "", runner)
	args = strings.Join(captured[0], " ")
	if !strings.Contains(args, "issue edit 5") || !strings.Contains(args, "--add-label performance") {
		t.Errorf("update args wrong: %s", args)
	}
}

func TestParseCatalogAndDefaultRows(t *testing.T) {
	rows, err := ParseCatalog([]byte(`[{"key":"a/b","title":"do the thing"}]`))
	if err != nil || len(rows) != 1 || rows[0].Key != "a/b" {
		t.Fatalf("ParseCatalog round-trip failed: %v %+v", err, rows)
	}
	// Unknown fields (expected_steps) decode away harmlessly.
	rows, err = ParseCatalog([]byte(`[{"key":"a/b","title":"t","expected_steps":3}]`))
	if err != nil || len(rows) != 1 {
		t.Fatalf("unknown field should not break parse: %v", err)
	}
	// The embedded default catalog is always valid JSON (may be empty during bring-up).
	if _, err := DefaultRows(); err != nil {
		t.Fatalf("embedded default catalog must parse: %v", err)
	}
}

func TestResultJSONShape(t *testing.T) {
	r := Result{Schema: Schema, Mode: "dry-run", Catalog: "x", Total: 1,
		Planned: []PlanRow{{Action: "create", Key: "a/b", Title: "t"}}}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"schema":"`+Schema+`"`) {
		t.Errorf("result json missing schema: %s", b)
	}
}

func containsStr(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
