package fleetmon

import "testing"

func TestParseRunPlanObjectShape(t *testing.T) {
	data := []byte(`{"schema":"fak-fleet-runplan/1","run_id":"r1","workers":[
		{"issue":1856,"session":"issue-1856","account":"acct-a","area":"fleet","pid":100},
		{"issue":1857,"session":"issue-1857","account":"acct-b"}
	]}`)
	plan, err := ParseRunPlan(data)
	if err != nil {
		t.Fatal(err)
	}
	if plan.RunID != "r1" || len(plan.Workers) != 2 {
		t.Fatalf("unexpected plan: %+v", plan)
	}
	if plan.Workers[0].PID != 100 {
		t.Errorf("pid should parse, got %d", plan.Workers[0].PID)
	}
}

func TestParseRunPlanBareArrayShape(t *testing.T) {
	data := []byte(`[{"issue":1,"session":"issue-1"},{"issue":2,"session":"issue-2"}]`)
	plan, err := ParseRunPlan(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Workers) != 2 {
		t.Fatalf("bare array should parse to 2 workers, got %d", len(plan.Workers))
	}
}

func TestParseRunPlanDropsEmptyRows(t *testing.T) {
	data := []byte(`[{"issue":1,"session":"issue-1"},{"issue":0,"session":""}]`)
	plan, _ := ParseRunPlan(data)
	if len(plan.Workers) != 1 {
		t.Fatalf("a row with no issue and no session should be dropped, got %d", len(plan.Workers))
	}
}

func TestParseRunPlanEmptyIsError(t *testing.T) {
	if _, err := ParseRunPlan([]byte("   ")); err == nil {
		t.Fatal("empty input should error")
	}
}

func TestRunPlanSessionsForIssueLinksReplacements(t *testing.T) {
	plan := RunPlan{Workers: []PlanWorker{
		{Issue: 1856, Session: "issue-1856"},
		{Issue: 1856, Session: "issue-1856-replacement-1", ReplacementOf: "issue-1856"},
		{Issue: 1857, Session: "issue-1857"},
	}}
	got := plan.SessionsForIssue(1856)
	if len(got) != 2 {
		t.Fatalf("issue 1856 should have 2 sessions (original + replacement), got %v", got)
	}
}

func TestRunPlanWorkerBySession(t *testing.T) {
	plan := RunPlan{Workers: []PlanWorker{{Issue: 1, Session: "a"}, {Issue: 2, Session: "b"}}}
	w, ok := plan.WorkerBySession("b")
	if !ok || w.Issue != 2 {
		t.Fatalf("lookup failed: %+v ok=%v", w, ok)
	}
	if _, ok := plan.WorkerBySession("missing"); ok {
		t.Fatal("missing session should return ok=false")
	}
}

func TestRunPlanIssues(t *testing.T) {
	plan := RunPlan{Workers: []PlanWorker{{Issue: 3, Session: "c"}, {Issue: 1, Session: "a"}, {Issue: 3, Session: "c2"}}}
	got := plan.Issues()
	if len(got) != 2 || got[0] != 1 || got[1] != 3 {
		t.Fatalf("issues should be sorted+deduped [1,3], got %v", got)
	}
}
