package agentquery

import (
	"testing"
	"time"
)

func TestApplyListPlanFiltersAndNullOrdering(t *testing.T) {
	lane, host := "cmd", "h1"
	a, b := int64(10), int64(20)
	rows := []Row{{AgentID: "b", State: "LIVE", Liveness: "LIVE", Lane: &lane, Host: &host, ElapsedMS: &b}, {AgentID: "a", State: "LIVE", Liveness: "LIVE", Lane: &lane, Host: &host, ElapsedMS: &a}, {AgentID: "null", State: "LIVE", Liveness: "LIVE", Lane: &lane, Host: &host}, {AgentID: "other", State: "CLOSED", Liveness: "CLOSED"}}
	p := ListPlan{Schema: ListPlanSchema, State: "live", Lane: "CMD", OrderBy: "elapsed_desc", Limit: 10}
	got, trunc, err := ApplyListPlan(rows, p, time.Now())
	if err != nil || trunc || len(got) != 3 || got[0].AgentID != "b" || got[1].AgentID != "a" || got[2].AgentID != "null" {
		t.Fatalf("got=%+v trunc=%v err=%v", got, trunc, err)
	}
}
func TestValidateListPlanRejectsUnknownAndUnbounded(t *testing.T) {
	for _, p := range []ListPlan{{OrderBy: "bogus", Limit: 1}, {OrderBy: "identity_asc", Limit: 0}} {
		if err := ValidateListPlan(p); err == nil {
			t.Fatalf("accepted %+v", p)
		}
	}
}

func TestApplyListPlanAllFiltersAndSorts(t *testing.T) {
	owner, host, lane, group, model, provider, root, parent := "alice", "h1", "cmd", "g", "m", "p", "r", "parent"
	start1, start2, end1, last1 := "2026-08-17T00:00:00Z", "2026-08-18T00:00:00Z", "2026-08-18T01:00:00Z", "2026-08-18T00:30:00Z"
	cost1, cost2 := 1.0, 2.0
	rows := []Row{{AgentID: "match", State: "CLOSED", Liveness: "CLOSED", Owner: &owner, Host: &host, Lane: &lane, Group: &group, Model: &model, Provider: &provider, RootID: &root, ParentID: &parent, StartedAt: &start1, EndedAt: &end1, LastProgressAt: &last1, Cost: &cost1}, {AgentID: "other", State: "LIVE", Liveness: "LIVE", StartedAt: &start2, Cost: &cost2}}
	after := "2026-08-16T00:00:00Z"
	before := "2026-08-17T00:00:00Z"
	p := ListPlan{Schema: ListPlanSchema, State: "closed", Liveness: "closed", Owner: "alice", Host: "h1", Lane: "cmd", Group: "g", Model: "m", Provider: "p", RootID: "r", ParentID: "parent", StartedAfter: &after, StartedBefore: &before, OrderBy: "cost_desc", Limit: 10}
	got, _, err := ApplyListPlan(rows, p, time.Date(2026, 8, 18, 2, 0, 0, 0, time.UTC))
	if err != nil || len(got) != 1 || got[0].AgentID != "match" {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	for _, order := range []string{"elapsed_desc", "elapsed_asc", "progress_age_desc", "progress_age_asc", "started_desc", "started_asc", "ended_desc", "ended_asc", "cost_desc", "cost_asc", "identity_asc", "identity_desc"} {
		p = ListPlan{Schema: ListPlanSchema, OrderBy: order, Limit: 10}
		if _, _, err := ApplyListPlan(rows, p, time.Date(2026, 8, 18, 2, 0, 0, 0, time.UTC)); err != nil {
			t.Errorf("order=%s err=%v", order, err)
		}
	}
}
