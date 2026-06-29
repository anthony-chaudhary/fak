package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/loopindex"
)

func TestLoopIndexIssue1154PricedFanoutDefaultFeedsS0(t *testing.T) {
	rep := loopindex.Score(collectLoopIndex(repoRoot()))
	plan := rep.StageDetail[1]
	if plan.Name != loopindex.StagePlan {
		t.Fatalf("stage[1] = %q, want plan", plan.Name)
	}

	probes := map[string]bool{}
	for _, p := range plan.Probes {
		probes[p.Name] = p.Pass
	}
	for _, name := range []string{
		"arbitrate_substrate",
		"priced_fanout_default",
		"lane_trees_declared",
	} {
		if !probes[name] {
			t.Fatalf("plan probe %s = false; issue #1154 is not witnessed by the tree", name)
		}
	}

	var planKPI loopindex.KPI
	for _, k := range rep.KPIs {
		if k.Name == loopindex.StagePlan {
			planKPI = k
			break
		}
	}
	if !planKPI.Wired || planKPI.Debt != 0 {
		t.Fatalf("plan KPI = %+v, want wired with no debt for #1154", planKPI)
	}
}

func TestDispatchOrderIssue1154RefusesCollisionBeforeLaunch(t *testing.T) {
	res := dispatchorder.Plan(dispatchorder.Input{
		NowUnix: 10_000,
		Candidates: []dispatchorder.Candidate{
			{ID: "safe-docs", Key: "docs", Lane: "docs", Tree: []string{"docs/**"}, UpdatedUnix: 9_800},
			{ID: "hot-a", Key: "a", Lane: "gateway", Tree: []string{"internal/gateway/**"}, UpdatedUnix: 9_700},
			{ID: "hot-b", Key: "b", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}, UpdatedUnix: 9_900},
		},
	})
	if res.CollisionsAvoided != 1 || res.LanesUtilized != 2 || res.SerializationWasted != 1 {
		t.Fatalf("S0 counts avoided=%d lanes=%d wasted=%d, want 1/2/1",
			res.CollisionsAvoided, res.LanesUtilized, res.SerializationWasted)
	}
	if res.Pick() != "hot-b" {
		t.Fatalf("pick = %q, want hot-b", res.Pick())
	}
	for _, row := range res.Order {
		if row.ID == "hot-a" && (row.Disposition != dispatchorder.DispCollisionRisk || row.Reason != dispatchorder.ReasonCollisionRisk) {
			t.Fatalf("hot-a = %+v, want pre-launch COLLISION_RISK serialization", row)
		}
	}
}
