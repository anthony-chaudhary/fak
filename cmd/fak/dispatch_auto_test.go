package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchauto"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestDispatchTickExitCodeOnlyFailsActualLauncherFailures(t *testing.T) {
	for _, action := range []string{
		"spawned",
		"would_spawn",
		"refused",
		"no_lane",
		"no_issue",
		"self_modify_hold",
		"in_flight_duplicate",
		"collision_risk",
		"lane_busy",
		"lane_leased",
		"broker_denied",
	} {
		t.Run(action, func(t *testing.T) {
			if got := dispatchTickExitCode(map[string]any{"action": action, "ok": false}); got != 0 {
				t.Fatalf("dispatchTickExitCode(%q) = %d, want 0", action, got)
			}
		})
	}

	for _, action := range []string{"spawn_failed", "", "unexpected"} {
		t.Run(action, func(t *testing.T) {
			if got := dispatchTickExitCode(map[string]any{"action": action}); got != 1 {
				t.Fatalf("dispatchTickExitCode(%q) = %d, want 1", action, got)
			}
		})
	}
	if got := dispatchTickExitCode(nil); got != 1 {
		t.Fatalf("dispatchTickExitCode(nil) = %d, want 1", got)
	}
}

func TestDispatchAutoReadyWorkScopesLaneAndExclusions(t *testing.T) {
	router := dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 10, Lane: "docs"},
			{Number: 11, Lane: "gateway"},
			{Number: 12, Lane: "gateway"},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"docs":    {Count: 9},
			"gateway": {Count: 9},
		},
	}

	if got := dispatchAutoReadyWork(router, "", nil); got != 3 {
		t.Fatalf("unscoped ready work = %d, want 3", got)
	}
	if got := dispatchAutoReadyWork(router, "gateway", nil); got != 2 {
		t.Fatalf("gateway ready work = %d, want 2", got)
	}
	if got := dispatchAutoReadyWork(router, "", []string{"docs"}); got != 2 {
		t.Fatalf("exclude docs ready work = %d, want 2", got)
	}

	router.Issues = nil
	if got := dispatchAutoReadyWork(router, "docs", nil); got != 9 {
		t.Fatalf("lane-group fallback ready work = %d, want 9", got)
	}
}

func TestDispatchAutoProbeErrorsAreFatalAndRendered(t *testing.T) {
	rec := map[string]any{
		"ok":      false,
		"live":    true,
		"backend": "claude",
		"errors":  []string{"issue router probe failed: rate limit exceeded"},
	}
	plan := dispatchauto.Plan{Reason: "no wave: ready_work is 0"}
	out := renderDispatchAuto(rec, plan)
	if !strings.Contains(out, "error: issue router probe failed: rate limit exceeded") {
		t.Fatalf("renderDispatchAuto missing probe error:\n%s", out)
	}
	var stdout, stderr bytes.Buffer
	if code := writeDispatchAutoResult(&stdout, &stderr, rec, plan, true); code != 1 {
		t.Fatalf("writeDispatchAutoResult code = %d, want 1 for probe error", code)
	}
}

func TestDispatchAutoRouterFetchErrorIsProbeError(t *testing.T) {
	router := dispatchtick.RouterPayload{
		OK:      false,
		Verdict: "FETCH_ERROR",
		Finding: "fetch_error",
		Reason:  "GraphQL: API rate limit already exceeded",
	}
	if got := dispatchAutoRouterProbeError(router); got != router.Reason {
		t.Fatalf("dispatchAutoRouterProbeError = %q, want %q", got, router.Reason)
	}
	router = dispatchtick.RouterPayload{OK: false, Verdict: "ACTION", Finding: "high_unrouted"}
	if got := dispatchAutoRouterProbeError(router); got != "" {
		t.Fatalf("dispatchAutoRouterProbeError ACTION = %q, want benign empty", got)
	}
}

func TestDispatchAutoAccountCapacityUsesLivePlusFreeSlots(t *testing.T) {
	if got := dispatchAutoAccountCapacity(13, 7); got != 20 {
		t.Fatalf("capacity = %d, want 20 live+free slots", got)
	}
	if got := dispatchAutoAccountCapacity(0, 5); got != 5 {
		t.Fatalf("capacity with no live workers = %d, want 5 free slots", got)
	}
	if got := dispatchAutoAccountCapacity(9, -3); got != 9 {
		t.Fatalf("negative free slots capacity = %d, want live-only 9", got)
	}
}

func TestDispatchWavePrelaunchGateLaunchesReadySubset(t *testing.T) {
	plan := []dispatchWaveExecutionPlan{
		{Rank: 0, Target: dispatchLaunchTarget{ID: "docs#10"}},
		{Rank: 1, Target: dispatchLaunchTarget{ID: "cmd#11"}},
	}
	audit := []dispatchWaveExecutionAudit{
		{Rank: 0, Target: dispatchLaunchTarget{ID: "docs#10"}, OK: true, Action: "would_spawn", Verdict: "WOULD_SPAWN"},
		{Rank: 1, Target: dispatchLaunchTarget{ID: "cmd#11"}, OK: false, Action: "self_modify_hold", Verdict: "SELF_MODIFY_HOLD"},
	}

	gate := dispatchWavePrelaunchGateFromAudit("wave-test", audit)
	if !gate.OK || gate.Action != "LAUNCH_READY" || gate.ReadyCount != 1 || gate.RefusedCount != 1 {
		t.Fatalf("gate = %+v, want partial launch with one ready and one refused", gate)
	}
	ready := dispatchWaveReadyExecutionPlan(plan, audit)
	if len(ready) != 1 || ready[0].Rank != 0 || ready[0].Target.ID != "docs#10" {
		t.Fatalf("ready plan = %+v, want only rank 0 docs#10", ready)
	}
}

func TestDispatchWaveAllocationCountHonorsPreflightHeadroomAndSeats(t *testing.T) {
	pre := map[string]any{
		"headroom": 3,
		"seat":     map[string]any{"free": 5},
	}
	if got := dispatchWaveAllocationCount(20, pre); got != 3 {
		t.Fatalf("allocation count = %d, want headroom cap 3", got)
	}
	pre["headroom"] = 8
	pre["seat"] = map[string]any{"free": 2}
	if got := dispatchWaveAllocationCount(20, pre); got != 2 {
		t.Fatalf("allocation count = %d, want seat cap 2", got)
	}
	pre["headroom"] = -1
	if got := dispatchWaveAllocationCount(20, pre); got != 0 {
		t.Fatalf("allocation count = %d, want exhausted 0", got)
	}
	if got := dispatchWaveAllocationCount(4, map[string]any{}); got != 4 {
		t.Fatalf("allocation count without preflight terms = %d, want requested 4", got)
	}
}

func TestDispatchWavePrelaunchGateBacksOffOnAuditError(t *testing.T) {
	gate := dispatchWavePrelaunchGateFromAudit("wave-test", []dispatchWaveExecutionAudit{
		{Rank: 0, Target: dispatchLaunchTarget{ID: "docs#10"}, OK: true, Action: "would_spawn", Verdict: "WOULD_SPAWN"},
		{Rank: 1, Target: dispatchLaunchTarget{ID: "gateway#11"}, Error: "router exploded"},
	})
	if gate.OK || gate.Action != "HOLD_ERROR" || gate.ErrorCount != 1 {
		t.Fatalf("gate = %+v, want HOLD_ERROR", gate)
	}
}

func TestDispatchWaveDryRunRepricesBenignPrelaunchHold(t *testing.T) {
	rows := []dispatchWaveExecutionAudit{{
		Action: "lane_leased",
		Target: dispatchLaunchTarget{Issue: 6046},
	}}
	got := dispatchWavePrelaunchRetryIssues(rows, false, 0, 8)
	if len(got) != 1 || got[0] != 6046 {
		t.Fatalf("retry issues = %v, want [6046] for dry-run planning", got)
	}
	if got := dispatchWavePrelaunchRetryIssues(rows, false, 8, 8); len(got) != 0 {
		t.Fatalf("retry issues at limit = %v, want none", got)
	}
}

func TestDispatchWaveRetryableAuditIssuesOnlyBenignHolds(t *testing.T) {
	rows := []dispatchWaveExecutionAudit{
		{Rank: 0, Target: dispatchLaunchTarget{ID: "gateway#10", Issue: 10}, OK: false, Action: "self_modify_hold"},
		{Rank: 1, Target: dispatchLaunchTarget{ID: "trajctl#11", Issue: 11}, OK: false, Action: "lane_busy"},
	}
	if got := dispatchWaveRetryableAuditIssues(rows); len(got) != 2 || got[0] != 10 || got[1] != 11 {
		t.Fatalf("retryable issues = %+v, want [10 11]", got)
	}
	if got := dispatchWaveRetryableAuditIssues([]dispatchWaveExecutionAudit{
		{Rank: 0, Target: dispatchLaunchTarget{ID: "gateway#10", Issue: 10}, OK: false, Action: "spawn_failed"},
	}); got != nil {
		t.Fatalf("spawn failure retryable issues = %+v, want nil", got)
	}
	if got := dispatchWaveRetryableAuditIssues([]dispatchWaveExecutionAudit{
		{Rank: 0, Target: dispatchLaunchTarget{ID: "gateway#10", Issue: 10}, Error: "audit failed"},
	}); got != nil {
		t.Fatalf("audit error retryable issues = %+v, want nil", got)
	}
}

func TestDispatchWavePriceFilteredSkipsHeldIssue(t *testing.T) {
	router := dispatchtick.RouterPayload{
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"gateway": {Count: 1, StepBudget: 9, Issues: []int{10}, Priority: map[int]int{10: dispatchtick.PriorityWeightDefault}},
			"docs":    {Count: 1, StepBudget: 1, Issues: []int{20}, Priority: map[int]int{20: dispatchtick.PriorityWeightDefault}},
		},
		Issues: []dispatchtick.IssueRoute{
			{Number: 10, Lane: "gateway", Paths: []string{"internal/gateway/"}, ExpectedSteps: 9},
			{Number: 20, Lane: "docs", Paths: []string{"docs/dispatch-loop.md"}, ExpectedSteps: 1},
		},
	}
	price, err := priceDispatchWavePayload(t.TempDir(), router, 1, 1, "", nil, 0, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if len(price.RunTargets) != 1 || price.RunTargets[0].Issue != 10 {
		t.Fatalf("unfiltered run targets = %+v, want issue 10 first", price.RunTargets)
	}
	price, err = priceDispatchWavePayloadFiltered(t.TempDir(), router, 1, 1, "", nil, 0, map[int]bool{10: true}, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if len(price.RunTargets) != 1 || price.RunTargets[0].Issue != 20 {
		t.Fatalf("filtered run targets = %+v, want issue 20 after holding issue 10", price.RunTargets)
	}
}

func TestDispatchWaveExitBenignForPrelaunchHold(t *testing.T) {
	rec := map[string]any{
		"ok": false,
		"prelaunch_gate": dispatchWavePrelaunchGate{
			Action:       "HOLD",
			RefusedCount: 1,
			Refused: []dispatchWavePrelaunchRefusal{
				{Target: "gateway#10", Action: "self_modify_hold", Verdict: "SELF_MODIFY_HOLD"},
			},
		},
	}
	if !dispatchWaveExitBenign(rec) {
		t.Fatalf("self-modify prelaunch hold should be a benign wave exit")
	}
	rec["prelaunch_gate"] = dispatchWavePrelaunchGate{Action: "HOLD_ERROR", ErrorCount: 1}
	if dispatchWaveExitBenign(rec) {
		t.Fatalf("audit error hold must not be a benign wave exit")
	}
}

func TestDispatchAutoCompatibleCapacityMatchesWaveTierPolicy(t *testing.T) {
	serve := true
	rows := []dispatchtick.AccountRow{
		{Account: "opencode-pm-a", Tag: "pm-a", Kind: "worker", Product: "opencode", ModelTier: 2, Available: true, CanServe: &serve},
		{Account: "opencode-pm-b", Tag: "pm-b", Kind: "worker", Product: "opencode", ModelTier: 2, Available: true, CanServe: &serve},
	}
	pool := dispatchtick.BuildSeatPool(rows, nil, "opencode")
	if pool.FreeSeats != 2 {
		t.Fatalf("aggregate free seats = %d, want 2", pool.FreeSeats)
	}

	engineering := dispatchtick.AllocateWave(dispatchtick.AccountWaveInput{
		Rows: rows, Count: pool.FreeSeats, WorkKind: "engineering", Product: "opencode",
	})
	if engineering.Granted != 0 || engineering.TargetTier != 1 {
		t.Fatalf("engineering allocation = %+v, want zero tier-1-compatible seats", engineering)
	}
	plan := dispatchauto.PlanAuto(dispatchauto.Input{EffectiveCap: 2, DistinctPools: engineering.Granted, ReadyWork: 10})
	if plan.Refill != 0 || plan.Binding != "distinct_pools" {
		t.Fatalf("engineering auto plan = %+v, want truthful zero refill bound by compatible pools", plan)
	}

	gardening := dispatchtick.AllocateWave(dispatchtick.AccountWaveInput{
		Rows: rows, Count: pool.FreeSeats, WorkKind: "gardening", Product: "opencode",
	})
	if gardening.Granted != 2 || gardening.TargetTier != 2 {
		t.Fatalf("gardening allocation = %+v, want both tier-2 seats", gardening)
	}
}
