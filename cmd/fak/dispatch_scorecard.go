package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

const dispatchPlanScorecardSchema = "fak-dispatch-plan-scorecard/1"

type dispatchPlanScorecard struct {
	Schema     string                     `json:"schema"`
	OK         bool                       `json:"ok"`
	Verdict    string                     `json:"verdict"`
	Finding    string                     `json:"finding"`
	Reason     string                     `json:"reason"`
	NextAction string                     `json:"next_action"`
	Corpus     dispatchPlanCorpus         `json:"corpus"`
	KPIs       []dispatchPlanProbe        `json:"kpis"`
	Live       *dispatchPlanLiveTelemetry `json:"live,omitempty"`
}

type dispatchPlanCorpus struct {
	Score            int    `json:"score"`
	Grade            string `json:"grade"`
	DispatchPlanDebt int    `json:"dispatch_plan_debt"`
	Probes           int    `json:"probes"`
	Passed           int    `json:"passed"`
}

type dispatchPlanProbe struct {
	Name   string         `json:"kpi"`
	Group  string         `json:"group"`
	Pass   bool           `json:"pass"`
	Score  int            `json:"score"`
	Debt   int            `json:"debt"`
	Detail string         `json:"detail"`
	Data   map[string]any `json:"data,omitempty"`
}

type dispatchPlanLiveTelemetry struct {
	Schema               string                            `json:"schema"`
	OK                   bool                              `json:"ok"`
	Verdict              string                            `json:"verdict"`
	Reason               string                            `json:"reason"`
	NextAction           string                            `json:"next_action"`
	Workspace            string                            `json:"workspace"`
	Requested            int                               `json:"requested"`
	RouterOK             bool                              `json:"router_ok"`
	RouterVerdict        string                            `json:"router_verdict"`
	Coverage             dispatchtick.RouterCoverage       `json:"coverage"`
	Counts               dispatchtick.RouterCounts         `json:"counts"`
	LaneCount            int                               `json:"lane_count"`
	Action               string                            `json:"action"`
	CandidateCount       int                               `json:"candidate_count"`
	ScopedCount          int                               `json:"scoped_count"`
	UnscopedCount        int                               `json:"unscoped_count"`
	ScopeCoveragePct     int                               `json:"scope_coverage_pct"`
	SafeConcurrency      int                               `json:"safe_concurrency"`
	SafeConcurrencyPct   int                               `json:"safe_concurrency_pct"`
	SameLaneParallelism  int                               `json:"same_lane_parallelism"`
	WaveCount            int                               `json:"wave_count"`
	Waves                []dispatchPriceWave               `json:"waves,omitempty"`
	LaunchPlan           []dispatchLaunchWave              `json:"launch_plan,omitempty"`
	LaneSerialWaveCount  int                               `json:"lane_serial_wave_count"`
	ScopedParallelGain   int                               `json:"scoped_parallelism_gain"`
	CollisionWavePenalty int                               `json:"collision_wave_penalty"`
	CollisionsAvoided    int                               `json:"collisions_avoided"`
	SerializationWasted  int                               `json:"serialization_wasted"`
	RepartitionCount     int                               `json:"repartition_count"`
	RunTargets           []dispatchWaveCandidate           `json:"run_targets,omitempty"`
	Repartition          []dispatchorder.RepartitionAdvice `json:"repartition,omitempty"`
}

func runDispatchScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root for opt-in live router telemetry")
	liveRouter := fs.Bool("live-router", false, "include non-gating telemetry from the current workspace issue router")
	count := fs.Int("count", 4, "live-router wave width to price")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *count < 1 {
		fmt.Fprintln(stderr, "fak dispatch scorecard: --count must be >= 1")
		return 2
	}
	rep := buildDispatchPlanScorecard()
	if *liveRouter {
		root, err := dispatchScorecardWorkspace(*workspace)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch scorecard: %v\n", err)
			return 1
		}
		live := buildDispatchPlanLiveTelemetry(root, *count)
		rep.Live = &live
		if rep.OK {
			rep.NextAction = live.NextAction
		}
	}
	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak dispatch scorecard: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderDispatchPlanScorecard(rep))
	}
	if rep.OK {
		return 0
	}
	return 1
}

func dispatchScorecardWorkspace(workspace string) (string, error) {
	root := strings.TrimSpace(workspace)
	if root != "" {
		return root, nil
	}
	wd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("getwd: %w", err)
	}
	return wd, nil
}

func buildDispatchPlanScorecard() dispatchPlanScorecard {
	root, err := os.MkdirTemp("", "fak-dispatch-plan-score-*")
	if err != nil {
		probe := dispatchScoreProbe("scorecard_temp_root", false, "could not create temp root for isolated lease probes: "+err.Error(), nil)
		return foldDispatchPlanScorecard([]dispatchPlanProbe{probe})
	}
	defer os.RemoveAll(root)
	if err := initDispatchScoreRoot(root); err != nil {
		probe := dispatchScoreProbe("scorecard_temp_root", false, "could not initialize isolated temp Git repo for lease probes: "+err.Error(), nil)
		return foldDispatchPlanScorecard([]dispatchPlanProbe{probe})
	}

	probes := []dispatchPlanProbe{
		dispatchScoreCollisionPrice(root),
		dispatchScorePriceWaveSchedule(),
		dispatchScoreScopedParallelismGain(),
		dispatchScoreWaveExecutionPlan(root),
		dispatchScorePrelaunchGate(),
		dispatchScoreRepartitionAdvice(root),
		dispatchScoreIssueScopedWave(root),
		dispatchScoreRouterPathScope(),
		dispatchScoreReactiveLeaseFloor(root),
	}
	return foldDispatchPlanScorecard(probes)
}

func initDispatchScoreRoot(root string) error {
	cmd := exec.Command("git", "init", "-q", root)
	configureDispatchHelperCommand(cmd)
	return cmd.Run()
}

func foldDispatchPlanScorecard(probes []dispatchPlanProbe) dispatchPlanScorecard {
	passed := 0
	for i := range probes {
		if probes[i].Pass {
			passed++
			probes[i].Score = 100
		} else {
			probes[i].Debt = 1
		}
	}
	score := 0
	if len(probes) > 0 {
		score = dispatchWavePct(passed, len(probes))
	}
	debt := len(probes) - passed
	rep := dispatchPlanScorecard{
		Schema: dispatchPlanScorecardSchema,
		OK:     debt == 0,
		Corpus: dispatchPlanCorpus{
			Score:            score,
			Grade:            dispatchPlanGrade(score),
			DispatchPlanDebt: debt,
			Probes:           len(probes),
			Passed:           passed,
		},
		KPIs: probes,
	}
	if rep.OK {
		rep.Verdict = "OK"
		rep.Finding = fmt.Sprintf("dispatch planning floor is fully witnessed (%d/%d probes)", passed, len(probes))
		rep.Reason = "predictive price, wave schedule, scoped parallelism gain, wave execution plan, prelaunch audit gate, repartition advice, issue-path scoped fan-out, router path extraction, and reactive lease overlap refusal all pass"
		rep.NextAction = "run dispatch scorecard --live-router for current issue-shape telemetry, or add semantic conflict probes"
	} else {
		rep.Verdict = "ACTION"
		rep.Finding = fmt.Sprintf("dispatch planning floor has %d debt item(s)", debt)
		rep.Reason = "one or more predictive/reactive deconflict probes failed"
		rep.NextAction = firstDispatchPlanFailure(probes)
	}
	return rep
}

func buildDispatchPlanLiveTelemetry(root string, requested int) dispatchPlanLiveTelemetry {
	live := dispatchPlanLiveTelemetry{
		Schema:    "fak-dispatch-plan-live/1",
		Workspace: root,
		Requested: requested,
	}
	router, err := dispatchRouteIssues(root, io.Discard)
	if err != nil {
		live.OK = false
		live.Verdict = "UNAVAILABLE"
		live.Reason = "current issue router failed: " + err.Error()
		live.NextAction = "repair the live router inputs, then re-run dispatch scorecard --live-router"
		return live
	}
	live.RouterOK = router.OK
	live.RouterVerdict = router.Verdict
	live.Coverage = router.Coverage
	live.Counts = router.Counts
	live.LaneCount = len(router.Lanes)
	price, err := priceDispatchWavePayload(root, router, requested, requested, "", nil, dispatchtick.DefaultCooldownMinutes)
	if err != nil {
		live.OK = false
		live.Verdict = "UNAVAILABLE"
		live.Reason = "current issue fan-out could not be priced: " + err.Error()
		live.NextAction = "repair the live fan-out inputs, then re-run dispatch scorecard --live-router"
		return live
	}
	live.Action = price.Action
	live.CandidateCount = price.CandidateCount
	live.ScopedCount = price.ScopedCount
	live.UnscopedCount = price.UnscopedCount
	live.ScopeCoveragePct = price.ScopeCoveragePct
	live.SafeConcurrency = price.SafeConcurrency
	live.SafeConcurrencyPct = price.SafeConcurrencyPct
	live.SameLaneParallelism = price.SameLaneParallelism
	live.WaveCount = price.WaveCount
	live.Waves = append([]dispatchPriceWave(nil), price.Waves...)
	live.LaunchPlan = append([]dispatchLaunchWave(nil), price.LaunchPlan...)
	live.LaneSerialWaveCount = price.LaneSerialWaveCount
	live.ScopedParallelGain = price.ScopedParallelGain
	live.CollisionWavePenalty = price.CollisionWavePenalty
	live.CollisionsAvoided = price.CollisionsAvoided
	live.SerializationWasted = price.SerializationWasted
	live.RepartitionCount = len(price.Repartition)
	live.RunTargets = append([]dispatchWaveCandidate(nil), price.RunTargets...)
	live.Repartition = append([]dispatchorder.RepartitionAdvice(nil), price.Repartition...)
	live.OK = router.OK
	if !router.OK {
		live.Verdict = "ROUTER_ACTION"
		live.Reason = dispatchScoreFirstNonEmpty(router.Reason, "live router reported a non-OK verdict")
		live.NextAction = "fix the live router finding before trusting current issue-shape telemetry"
		return live
	}
	live.Verdict = liveDispatchPlanVerdict(price)
	live.Reason = liveDispatchPlanReason(price)
	live.NextAction = liveDispatchPlanNextAction(price)
	return live
}

func dispatchScoreFirstNonEmpty(vals ...string) string {
	for _, val := range vals {
		if strings.TrimSpace(val) != "" {
			return val
		}
	}
	return ""
}

func liveDispatchPlanVerdict(price dispatchWavePrice) string {
	switch {
	case price.CandidateCount == 0:
		return "EMPTY"
	case price.CollisionsAvoided > 0:
		return "REPARTITION"
	default:
		return "OK"
	}
}

func liveDispatchPlanReason(price dispatchWavePrice) string {
	switch {
	case price.CandidateCount == 0:
		return "no current launchable issue candidates after held-lane and cooldown filters"
	case price.CollisionsAvoided > 0:
		return fmt.Sprintf("current router shape has %d geometric collision(s) before launch", price.CollisionsAvoided)
	case price.SameLaneParallelism > 0:
		return fmt.Sprintf("current router shape exposes %d extra same-lane parallel target(s)", price.SameLaneParallelism)
	default:
		return "current launchable issue candidates are collision-priced"
	}
}

func liveDispatchPlanNextAction(price dispatchWavePrice) string {
	switch {
	case price.CandidateCount == 0:
		return "route or unblock issues before launching another dispatch wave"
	case price.CollisionsAvoided > 0:
		return "apply the repartition advice, re-price, then launch only the safe set"
	case price.ScopeCoveragePct < 50:
		return "raise issue path-scope coverage so one display lane can split into narrower leases"
	case price.SameLaneParallelism > 0:
		return "launch the priced issue-scoped wave; same-lane targets are already disjoint"
	default:
		return "hold the deterministic floor; add semantic conflict probes for the next planning gain"
	}
}

func dispatchScoreScopedParallelismGain() dispatchPlanProbe {
	price := buildDispatchPriceReport([]dispatchPriceAgent{
		{Name: "gateway-http", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}},
		{Name: "gateway-mcp", Lane: "gateway", Tree: []string{"internal/gateway/mcp.go"}},
	}, dispatchtick.LaneTaxonomy{
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
		},
	})
	pass := price.Action == "LAUNCH_ALL" &&
		price.WaveCount == 1 &&
		price.LaneSerialWaveCount == 2 &&
		price.ScopedParallelGain == 1 &&
		price.CollisionWavePenalty == 0 &&
		launchPlanHasDistinctLeases(price.LaunchPlan, "gateway")
	return dispatchScoreProbe("scoped_parallelism_gain", pass,
		"path-scoped workers in one display lane beat lane-level serialization by one wave",
		map[string]any{
			"action":                  price.Action,
			"wave_count":              price.WaveCount,
			"lane_serial_wave_count":  price.LaneSerialWaveCount,
			"scoped_parallelism_gain": price.ScopedParallelGain,
			"collision_wave_penalty":  price.CollisionWavePenalty,
			"waves":                   price.Waves,
			"launch_plan":             price.LaunchPlan,
		})
}

func dispatchScorePriceWaveSchedule() dispatchPlanProbe {
	price := buildDispatchPriceReport([]dispatchPriceAgent{
		{Name: "gateway", Lane: "gateway"},
		{Name: "gateway-http", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}},
		{Name: "docs", Lane: "docs"},
	}, dispatchtick.LaneTaxonomy{
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
			"docs":    {"docs/**"},
		},
	})
	pass := price.Action == "LAUNCH_SAFE_SET" &&
		price.WaveCount == 2 &&
		strings.Join(price.Waves[0].Agents, ",") == "gateway,docs" &&
		strings.Join(price.Waves[1].Agents, ",") == "gateway-http" &&
		price.ExpectedRework == 2 &&
		len(price.LaunchPlan) == 2 &&
		len(price.LaunchPlan[0].Targets) == 2 &&
		len(price.LaunchPlan[1].Targets) == 1
	return dispatchScoreProbe("price_wave_schedule", pass,
		"direct proposed fan-out pricing returns a deterministic launch schedule for all agents",
		map[string]any{
			"action":          price.Action,
			"safe_now":        price.SafeNow,
			"wave_count":      price.WaveCount,
			"waves":           price.Waves,
			"launch_plan":     price.LaunchPlan,
			"expected_rework": price.ExpectedRework,
		})
}

func dispatchScoreWaveExecutionPlan(root string) dispatchPlanProbe {
	price, err := priceDispatchWavePayload(root, dispatchtick.RouterPayload{
		Schema: dispatchtick.RouterSchema,
		OK:     true,
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"gateway": {
				Tree:   []string{"internal/gateway/**"},
				Issues: []int{20, 21},
				Count:  2,
			},
		},
		Issues: []dispatchtick.IssueRoute{
			{Number: 20, Title: "gateway http", Lane: "gateway", Confidence: "path-confirmed", Paths: []string{"internal/gateway/http.go"}},
			{Number: 21, Title: "gateway mcp", Lane: "gateway", Confidence: "path-confirmed", Paths: []string{"internal/gateway/mcp.go"}},
		},
	}, 2, 2, "", nil, 0)
	if err != nil {
		return dispatchScoreProbe("wave_execution_plan", false, err.Error(), nil)
	}
	plan := dispatchWaveExecutionPlans(root, "claude", "engineering", "wave-score", 0, price.RunTargets, []dispatchtick.AccountWaveLane{
		{Tag: "acct-a", Account: ".claude-a", ConfigDir: "acct-a", Model: "claude", SelectedTier: 1, Rank: 0, WaveID: "wave-score", Size: 2},
		{Tag: "acct-b", Account: ".claude-b", ConfigDir: "acct-b", Model: "claude", SelectedTier: 1, Rank: 1, WaveID: "wave-score", Size: 2},
	}, false)
	pass := len(plan) == 2 &&
		plan[0].Account["tag"] != plan[1].Account["tag"] &&
		plan[0].Target.LeaseID != plan[1].Target.LeaseID &&
		dispatchStringListContains(plan[0].DispatchTickCommand, "fak") &&
		dispatchStringListContains(plan[0].DispatchTickCommand, "--target-issue") &&
		dispatchStringListContains(plan[0].DispatchTickCommand, "--account-tag") &&
		dispatchStringListContains(plan[0].DispatchTickCommand, "--wave-id")
	return dispatchScoreProbe("wave_execution_plan", pass,
		"priced safe-set targets are joined to account seats and executable dispatch tick commands",
		map[string]any{
			"run_targets":    price.RunTargets,
			"execution_plan": plan,
		})
}

func dispatchScorePrelaunchGate() dispatchPlanProbe {
	gate := dispatchWavePrelaunchGateFromAudit("score-plan", []dispatchWaveExecutionAudit{
		{
			Rank:    0,
			Target:  dispatchLaunchTarget{ID: "gateway#20", LeaseID: "resolve-gateway-20"},
			OK:      true,
			Action:  "would_spawn",
			Verdict: "WOULD_SPAWN",
		},
		{
			Rank:    1,
			Target:  dispatchLaunchTarget{ID: "gateway#21", LeaseID: "resolve-gateway-21"},
			OK:      false,
			Action:  "self_modify_hold",
			Verdict: "SELF_MODIFY_HOLD",
			Reason:  "issue #21 targets fak's own running source",
		},
	})
	pass := !gate.OK &&
		gate.Action == "HOLD" &&
		gate.TargetCount == 2 &&
		gate.ExecutionPlanID == "score-plan" &&
		gate.ReadyCount == 1 &&
		gate.RefusedCount == 1 &&
		len(gate.Refused) == 1 &&
		gate.Refused[0].Target == "gateway#21"
	return dispatchScoreProbe("prelaunch_audit_gate", pass,
		"one failed execution audit holds the whole live wave before any worker launches",
		map[string]any{
			"prelaunch_gate": gate,
		})
}

func dispatchScoreCollisionPrice(root string) dispatchPlanProbe {
	price, err := priceDispatchWavePayload(root, dispatchtick.RouterPayload{
		Schema: dispatchtick.RouterSchema,
		OK:     true,
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"gateway": {
				Tree:   []string{"internal/gateway/**"},
				Issues: []int{10},
				Count:  3,
			},
			"gateway-http": {
				Tree:   []string{"internal/gateway/http.go"},
				Issues: []int{11},
				Count:  2,
			},
			"docs": {
				Tree:   []string{"docs/**"},
				Issues: []int{12},
				Count:  1,
			},
		},
	}, 3, 3, "", nil, 0)
	if err != nil {
		return dispatchScoreProbe("collision_price_serializes", false, err.Error(), nil)
	}
	pass := price.Action == "LAUNCH_SAFE_SET" &&
		price.CollisionsAvoided == 1 &&
		price.SerializationWasted == 1 &&
		price.SafeConcurrency == 2 &&
		price.WaveCount == 2 &&
		price.CollisionWavePenalty == 1 &&
		strings.Join(price.RunLanes, ",") == "gateway,docs"
	return dispatchScoreProbe("collision_price_serializes", pass,
		"overlapping lane trees serialize before launch while the disjoint safe set remains runnable",
		map[string]any{
			"action":               price.Action,
			"run_lanes":            price.RunLanes,
			"wave_count":           price.WaveCount,
			"collision_penalty":    price.CollisionWavePenalty,
			"collisions_avoided":   price.CollisionsAvoided,
			"safe_concurrency":     price.SafeConcurrency,
			"serialization_wasted": price.SerializationWasted,
		})
}

func dispatchScoreRepartitionAdvice(root string) dispatchPlanProbe {
	price, err := priceDispatchWavePayload(root, dispatchtick.RouterPayload{
		Schema: dispatchtick.RouterSchema,
		OK:     true,
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"gateway": {
				Tree:   []string{"internal/gateway/**"},
				Issues: []int{10},
				Count:  3,
			},
			"gateway-http": {
				Tree:   []string{"internal/gateway/http.go"},
				Issues: []int{11},
				Count:  2,
			},
		},
	}, 2, 2, "", nil, 0)
	if err != nil {
		return dispatchScoreProbe("repartition_advice", false, err.Error(), nil)
	}
	pass := len(price.Repartition) == 1 &&
		price.Repartition[0].Candidate == "gateway-http#11" &&
		price.Repartition[0].Action == "narrow_to_issue_paths" &&
		price.Repartition[0].Reason == dispatchorder.ReasonCollisionRisk
	return dispatchScoreProbe("repartition_advice", pass,
		"a colliding safe-set hold carries machine-readable narrowing advice before the next re-price",
		map[string]any{
			"action":      price.Action,
			"repartition": price.Repartition,
		})
}

func dispatchScoreIssueScopedWave(root string) dispatchPlanProbe {
	price, err := priceDispatchWavePayload(root, dispatchtick.RouterPayload{
		Schema: dispatchtick.RouterSchema,
		OK:     true,
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"gateway": {
				Tree:   []string{"internal/gateway/**"},
				Issues: []int{20, 21},
				Count:  2,
			},
		},
		Issues: []dispatchtick.IssueRoute{
			{Number: 20, Title: "gateway http", Lane: "gateway", Confidence: "path-confirmed", Paths: []string{"internal/gateway/http.go"}},
			{Number: 21, Title: "gateway mcp", Lane: "gateway", Confidence: "path-confirmed", Paths: []string{"internal/gateway/mcp.go"}},
		},
	}, 2, 2, "", nil, 0)
	if err != nil {
		return dispatchScoreProbe("issue_scoped_same_lane_parallelism", false, err.Error(), nil)
	}
	distinctLease := len(price.RunTargets) == 2 && price.RunTargets[0].LeaseID != price.RunTargets[1].LeaseID
	scopedTargets := true
	for _, target := range price.RunTargets {
		scopedTargets = scopedTargets && target.Scoped && target.Lane == "gateway" && len(target.Tree) == 1
	}
	pass := price.Action == "LAUNCH_ALL" &&
		price.ScopeCoveragePct == 100 &&
		price.SafeConcurrencyPct == 100 &&
		price.SameLaneParallelism == 1 &&
		price.WaveCount == 1 &&
		price.LaneSerialWaveCount == 2 &&
		price.ScopedParallelGain == 1 &&
		launchPlanHasDistinctLeases(price.LaunchPlan, "gateway") &&
		launchPlanHasDispatchTickArgs(price.LaunchPlan, "gateway") &&
		distinctLease &&
		scopedTargets
	return dispatchScoreProbe("issue_scoped_same_lane_parallelism", pass,
		"path-confirmed issues in one display lane get distinct lease IDs and run together when their file trees are disjoint",
		map[string]any{
			"action":                price.Action,
			"run_lanes":             price.RunLanes,
			"scope_coverage_pct":    price.ScopeCoveragePct,
			"safe_concurrency_pct":  price.SafeConcurrencyPct,
			"same_lane_parallelism": price.SameLaneParallelism,
			"wave_count":            price.WaveCount,
			"lane_serial":           price.LaneSerialWaveCount,
			"scoped_gain":           price.ScopedParallelGain,
			"run_targets":           price.RunTargets,
			"launch_plan":           price.LaunchPlan,
		})
}

func launchPlanHasDispatchTickArgs(plan []dispatchLaunchWave, lane string) bool {
	count := 0
	for _, wave := range plan {
		for _, target := range wave.Targets {
			if target.Lane != lane {
				continue
			}
			count++
			for _, want := range []string{"--lane", lane, "--target-issue", "--lease-id", target.LeaseID, "--lease-tree", strings.Join(target.Tree, ",")} {
				if !dispatchStringListContains(target.TickArgs, want) {
					return false
				}
			}
		}
	}
	return count > 1
}

func dispatchStringListContains(values []string, needle string) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func launchPlanHasDistinctLeases(plan []dispatchLaunchWave, lane string) bool {
	seen := map[string]bool{}
	count := 0
	for _, wave := range plan {
		for _, target := range wave.Targets {
			if target.Lane != lane {
				continue
			}
			if target.LeaseID == "" || seen[target.LeaseID] || len(target.Tree) == 0 {
				return false
			}
			seen[target.LeaseID] = true
			count++
		}
	}
	return count > 1
}

func dispatchScoreRouterPathScope() dispatchPlanProbe {
	taxonomy := dispatchtick.LaneTaxonomy{
		Concurrent: []string{"gateway", "docs"},
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
			"docs":    {"docs/**"},
		},
	}
	route := dispatchtick.RouteIssue(dispatchtick.Issue{
		Number: 30,
		Title:  "fix(gateway): split handlers",
		Body:   "touches fak/internal/gateway/http.go and fak/internal/gateway/mcp.go",
	}, taxonomy, dispatchtick.RouteOptions{})
	want := []string{"internal/gateway/http.go", "internal/gateway/mcp.go"}
	pass := route.Lane == "gateway" && route.Confidence == "path-confirmed" && equalStringSlices(route.Paths, want)
	return dispatchScoreProbe("router_carries_path_scope", pass,
		"the issue router carries normalized repo paths into the wave-pricing partition",
		map[string]any{
			"lane":       route.Lane,
			"confidence": route.Confidence,
			"paths":      route.Paths,
		})
}

func dispatchScoreReactiveLeaseFloor(root string) dispatchPlanProbe {
	first := acquireDispatchLaneLease(root, "score-gateway", "gateway", []string{"internal/gateway/**"}, 60)
	overlap := acquireDispatchLaneLease(root, "score-gateway-http", "gateway-http", []string{"internal/gateway/http.go"}, 60)
	disjoint := acquireDispatchLaneLease(root, "score-docs", "docs", []string{"docs/**"}, 60)
	pass := dispatchMapBool(first, "acquired") &&
		dispatchMapBool(overlap, "refused") &&
		dispatchMapString(overlap, "reason") == dispatchorder.ReasonCollisionRisk &&
		dispatchMapBool(disjoint, "acquired")
	return dispatchScoreProbe("reactive_lease_overlap_floor", pass,
		"the acquire-time lease floor refuses an overlapping tree and still admits a disjoint tree",
		map[string]any{
			"first":    first,
			"overlap":  overlap,
			"disjoint": disjoint,
		})
}

func dispatchScoreProbe(name string, pass bool, detail string, data map[string]any) dispatchPlanProbe {
	return dispatchPlanProbe{
		Name:   name,
		Group:  "dispatch_plan",
		Pass:   pass,
		Detail: detail,
		Data:   data,
	}
}

func renderDispatchPlanScorecard(rep dispatchPlanScorecard) string {
	var b strings.Builder
	fmt.Fprintf(&b, "dispatch planning scorecard: %s score=%d grade=%s debt=%d\n",
		rep.Verdict, rep.Corpus.Score, rep.Corpus.Grade, rep.Corpus.DispatchPlanDebt)
	fmt.Fprintf(&b, "  %s\n", rep.Finding)
	for _, k := range rep.KPIs {
		state := "PASS"
		if !k.Pass {
			state = "DEBT"
		}
		fmt.Fprintf(&b, "  %-4s %-36s %s\n", state, k.Name, k.Detail)
	}
	if rep.Live != nil {
		fmt.Fprintf(&b, "  live-router: %s action=%s candidates=%d scoped=%d scope=%d%% safe=%d/%d waves=%d lane_serial=%d scoped_gain=%d collision_penalty=%d same-lane=%d collisions=%d repartition=%d\n",
			rep.Live.Verdict, rep.Live.Action, rep.Live.CandidateCount, rep.Live.ScopedCount,
			rep.Live.ScopeCoveragePct, rep.Live.SafeConcurrency, rep.Live.CandidateCount,
			rep.Live.WaveCount, rep.Live.LaneSerialWaveCount, rep.Live.ScopedParallelGain, rep.Live.CollisionWavePenalty,
			rep.Live.SameLaneParallelism, rep.Live.CollisionsAvoided, rep.Live.RepartitionCount)
		if !rep.Live.OK {
			fmt.Fprintf(&b, "    %s\n", rep.Live.Reason)
		}
	}
	if rep.NextAction != "" {
		fmt.Fprintf(&b, "  next: %s\n", rep.NextAction)
	}
	return b.String()
}

func firstDispatchPlanFailure(probes []dispatchPlanProbe) string {
	for _, p := range probes {
		if !p.Pass {
			return "fix " + p.Name + ": " + p.Detail
		}
	}
	return "run dispatch scorecard --live-router for current issue-shape telemetry, or add semantic conflict probes"
}

func dispatchPlanGrade(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
