package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

type dispatchWavePrice struct {
	Schema               string                            `json:"schema"`
	Action               string                            `json:"action"`
	ActionReason         string                            `json:"action_reason"`
	Requested            int                               `json:"requested"`
	Granted              int                               `json:"granted"`
	EffectiveCap         int                               `json:"effective_cap"`
	FreshStartCap        int                               `json:"fresh_start_cap"`
	FreshStarts          int                               `json:"fresh_starts"`
	CandidateCount       int                               `json:"candidate_count"`
	CandidateStepBudget  int                               `json:"candidate_step_budget,omitempty"`
	ScopedCount          int                               `json:"scoped_count"`
	UnscopedCount        int                               `json:"unscoped_count"`
	ScopeCoveragePct     int                               `json:"scope_coverage_pct"`
	RunLanes             []string                          `json:"run_lanes"`
	RunStepBudget        int                               `json:"run_step_budget,omitempty"`
	RunTargets           []dispatchWaveCandidate           `json:"run_targets"`
	HeldLanes            []string                          `json:"held_lanes,omitempty"`
	ExcludedLanes        []string                          `json:"excluded_lanes,omitempty"`
	Candidates           []dispatchWaveCandidate           `json:"candidates"`
	Collisions           []dispatchorder.Collision         `json:"collisions,omitempty"`
	Repartition          []dispatchorder.RepartitionAdvice `json:"repartition,omitempty"`
	CollisionsAvoided    int                               `json:"collisions_avoided"`
	LanesUtilized        int                               `json:"lanes_utilized"`
	SerializationWasted  int                               `json:"serialization_wasted"`
	SafeConcurrency      int                               `json:"safe_concurrency"`
	SafeConcurrencyPct   int                               `json:"safe_concurrency_pct"`
	SameLaneParallelism  int                               `json:"same_lane_parallelism"`
	WaveCount            int                               `json:"wave_count"`
	Waves                []dispatchPriceWave               `json:"waves,omitempty"`
	LaunchPlan           []dispatchLaunchWave              `json:"launch_plan,omitempty"`
	LaneSerialWaveCount  int                               `json:"lane_serial_wave_count"`
	ScopedParallelGain   int                               `json:"scoped_parallelism_gain"`
	CollisionWavePenalty int                               `json:"collision_wave_penalty"`
	ExpectedRework       int                               `json:"expected_rework"`
}

const (
	dispatchWaveReasonWaveCap     = "wave-cap"
	dispatchWaveReasonFreshWIPCap = "fresh-wip-cap"
	dispatchWaveDefaultFreshCap   = 1
)

type dispatchWaveCandidate struct {
	ID           string                    `json:"id"`
	Lane         string                    `json:"lane"`
	LeaseID      string                    `json:"lease_id"`
	Issue        int                       `json:"issue,omitempty"`
	BaseWeight   int                       `json:"base_weight"`
	ReadySince   int64                     `json:"ready_since"`
	StepBudget   int                       `json:"step_budget,omitempty"`
	Tree         []string                  `json:"tree,omitempty"`
	Scoped       bool                      `json:"scoped"`
	Disposition  dispatchorder.Disposition `json:"disposition"`
	Reason       string                    `json:"reason"`
	CollidesWith []string                  `json:"collides_with,omitempty"`
	Rank         int                       `json:"rank"`
	Selected     bool                      `json:"selected"`
}

type dispatchWaveExecutionPlan struct {
	Rank                int                  `json:"rank"`
	WaveID              string               `json:"wave_id"`
	WaveSize            int                  `json:"wave_size"`
	Shortfall           int                  `json:"shortfall"`
	Backend             string               `json:"backend"`
	WorkKind            string               `json:"work_kind"`
	Goal                string               `json:"goal,omitempty"`
	GoalProfile         string               `json:"goal_profile,omitempty"`
	Target              dispatchLaunchTarget `json:"target"`
	Account             map[string]any       `json:"account"`
	RecordLoop          bool                 `json:"record_loop"`
	DispatchTickArgs    []string             `json:"dispatch_tick_args"`
	DispatchTickCommand []string             `json:"dispatch_tick_command"`
}

type dispatchWaveExecutionAudit struct {
	Rank          int                  `json:"rank"`
	Target        dispatchLaunchTarget `json:"target"`
	Account       map[string]any       `json:"account,omitempty"`
	OK            bool                 `json:"ok"`
	Action        string               `json:"action"`
	Verdict       string               `json:"verdict"`
	Reason        string               `json:"reason,omitempty"`
	TargetIssue   any                  `json:"target_issue,omitempty"`
	LeaseID       string               `json:"lease_id,omitempty"`
	LeaseTree     []string             `json:"lease_tree,omitempty"`
	Guarded       bool                 `json:"guarded"`
	LaunchCommand []string             `json:"launch_command,omitempty"`
	Error         string               `json:"error,omitempty"`
}

type dispatchWavePrelaunchGate struct {
	OK              bool                           `json:"ok"`
	Action          string                         `json:"action"`
	ExecutionPlanID string                         `json:"execution_plan_id,omitempty"`
	TargetCount     int                            `json:"target_count"`
	ReadyCount      int                            `json:"ready_count"`
	RefusedCount    int                            `json:"refused_count"`
	ErrorCount      int                            `json:"error_count,omitempty"`
	Reason          string                         `json:"reason,omitempty"`
	Refused         []dispatchWavePrelaunchRefusal `json:"refused,omitempty"`
}

type dispatchWavePrelaunchRefusal struct {
	Rank    int    `json:"rank"`
	Target  string `json:"target"`
	Action  string `json:"action,omitempty"`
	Verdict string `json:"verdict,omitempty"`
	Reason  string `json:"reason,omitempty"`
	Error   string `json:"error,omitempty"`
}

func runDispatchWave(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch wave", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	count := fs.Int("count", 2, "number of account session slots to allocate")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "hard cap on live workers, enforced by each tick's preflight")
	freshStartCap := fs.Int("fresh-start-cap", dispatchWaveDefaultFreshCap, "maximum never-attempted issues admitted this wave (attempted WIP is not counted)")
	backend := fs.String("backend", "codex", "worker backend (claude|opencode|codex); default codex")
	workKind := fs.String("work-kind", "", "switcher work kind (default follows --backend)")
	goal := fs.String("goal", "", "durable dispatch loop goal id (for example throughput or high-priority); forwarded to each tick")
	goalProfile := fs.String("goal-profile", "", "dispatch picker profile: throughput|high-priority (default follows --goal, else throughput)")
	lane := fs.String("lane", "", "pin every tick to this repo lane (default: largest step-budget lane pick)")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the step-budget pick")
	settleS := fs.Float64("settle-s", 2.0, "seconds to wait after each live spawn")
	noLedger := fs.Bool("no-loop-ledger", false, "disable loop-ledger append for spawned ticks")
	codexLoopGate := fs.String("codex-loop-gate", dispatchCodexLoopGateDefaultThreshold(), "for live Codex workers, audit recent Codex sessions before spawn and refuse at threshold: loop|action|off")
	codexLoopGateSinceHours := fs.Float64("codex-loop-gate-since-hours", dispatchCodexLoopGateDefaultSinceHoursValue(), "with --codex-loop-gate, only scan Codex sessions modified within N hours (0 = all)")
	codexLoopGateLimit := fs.Int("codex-loop-gate-limit", dispatchCodexLoopGateDefaultLimitValue(), "with --codex-loop-gate, maximum newest Codex sessions to scan")
	live := fs.Bool("live", false, "actually spawn workers")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	root := *workspace
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch wave: getwd: %v\n", err)
			return 1
		}
		root = wd
	}
	if *freshStartCap < 0 {
		fmt.Fprintln(stderr, "fak dispatch wave: --fresh-start-cap must be >= 0")
		return 2
	}
	backendNorm, err := dispatchtick.NormalizeBackend(*backend)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch wave: %v\n", err)
		return 2
	}
	wk := strings.TrimSpace(*workKind)
	if wk == "" {
		wk = dispatchtick.DefaultWorkKind(backendNorm)
	}
	if *count <= 0 {
		fmt.Fprintln(stderr, "fak dispatch wave: --count must be > 0")
		return 2
	}
	goalID, profile, goalErr := normalizeDispatchGoal(*goal, *goalProfile)
	if goalErr != nil {
		fmt.Fprintf(stderr, "fak dispatch wave: %v\n", goalErr)
		return 2
	}

	preflightResult, preflightErr := dispatchWaveDependencyRetry(3*dispatchWaveDependencyTimeout, "dispatch preflight", 2, func(error) bool { return true }, func() (dispatchWavePreflightResult, error) {
		product, allocationCount, shortfall, preflight, err := dispatchWavePreflightAlloc(root, stderr, *maxWorkers, wk, backendNorm, *count)
		return dispatchWavePreflightResult{Product: product, AllocationCount: allocationCount, Shortfall: shortfall, Payload: preflight}, err
	})
	if preflightErr != nil {
		rec := newDispatchWaveRecord(root, *live, backendNorm, wk, goalID, profile, *count, 0, nil)
		rec["granted"] = 0
		rec["shortfall"] = *count
		dispatchWaveRecordDependencyError(rec, preflightErr)
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}
	product, allocationCount, preflightShortfall, preflight := preflightResult.Product, preflightResult.AllocationCount, preflightResult.Shortfall, preflightResult.Payload

	rec := newDispatchWaveRecord(root, *live, backendNorm, wk, goalID, profile, *count, allocationCount, preflight)
	if allocationCount <= 0 {
		rec["granted"] = 0
		rec["shortfall"] = *count
		rec["stop_reason"] = "preflight headroom exhausted before account allocation"
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}

	rows, err := dispatchReadAccountRoster(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch wave: allocate accounts: %v\n", err)
		return 1
	}
	alloc := dispatchtick.AllocateWave(dispatchtick.AccountWaveInput{
		Rows:     rows,
		Leases:   dispatchLiveSeatLeases(filepath.Join(root, dispatchtick.RunsDirName)),
		Count:    allocationCount,
		WorkKind: wk,
		Product:  product,
	})
	lanes := alloc.Lanes
	waveID := alloc.WaveID
	shortfall := alloc.Shortfall + preflightShortfall
	rec["granted"] = len(lanes)
	rec["shortfall"] = shortfall
	rec["wave_id"] = waveID
	rec["allocation"] = scrubDispatchSecrets(alloc.Map())
	if len(lanes) == 0 {
		rec["stop_reason"] = firstString(alloc.Reason, "no account session slots available")
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}

	excludedLanes := splitCommaList(*excludeLane)
	router, err := dispatchWaveRouteIssuesBounded(root, stderr, dispatchWaveDependencyTimeout)
	if err != nil {
		dispatchWaveRecordDependencyError(rec, err)
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}
	heldIssues := map[int]bool{}
	var executionPlan []dispatchWaveExecutionPlan
	const maxPrelaunchReprice = 8
	for attempt := 0; ; attempt++ {
		price, err := priceDispatchWavePayloadFilteredWithFreshCap(root, router, *count, len(lanes), *lane, excludedLanes, dispatchtick.DefaultCooldownMinutes, heldIssues, *freshStartCap, profile)
		if err != nil {
			rec["stop_reason"] = "price fan-out: " + err.Error()
			return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
		}
		rec["price"] = price
		rec["planned_lanes"] = append([]string(nil), price.RunLanes...)
		executionPlan = dispatchWaveExecutionPlans(root, backendNorm, wk, goalID, profile, waveID, shortfall, price.RunTargets, lanes, !*noLedger, *codexLoopGate, maxFloat64(0, *codexLoopGateSinceHours), *codexLoopGateLimit)
		executionPlanID := dispatchWaveExecutionPlanID(executionPlan)
		rec["execution_plan_id"] = executionPlanID
		rec["execution_plan"] = executionPlan
		if len(price.RunLanes) == 0 {
			rec["stop_reason"] = "priced fan-out found no launchable lane"
			return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
		}
		executionAudit, auditErr := auditDispatchWaveExecutionPlanBounded(root, *maxWorkers, excludedLanes, executionPlan, *codexLoopGate, maxFloat64(0, *codexLoopGateSinceHours), *codexLoopGateLimit, dispatchWaveDependencyTimeout)
		if auditErr != nil {
			rec["execution_plan_audit"] = executionAudit
			dispatchWaveRecordDependencyError(rec, auditErr)
			return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
		}
		rec["execution_plan_audit"] = executionAudit
		prelaunchGate := dispatchWavePrelaunchGateFromAudit(executionPlanID, executionAudit)
		rec["prelaunch_gate"] = prelaunchGate
		readyPlan := dispatchWaveReadyExecutionPlan(executionPlan, executionAudit)
		rec["ready_execution_plan"] = readyPlan
		if prelaunchGate.OK {
			executionPlan = readyPlan
		}
		if *live {
			rec["live_execution_plan"] = readyPlan
		}
		if prelaunchGate.OK {
			break
		}
		// Dry-runs must reprice benign lease/intent races too. Otherwise the required
		// approval plan can refuse on its first stale candidate even though another safe
		// lane is available, while --live would silently use a different plan.
		retryIssues := dispatchWavePrelaunchRetryIssues(executionAudit, *live, attempt, maxPrelaunchReprice)
		if len(retryIssues) > 0 {
			added := false
			for _, issue := range retryIssues {
				if !heldIssues[issue] {
					heldIssues[issue] = true
					added = true
				}
			}
			if added {
				rec["prelaunch_retries"] = appendDispatchWavePrelaunchRetry(rec["prelaunch_retries"], attempt+1, retryIssues, prelaunchGate)
				continue
			}
		}
		rec["stop_reason"] = "prelaunch execution audit refused: " + prelaunchGate.Reason
		rec["ticks"] = []any{}
		rec["spawned"] = 0
		rec["ok"] = false
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}

	ticks := []any{}
	spawned := 0
	limit := len(executionPlan)
	if !*live {
		limit = 1
	}
	discovery := subscribeDispatchWaveDiscovery(root, limit)
	defer closeDispatchDiscoverySubscriptions(discovery)
	for i := 0; i < limit; i++ {
		row := executionPlan[i]
		snapshot := <-discovery[i].Snapshots
		payload, err := evaluateDispatchTick(dispatchWaveExecutionTickOptions(root, *maxWorkers, splitCommaList(*excludeLane), row, *live, i == 0, *codexLoopGate, maxFloat64(0, *codexLoopGateSinceHours), *codexLoopGateLimit, snapshot), stderr)
		if err != nil {
			ticks = append(ticks, map[string]any{"ok": false, "error": err.Error(), "rank": i})
			rec["stop_reason"] = err.Error()
			break
		}
		payload["wave_rank"] = row.Rank
		payload["wave_target"] = row.Target
		ticks = append(ticks, payload)
		if dispatchMapString(payload, "action") == "spawned" {
			spawned++
			if *settleS > 0 {
				time.Sleep(time.Duration(*settleS * float64(time.Second)))
			}
			continue
		}
		if !*live {
			if gate, ok := rec["prelaunch_gate"].(dispatchWavePrelaunchGate); ok && !gate.OK {
				rec["stop_reason"] = "prelaunch execution audit refused: " + gate.Reason
			} else {
				rec["stop_reason"] = "dry-run: planned the first wave tick only; re-run with --live to spawn"
			}
		} else {
			rec["stop_reason"] = firstString(dispatchMapString(payload, "verdict"), dispatchMapString(payload, "action"))
		}
		break
	}
	rec["ticks"] = ticks
	rec["spawned"] = spawned
	if rec["stop_reason"] == "" {
		rec["stop_reason"] = "filled requested wave"
	}
	rec["ok"] = !*live || spawned > 0 || len(ticks) > 0 && dispatchMapBool(ticks[len(ticks)-1].(map[string]any), "ok")
	return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
}

// newDispatchWaveRecord seeds the mutable dispatch-wave result record with the run's static
// header fields and the empty/false defaults runDispatchWave fills in as it progresses.
func newDispatchWaveRecord(root string, live bool, backendNorm, wk, goalID, profile string, count, allocationCount int, preflight map[string]any) map[string]any {
	return map[string]any{
		"schema":               "fleet-issue-dispatch-wave/1",
		"workspace":            root,
		"live":                 live,
		"backend":              backendNorm,
		"work_kind":            wk,
		"goal":                 goalID,
		"goal_profile":         profile,
		"requested":            count,
		"allocation_requested": allocationCount,
		"preflight":            preflight,
		"ticks":                []any{},
		"spawned":              0,
		"stop_reason":          "",
		"ok":                   false,
	}
}

// dispatchWavePreflightAlloc runs the tick preflight for the wave and folds its headroom
// verdict into the account allocation count. It returns the backend product, the (possibly
// reduced) allocation count, the preflight-driven shortfall, and the preflight record (an
// {"error": …} map when the preflight itself failed — fail-open, matching runDispatchWave's
// prior inline behavior).
func dispatchWavePreflightAlloc(root string, stderr io.Writer, maxWorkers int, wk, backendNorm string, count int) (string, int, int, map[string]any, error) {
	product := dispatchtick.ProductForBackend(backendNorm)
	preflight, err := dispatchPreflight(root, stderr, maxWorkers, wk, product)
	if err != nil {
		return product, 0, count, nil, err
	}
	allocationCount := dispatchWaveAllocationCount(count, preflight)
	preflightShortfall := max(0, count-allocationCount)
	return product, allocationCount, preflightShortfall, preflight, nil
}

func priceDispatchWave(root string, stderr io.Writer, requested, granted int, explicitLane string, excluded []string, cooldownMin int, goalProfile ...string) (dispatchWavePrice, error) {
	router, err := dispatchRouteIssues(root, stderr)
	if err != nil {
		return dispatchWavePrice{}, err
	}
	return priceDispatchWavePayload(root, router, requested, granted, explicitLane, excluded, cooldownMin, goalProfile...)
}

func priceDispatchWavePayload(root string, router dispatchtick.RouterPayload, requested, granted int, explicitLane string, excluded []string, cooldownMin int, goalProfile ...string) (dispatchWavePrice, error) {
	return priceDispatchWavePayloadFiltered(root, router, requested, granted, explicitLane, excluded, cooldownMin, nil, goalProfile...)
}

func priceDispatchWavePayloadFiltered(root string, router dispatchtick.RouterPayload, requested, granted int, explicitLane string, excluded []string, cooldownMin int, excludedIssues map[int]bool, goalProfile ...string) (dispatchWavePrice, error) {
	// Library callers retain the historical no-extra-cap behavior unless they choose the
	// explicit admission helper. The live CLI path above always supplies its default-on cap.
	return priceDispatchWavePayloadFilteredWithFreshCap(root, router, requested, granted, explicitLane, excluded, cooldownMin, excludedIssues, requested, goalProfile...)
}

func priceDispatchWavePayloadFilteredWithFreshCap(root string, router dispatchtick.RouterPayload, requested, granted int, explicitLane string, excluded []string, cooldownMin int, excludedIssues map[int]bool, freshStartCap int, goalProfile ...string) (dispatchWavePrice, error) {
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	profile := dispatchWaveGoalProfile(goalProfile)
	newlyUnblocked := map[int]bool{}
	for _, n := range router.NewlyUnblocked {
		newlyUnblocked[n] = true
	}
	readyState := readDispatchPrereqState(dispatchPrereqStatePath(root))
	// One runs-directory scan feeds every view this pricing pass needs -- held lanes, live
	// issues, the cooldown set, and the poison-cap skip set -- instead of re-globbing the
	// sidecars four times (#3593).
	snap := scanRunsSnapshot(runsDir, time.Now())
	held := snap.liveLanes()
	liveIssues := snap.liveIssues()
	cooled := snap.recentlyAttempted(cooldownMin)
	skipIssues := dispatchWaveSkipIssuesFrom(snap, excludedIssues)
	exclude := map[string]bool{}
	for _, lane := range excluded {
		exclude[lane] = true
	}
	for lane := range held {
		exclude[lane] = true
	}
	// #4285: soft-exclude lanes the #2062 low-yield fold flagged (recent finished
	// sessions burned turns yet closed nothing). This is the load-bearing merge: each
	// priced row runs a tick with its lane PINNED, which bypasses the tick exclude, so
	// steering the fleet away from a poison lane has to happen here at pricing time.
	// Auto-pick only -- an explicit --lane still overrides. Fail-open (nil on error).
	if explicitLane == "" {
		for lane := range dispatchLowYieldExcludes(root) {
			exclude[lane] = true
		}
	}

	lanes := make([]string, 0, len(router.Lanes))
	for lane := range router.Lanes {
		lanes = append(lanes, lane)
	}
	sort.Slice(lanes, func(i, j int) bool {
		bi, bj := dispatchWaveLaneStepBudget(router.Lanes[lanes[i]]), dispatchWaveLaneStepBudget(router.Lanes[lanes[j]])
		if bi != bj {
			return bi > bj
		}
		ci, cj := router.Lanes[lanes[i]].Count, router.Lanes[lanes[j]].Count
		if ci != cj {
			return ci > cj
		}
		return lanes[i] < lanes[j]
	})

	issueByLane := map[string]int{}
	meta := map[string]dispatchWaveCandidate{}
	cands := make([]dispatchorder.Candidate, 0, len(router.Issues)+len(lanes))
	unscopedByLane := map[string][]int{}
	scopedByLane := map[string]bool{}
	for _, route := range router.Issues {
		lane := strings.TrimSpace(route.Lane)
		if lane == "" {
			continue
		}
		if explicitLane != "" && lane != explicitLane {
			continue
		}
		if exclude[lane] {
			continue
		}
		if liveIssues[route.Number] || cooled[route.Number] || skipIssues[route.Number] {
			continue
		}
		paths := append([]string(nil), route.Paths...)
		if len(paths) == 0 {
			unscopedByLane[lane] = append(unscopedByLane[lane], route.Number)
			continue
		}
		scopedByLane[lane] = true
		id := waveCandidateID(lane, route.Number)
		leaseID := dispatchIssueLeaseID(lane, route.Number)
		stepBudget := dispatchWaveRouteStepBudget(route)
		priority := dispatchtick.PriorityWeightDefault
		if grp, ok := router.Lanes[lane]; ok {
			if w, ok := grp.Priority[route.Number]; ok {
				priority = w
			}
		}
		meta[id] = dispatchWaveCandidate{
			ID:         id,
			Lane:       lane,
			LeaseID:    leaseID,
			Issue:      route.Number,
			BaseWeight: priority,
			ReadySince: dispatchIssueReadySinceStamp(root, readyState, route.Number),
			StepBudget: stepBudget,
			Tree:       paths,
			Scoped:     true,
		}
		lastAttempt := int64(0)
		if attemptedAt, ok := snap.latest[route.Number]; ok {
			lastAttempt = attemptedAt.Unix()
		}
		cands = append(cands, dispatchorder.Candidate{
			ID:              id,
			Key:             id,
			Lane:            leaseID,
			Tree:            paths,
			Mode:            "exclusive",
			UpdatedUnix:     dispatchWaveReleaseStamp(dispatchWaveOrderStamp(profile, priority, stepBudget, dispatchtick.IsCoreSourceLaneTree(paths)), newlyUnblocked[route.Number]),
			CreatedUnix:     int64(route.Number),
			LastAttemptUnix: lastAttempt,
		})
	}
	for i, lane := range lanes {
		if explicitLane != "" && lane != explicitLane {
			continue
		}
		if exclude[lane] {
			continue
		}
		if scopedByLane[lane] {
			continue
		}
		grp := router.Lanes[lane]
		nums := append([]int(nil), unscopedByLane[lane]...)
		if len(router.Issues) == 0 {
			nums = append([]int(nil), grp.Issues...)
		}
		nums = dispatchWaveOrderLaneIssues(root, nums, grp.Priority, readyState)
		issue, ok := firstLaunchableIssue(nums, liveIssues, cooled, skipIssues)
		if !ok {
			continue
		}
		priority := dispatchtick.PriorityWeightDefault
		if w, ok := grp.Priority[issue]; ok {
			priority = w
		}
		id := waveCandidateID(lane, issue)
		if _, exists := meta[id]; exists {
			continue
		}
		leaseID := dispatchIssueLeaseID(lane, issue)
		stepBudget := dispatchWaveLaneStepBudget(grp)
		issueByLane[lane] = issue
		meta[id] = dispatchWaveCandidate{
			ID:         id,
			Lane:       lane,
			LeaseID:    leaseID,
			Issue:      issue,
			BaseWeight: priority,
			ReadySince: dispatchIssueReadySinceStamp(root, readyState, issue),
			StepBudget: stepBudget,
			Tree:       append([]string(nil), grp.Tree...),
		}
		lastAttempt := int64(0)
		if attemptedAt, ok := snap.latest[issue]; ok {
			lastAttempt = attemptedAt.Unix()
		}
		cands = append(cands, dispatchorder.Candidate{
			ID:              id,
			Key:             id,
			Lane:            leaseID,
			Tree:            grp.Tree,
			Mode:            "exclusive",
			UpdatedUnix:     dispatchWaveReleaseStamp(dispatchWaveOrderStamp(profile, priority, stepBudget, dispatchtick.IsCoreSourceLaneTree(grp.Tree)), newlyUnblocked[issue]),
			CreatedUnix:     int64(grp.Count*len(lanes) + (len(lanes) - i)),
			LastAttemptUnix: lastAttempt,
		})
	}

	res := dispatchorder.Plan(dispatchorder.Input{
		Candidates:      cands,
		NowUnix:         time.Now().Unix(),
		CooldownSeconds: -1,
		FinishFirst:     true,
	})
	limit := minInt(requested, granted)
	selected := map[string]bool{}
	freshCapHeld := map[string]bool{}
	runLanes := make([]string, 0, limit)
	freshStarts := 0
	for _, id := range res.Keep {
		if len(runLanes) >= limit {
			break
		}
		cand := meta[id]
		attempted := cand.Issue > 0 && !snap.latest[cand.Issue].IsZero()
		if !attempted && freshStarts >= freshStartCap {
			freshCapHeld[id] = true
			continue
		}
		runLanes = append(runLanes, id)
		selected[id] = true
		if !attempted {
			freshStarts++
		}
	}
	rows := make([]dispatchWaveCandidate, 0, len(res.Order))
	runTargets := make([]dispatchWaveCandidate, 0, len(runLanes))
	for _, row := range res.Order {
		cand := meta[row.ID]
		if cand.ID == "" {
			cand.ID = row.ID
		}
		if cand.Issue == 0 && cand.Lane != "" {
			cand.Issue = issueByLane[cand.Lane]
		}
		cand.Disposition = row.Disposition
		cand.CollidesWith = append([]string(nil), row.CollidesWith...)
		cand.Rank = row.Rank
		cand.Selected = selected[row.ID]
		cand.Reason = dispatchWaveCandidateReason(row, cand.Selected)
		if freshCapHeld[row.ID] {
			cand.Reason = dispatchWaveReasonFreshWIPCap
		}
		rows = append(rows, cand)
		if cand.Selected {
			runTargets = append(runTargets, cand)
		}
	}
	sort.SliceStable(runTargets, func(i, j int) bool {
		return runTargets[i].Rank < runTargets[j].Rank
	})
	runLaneNames := make([]string, 0, len(runTargets))
	for _, target := range runTargets {
		runLaneNames = append(runLaneNames, target.Lane)
	}
	price := dispatchWaveBuildPrice(requested, granted, cands, rows, runTargets, runLanes, runLaneNames, held, exclude, res)
	price.FreshStartCap = freshStartCap
	price.FreshStarts = freshStarts
	return price, nil
}

// dispatchWaveBuildPrice folds the priced candidate rows, the selected run targets, and the
// order planner's result into the final dispatchWavePrice record. It is the pure assembly tail
// of priceDispatchWavePayloadFiltered — no I/O, no mutation of its inputs.
func dispatchWaveBuildPrice(requested, granted int, cands []dispatchorder.Candidate, rows, runTargets []dispatchWaveCandidate, runLanes, runLaneNames []string, held, exclude map[string]bool, res dispatchorder.Result) dispatchWavePrice {
	candidateStepBudget := dispatchWaveStepBudget(rows)
	runStepBudget := dispatchWaveStepBudget(runTargets)
	scopedCount := 0
	for _, cand := range rows {
		if cand.Scoped {
			scopedCount++
		}
	}
	unscopedCount := len(rows) - scopedCount
	waves := dispatchWaveWaves(rows, res.Collisions, res.Keep)
	launchPlan := dispatchWaveLaunchPlan(waves, rows)
	laneSerialWaves := dispatchWaveLaneSerialWaveCount(rows)
	action, actionReason := dispatchWaveAction(len(rows), len(runTargets), res.CollisionsAvoided, res.SerializationWasted)
	return dispatchWavePrice{
		Schema:               "fleet-issue-dispatch-wave-price/1",
		Action:               action,
		ActionReason:         actionReason,
		Requested:            requested,
		Granted:              granted,
		EffectiveCap:         len(runTargets),
		CandidateCount:       len(cands),
		CandidateStepBudget:  candidateStepBudget,
		ScopedCount:          scopedCount,
		UnscopedCount:        unscopedCount,
		ScopeCoveragePct:     dispatchWavePct(scopedCount, len(rows)),
		RunLanes:             runLaneNames,
		RunStepBudget:        runStepBudget,
		RunTargets:           runTargets,
		HeldLanes:            sortedStringSet(held),
		ExcludedLanes:        sortedStringSet(exclude),
		Candidates:           rows,
		Collisions:           res.Collisions,
		Repartition:          res.Repartition,
		CollisionsAvoided:    res.CollisionsAvoided,
		LanesUtilized:        len(runLanes),
		SerializationWasted:  res.SerializationWasted,
		SafeConcurrency:      res.SafeConcurrency,
		SafeConcurrencyPct:   dispatchWavePct(len(runTargets), len(rows)),
		SameLaneParallelism:  sameLaneParallelism(runTargets),
		WaveCount:            len(waves),
		Waves:                waves,
		LaunchPlan:           launchPlan,
		LaneSerialWaveCount:  laneSerialWaves,
		ScopedParallelGain:   positiveDelta(laneSerialWaves, len(waves)),
		CollisionWavePenalty: positiveDelta(len(waves), laneSerialWaves),
		ExpectedRework:       res.CollisionsAvoided + res.SerializationWasted,
	}
}

// dispatchWaveSkipIssuesFrom unions the caller's excluded-issue set with the poison-issue cap: an
// OPEN issue that has burned dispatchAttemptBudget() workers without shipping is held out of the
// wave (the time cooldown only pauses it ~2h, so without this it re-enters the pool and wastes a
// worker every window forever). It reads an already-captured snapshot so the wave pricer computes
// the skip set from the same one scan that fed its live/cooldown views. When the attempt budget
// yields no exhausted issues the result stays == excludedIssues, so behavior is byte-identical.
func dispatchWaveSkipIssuesFrom(snap *runsSnapshot, excludedIssues map[int]bool) map[int]bool {
	skipIssues := excludedIssues
	if exhausted := snap.attemptExhausted(dispatchAttemptBudget()); len(exhausted) > 0 {
		skipIssues = make(map[int]bool, len(excludedIssues)+len(exhausted))
		for n := range excludedIssues {
			skipIssues[n] = true
		}
		for n := range exhausted {
			skipIssues[n] = true
		}
	}
	return skipIssues
}

func dispatchWaveCandidateReason(row dispatchorder.Ranked, selected bool) string {
	if row.Disposition == dispatchorder.DispKeep && !selected {
		return dispatchWaveReasonWaveCap
	}
	return row.Reason
}

func dispatchWaveGoalProfile(profiles []string) string {
	if len(profiles) == 0 || strings.TrimSpace(profiles[0]) == "" {
		return dispatchGoalProfileThroughput
	}
	if p, ok := dispatchGoalProfileAlias(profiles[0]); ok {
		return p
	}
	return dispatchGoalProfileThroughput
}

func dispatchWaveOrderLaneIssues(root string, nums []int, weights map[int]int, readyState dispatchPrereqState) []int {
	cands := make([]dispatchtick.LaneCandidate, len(nums))
	for i, n := range nums {
		weight := dispatchtick.PriorityWeightDefault
		if w, ok := weights[n]; ok {
			weight = w
		}
		cands[i] = dispatchtick.LaneCandidate{
			Number: n, Weight: weight,
			ReadySince: dispatchIssueReadySinceStamp(root, readyState, n),
		}
	}
	return dispatchtick.OrderLaneCandidates(cands, false)
}

func dispatchWaveReleaseStamp(stamp int64, newlyUnblocked bool) int64 {
	if newlyUnblocked {
		return stamp + 100_000_000
	}
	return stamp
}

func dispatchWaveOrderStamp(profile string, priority, stepBudget int, core bool) int64 {
	// core-source boost keeps the default wave progressing fak's own guard-shippable core
	// engineering ahead of the coarse docs/tools buckets: it dominates any step budget but
	// stays BELOW a priority label, so under high-priority the label still leads and under
	// throughput core leads (step budget breaks ties within a class). Assumes a lane's step
	// budget stays well under the 1e6 boost, which the per-lane budgets always do.
	coreBoost := int64(0)
	if core {
		coreBoost = 1_000_000
	}
	if profile == dispatchGoalProfileHighPriority {
		return int64(priority)*1_000_000_000 + coreBoost + int64(stepBudget)
	}
	return coreBoost + int64(stepBudget)
}

func dispatchWaveAction(candidates, run, collisions, wasted int) (string, string) {
	switch {
	case candidates == 0:
		return "HOLD_EMPTY", "no launchable candidates after live/cooldown/held-lane filters"
	case run == candidates && collisions == 0 && wasted == 0:
		return "LAUNCH_ALL", "every priced candidate is disjoint and within the requested wave"
	case run > 0:
		return "LAUNCH_SAFE_SET", "launch the priced disjoint safe set and hold the remaining candidates"
	default:
		return "REPARTITION_AND_REPRICE", "no collision-free candidate is launchable; narrow the scopes and re-price"
	}
}

func sameLaneParallelism(targets []dispatchWaveCandidate) int {
	byLane := map[string]int{}
	for _, target := range targets {
		byLane[target.Lane]++
	}
	extra := 0
	for _, n := range byLane {
		if n > 1 {
			extra += n - 1
		}
	}
	return extra
}

func dispatchWaveWaves(candidates []dispatchWaveCandidate, collisions []dispatchorder.Collision, safeNow []string) []dispatchPriceWave {
	return dispatchWavesForIDs(
		dispatchCandidateKeys(candidates, func(cand dispatchWaveCandidate) string { return cand.ID }),
		collisions, safeNow)
}

func dispatchWaveLaunchPlan(waves []dispatchPriceWave, candidates []dispatchWaveCandidate) []dispatchLaunchWave {
	if len(waves) == 0 {
		return nil
	}
	byID := map[string]dispatchWaveCandidate{}
	for _, cand := range candidates {
		byID[cand.ID] = cand
	}
	return dispatchLaunchPlanFromWaves(waves, func(id string) dispatchLaunchTarget {
		cand, ok := byID[id]
		if !ok {
			return dispatchLaunchTarget{ID: id}
		}
		return dispatchWaveLaunchTarget(cand)
	})
}

func dispatchWaveLaunchTarget(cand dispatchWaveCandidate) dispatchLaunchTarget {
	scopeSource := "lane"
	if cand.Scoped {
		scopeSource = "issue"
	}
	if len(cand.Tree) == 0 {
		scopeSource = "unknown"
	}
	return dispatchLaunchTarget{
		ID:          cand.ID,
		Lane:        cand.Lane,
		LeaseID:     cand.LeaseID,
		Issue:       cand.Issue,
		Tree:        append([]string(nil), cand.Tree...),
		Mode:        "exclusive",
		Scoped:      cand.Scoped,
		ScopeSource: scopeSource,
		Disposition: cand.Disposition,
		Reason:      cand.Reason,
		TickArgs:    dispatchTickArgsForLaunchTarget(cand),
	}
}

func dispatchTickArgsForLaunchTarget(cand dispatchWaveCandidate) []string {
	if cand.Lane == "" || cand.Issue <= 0 {
		return nil
	}
	args := []string{"--lane", cand.Lane, "--target-issue", fmt.Sprint(cand.Issue)}
	if cand.LeaseID != "" {
		args = append(args, "--lease-id", cand.LeaseID)
	}
	if len(cand.Tree) > 0 {
		args = append(args, "--lease-tree", strings.Join(cand.Tree, ","))
	}
	return args
}

// Preflight uses three times this planning-probe budget: its supported host/account/process
// probes can exceed 30s, and the old shared deadline manufactured WAVE_EMPTY before pricing.
const dispatchWaveDependencyTimeout = 30 * time.Second

type dispatchWavePreflightResult struct {
	Product         string
	AllocationCount int
	Shortfall       int
	Payload         map[string]any
}

type dispatchWaveDependencyError struct {
	Dependency string
	Kind       string
	Attempts   int
	Retryable  bool
	Timeout    time.Duration
	Err        error
}

func (e *dispatchWaveDependencyError) Error() string {
	if e.Kind == "timeout" {
		return fmt.Sprintf("%s timed out after %s", e.Dependency, e.Timeout)
	}
	return fmt.Sprintf("%s failed: %v", e.Dependency, e.Err)
}

func (e *dispatchWaveDependencyError) Unwrap() error { return e.Err }

func dispatchWaveDependency[T any](timeout time.Duration, name string, run func() (T, error)) (T, error) {
	return dispatchWaveDependencyRetry(timeout, name, 1, nil, run)
}

func dispatchWaveDependencyRetry[T any](timeout time.Duration, name string, maxAttempts int, retry func(error) bool, run func() (T, error)) (T, error) {
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		type result struct {
			value T
			err   error
		}
		done := make(chan result, 1)
		go func() {
			value, err := run()
			done <- result{value: value, err: err}
		}()
		timer := time.NewTimer(timeout)
		select {
		case got := <-done:
			timer.Stop()
			if got.err == nil {
				return got.value, nil
			}
			canRetry := retry != nil && retry(got.err)
			if canRetry && attempt < maxAttempts {
				continue
			}
			var zero T
			return zero, &dispatchWaveDependencyError{Dependency: name, Kind: "upstream", Attempts: attempt, Retryable: canRetry, Err: got.err}
		case <-timer.C:
			// The timed-out call may still be unwinding. Do not overlap it with an automatic
			// retry unless the dependency accepts cancellation; expose retryability instead.
			var zero T
			return zero, &dispatchWaveDependencyError{Dependency: name, Kind: "timeout", Attempts: attempt, Retryable: true, Timeout: timeout, Err: context.DeadlineExceeded}
		}
	}
	panic("unreachable")
}

func dispatchWaveRouteIssuesBounded(root string, stderr io.Writer, timeout time.Duration) (dispatchtick.RouterPayload, error) {
	routeIssues := dispatchRouteIssues
	return dispatchWaveDependencyRetry(timeout, "issue-contract discovery", 2, func(error) bool { return true }, func() (dispatchtick.RouterPayload, error) {
		return routeIssues(root, stderr)
	})
}

func dispatchWaveRecordDependencyError(rec map[string]any, err error) {
	var dep *dispatchWaveDependencyError
	if !errors.As(err, &dep) {
		rec["stop_reason"] = err.Error()
		rec["failure_class"] = "internal"
		rec["retryable"] = false
		return
	}
	rec["stop_reason"] = dep.Error()
	rec["failure_class"] = dep.Kind
	rec["dependency"] = dep.Dependency
	rec["attempts"] = dep.Attempts
	rec["retryable"] = dep.Retryable
	if dep.Err != nil && dep.Kind != "timeout" {
		rec["cause"] = dep.Err.Error()
	}
	if dep.Retryable {
		rec["retry_disposition"] = "safe_to_retry"
	}
	if dep.Timeout > 0 {
		rec["timeout_ms"] = dep.Timeout.Milliseconds()
	}
}

func auditDispatchWaveExecutionPlanBounded(root string, maxWorkers int, exclude []string, plan []dispatchWaveExecutionPlan, codexLoopGate string, codexLoopGateSinceHours float64, codexLoopGateLimit int, timeout time.Duration) ([]dispatchWaveExecutionAudit, error) {
	return dispatchWaveDependency(timeout, "prelaunch contract audit", func() ([]dispatchWaveExecutionAudit, error) {
		return auditDispatchWaveExecutionPlan(root, maxWorkers, exclude, plan, codexLoopGate, codexLoopGateSinceHours, codexLoopGateLimit), nil
	})
}

func auditDispatchWaveExecutionPlan(root string, maxWorkers int, exclude []string, plan []dispatchWaveExecutionPlan, codexLoopGate string, codexLoopGateSinceHours float64, codexLoopGateLimit int) []dispatchWaveExecutionAudit {
	if len(plan) == 0 {
		return nil
	}
	// Every prelaunch decider observes the same discovery source. The keyed registry opens
	// one upstream watch for the wave and tears it down after the last decider drops.
	subs := subscribeDispatchWaveDiscovery(root, len(plan))
	defer closeDispatchDiscoverySubscriptions(subs)
	out := make([]dispatchWaveExecutionAudit, 0, len(plan))
	for i, row := range plan {
		snapshot := <-subs[i].Snapshots
		payload, err := evaluateDispatchTick(dispatchWaveExecutionTickOptions(root, maxWorkers, exclude, row, false, false, codexLoopGate, codexLoopGateSinceHours, codexLoopGateLimit, snapshot), io.Discard)
		audit := dispatchWaveExecutionAudit{
			Rank:    row.Rank,
			Target:  row.Target,
			Account: row.Account,
		}
		if err != nil {
			audit.Error = err.Error()
			out = append(out, audit)
			continue
		}
		audit.OK = dispatchMapBool(payload, "ok")
		audit.Action = dispatchMapString(payload, "action")
		audit.Verdict = dispatchMapString(payload, "verdict")
		audit.Reason = dispatchMapString(payload, "reason")
		audit.TargetIssue = payload["target_issue"]
		audit.LeaseID = dispatchMapString(payload, "lease_id")
		audit.LeaseTree = stringSlice(payload["lease_tree"])
		audit.Guarded = dispatchMapBool(payload, "guarded")
		audit.LaunchCommand = stringSlice(payload["launch_command"])
		out = append(out, audit)
	}
	return out
}

func dispatchWaveExecutionTickOptions(root string, maxWorkers int, exclude []string, row dispatchWaveExecutionPlan, live bool, refresh bool, codexLoopGate string, codexLoopGateSinceHours float64, codexLoopGateLimit int, discovery ...*runsSnapshot) dispatchTickOptions {
	acct := dispatchWaveAccountFromPlan(row.Account)
	mem := dispatchtick.Membership{
		Rank:      row.Rank,
		WaveID:    row.WaveID,
		Size:      row.WaveSize,
		Shortfall: row.Shortfall,
	}
	opts := dispatchTickOptions{
		Workspace:               root,
		MaxWorkers:              maxWorkers,
		WorkKind:                row.WorkKind,
		Lane:                    row.Target.Lane,
		TargetIssue:             row.Target.Issue,
		LeaseID:                 row.Target.LeaseID,
		LeaseTree:               append([]string(nil), row.Target.Tree...),
		Backend:                 row.Backend,
		ExcludeLanes:            append([]string(nil), exclude...),
		Live:                    live,
		Refresh:                 refresh,
		CooldownMin:             dispatchtick.DefaultCooldownMinutes,
		WorkerTimeoutS:          dispatchtick.DefaultWorkerTimeoutS,
		SpawnProbeS:             dispatchtick.DefaultSpawnProbeS,
		RecordLoop:              live && row.RecordLoop,
		CodexLoopGate:           strings.TrimSpace(codexLoopGate),
		CodexLoopGateSinceHours: codexLoopGateSinceHours,
		CodexLoopGateLimit:      codexLoopGateLimit,
		Account:                 &acct,
		Membership:              &mem,
	}
	if len(discovery) > 0 {
		opts.DiscoverySnapshot = discovery[0]
	}
	return opts
}

func dispatchWaveAccountFromPlan(m map[string]any) dispatchtick.Account {
	return dispatchtick.Account{
		Tag:   dispatchMapString(m, "tag"),
		Tier:  m["tier"],
		Model: dispatchMapString(m, "model"),
		Dir:   dispatchMapString(m, "dir"),
	}
}

func dispatchWavePrelaunchGateFromAudit(executionPlanID string, rows []dispatchWaveExecutionAudit) dispatchWavePrelaunchGate {
	gate := dispatchWavePrelaunchGate{
		OK:              true,
		Action:          "LAUNCH",
		ExecutionPlanID: executionPlanID,
		TargetCount:     len(rows),
	}
	for _, row := range rows {
		if row.Error != "" {
			gate.OK = false
			gate.ErrorCount++
			gate.RefusedCount++
			gate.Refused = append(gate.Refused, dispatchWavePrelaunchRefusal{
				Rank:   row.Rank,
				Target: firstString(row.Target.ID, "target"),
				Error:  row.Error,
				Reason: "audit errored: " + row.Error,
			})
			continue
		}
		if !row.OK {
			reason := firstString(row.Reason, row.Verdict, row.Action, "not ok")
			gate.OK = false
			gate.RefusedCount++
			gate.Refused = append(gate.Refused, dispatchWavePrelaunchRefusal{
				Rank:    row.Rank,
				Target:  firstString(row.Target.ID, "target"),
				Action:  row.Action,
				Verdict: row.Verdict,
				Reason:  reason,
			})
			continue
		}
		gate.ReadyCount++
	}
	switch {
	case gate.ErrorCount > 0:
		gate.OK = false
		gate.Action = "HOLD_ERROR"
		first := gate.Refused[0]
		gate.Reason = strings.TrimSpace(first.Target + " " + firstString(first.Error, first.Reason, first.Verdict, first.Action, "audit error"))
	case gate.ReadyCount > 0:
		gate.OK = true
		if gate.RefusedCount > 0 {
			gate.Action = "LAUNCH_READY"
			gate.Reason = fmt.Sprintf("launching %d ready target(s); holding %d refused target(s)", gate.ReadyCount, gate.RefusedCount)
		}
	case gate.RefusedCount > 0:
		gate.OK = false
		gate.Action = "HOLD"
		first := gate.Refused[0]
		gate.Reason = strings.TrimSpace(first.Target + " " + firstString(first.Reason, first.Error, first.Verdict, first.Action, "not ok"))
	default:
		gate.OK = false
		gate.Action = "HOLD_EMPTY"
		gate.Reason = "execution audit produced no targets"
	}
	return gate
}

func dispatchWaveReadyExecutionPlan(plan []dispatchWaveExecutionPlan, rows []dispatchWaveExecutionAudit) []dispatchWaveExecutionPlan {
	if len(plan) == 0 || len(rows) == 0 {
		return nil
	}
	ready := map[int]bool{}
	for _, row := range rows {
		if row.OK {
			ready[row.Rank] = true
		}
	}
	out := make([]dispatchWaveExecutionPlan, 0, len(plan))
	for _, row := range plan {
		if ready[row.Rank] {
			out = append(out, row)
		}
	}
	return out
}

func dispatchWavePrelaunchRetryIssues(rows []dispatchWaveExecutionAudit, live bool, attempt, maxAttempts int) []int {
	if attempt >= maxAttempts {
		return nil
	}
	if live {
		return dispatchWaveRetryableAuditIssues(rows)
	}
	// A dry-run may route around transient ownership races, but structural holds must
	// remain visible to the operator rather than disappearing behind another candidate.
	for _, row := range rows {
		if row.OK || row.Error != "" {
			continue
		}
		switch row.Action {
		case "lane_busy", "lane_leased", "in_flight_duplicate", "collision_risk":
		default:
			return nil
		}
	}
	return dispatchWaveRetryableAuditIssues(rows)
}

func dispatchWaveRetryableAuditIssues(rows []dispatchWaveExecutionAudit) []int {
	if len(rows) == 0 {
		return nil
	}
	out := map[int]bool{}
	for _, row := range rows {
		if row.Error != "" || row.OK {
			continue
		}
		if !dispatchTickBenignActions[row.Action] {
			return nil
		}
		if row.Target.Issue <= 0 {
			return nil
		}
		out[row.Target.Issue] = true
	}
	return sortedIntSet(out)
}

func dispatchWaveAllocationCount(requested int, preflight map[string]any) int {
	if requested <= 0 {
		return 0
	}
	limit := requested
	if n := intPtrFromAny(preflight["headroom"]); n != nil {
		if *n < limit {
			limit = *n
		}
	}
	if seat, ok := preflight["seat"].(map[string]any); ok {
		if n := intPtrFromAny(seat["free"]); n != nil && *n < limit {
			limit = *n
		}
	}
	if limit < 0 {
		return 0
	}
	return limit
}

func appendDispatchWavePrelaunchRetry(existing any, attempt int, issues []int, gate dispatchWavePrelaunchGate) []any {
	out, _ := existing.([]any)
	out = append(out, map[string]any{
		"attempt":       attempt,
		"held_issues":   append([]int(nil), issues...),
		"gate_action":   gate.Action,
		"refused_count": gate.RefusedCount,
		"reason":        gate.Reason,
	})
	return out
}

func sortedIntSet(set map[int]bool) []int {
	if len(set) == 0 {
		return nil
	}
	out := make([]int, 0, len(set))
	for n := range set {
		out = append(out, n)
	}
	sort.Ints(out)
	return out
}

func dispatchWaveExecutionPlanID(plan []dispatchWaveExecutionPlan) string {
	if len(plan) == 0 {
		return ""
	}
	return dispatchStablePlanID(plan)
}

func dispatchWaveExecutionPlans(root, backend, workKind, goal, goalProfile, waveID string, shortfall int, targets []dispatchWaveCandidate, lanes []dispatchtick.AccountWaveLane, recordLoop bool, codexLoopGate string, codexLoopGateSinceHours float64, codexLoopGateLimit int) []dispatchWaveExecutionPlan {
	limit := minInt(len(targets), len(lanes))
	if limit <= 0 {
		return nil
	}
	out := make([]dispatchWaveExecutionPlan, 0, limit)
	for i := 0; i < limit; i++ {
		target := targets[i]
		acct := accountFromWaveLane(lanes[i])
		mem := dispatchtick.Membership{Rank: i, WaveID: waveID, Size: limit, Shortfall: shortfall}
		args := dispatchWaveExecutionTickArgs(root, backend, workKind, goal, goalProfile, target, acct, mem, recordLoop, codexLoopGate, codexLoopGateSinceHours, codexLoopGateLimit)
		out = append(out, dispatchWaveExecutionPlan{
			Rank:                i,
			WaveID:              waveID,
			WaveSize:            limit,
			Shortfall:           shortfall,
			Backend:             backend,
			WorkKind:            workKind,
			Goal:                goal,
			GoalProfile:         goalProfile,
			Target:              dispatchWaveLaunchTarget(target),
			Account:             dispatchtick.AccountSidecar(acct),
			RecordLoop:          recordLoop,
			DispatchTickArgs:    args,
			DispatchTickCommand: append([]string{"fak", "dispatch", "tick"}, args...),
		})
	}
	return out
}

func dispatchWaveExecutionTickArgs(root, backend, workKind, goal, goalProfile string, target dispatchWaveCandidate, account dispatchtick.Account, membership dispatchtick.Membership, recordLoop bool, codexLoopGate string, codexLoopGateSinceHours float64, codexLoopGateLimit int) []string {
	args := []string{"--workspace", root, "--backend", backend}
	if strings.TrimSpace(workKind) != "" {
		args = append(args, "--work-kind", workKind)
	}
	if strings.TrimSpace(goal) != "" {
		args = append(args, "--goal", strings.TrimSpace(goal))
	}
	if strings.TrimSpace(goalProfile) != "" && strings.TrimSpace(goalProfile) != dispatchGoalProfileThroughput {
		args = append(args, "--goal-profile", strings.TrimSpace(goalProfile))
	}
	args = append(args, dispatchTickArgsForLaunchTarget(target)...)
	if !recordLoop {
		args = append(args, "--no-loop-ledger")
	}
	if backend == "codex" {
		args = append(args,
			"--codex-loop-gate", firstString(strings.TrimSpace(codexLoopGate), dispatchCodexLoopGateDefaultThreshold()),
			"--codex-loop-gate-since-hours", fmt.Sprint(codexLoopGateSinceHours),
			"--codex-loop-gate-limit", fmt.Sprint(codexLoopGateLimit),
		)
	}
	if account.Tag != "" {
		args = append(args, "--account-tag", account.Tag)
	}
	if account.Tier != nil {
		args = append(args, "--account-tier", fmt.Sprint(account.Tier))
	}
	if account.Model != "" {
		args = append(args, "--account-model", account.Model)
	}
	if account.Dir != "" {
		args = append(args, "--account-dir", account.Dir)
	}
	if membership.WaveID != "" {
		args = append(args,
			"--wave-id", membership.WaveID,
			"--wave-rank", fmt.Sprint(membership.Rank),
			"--wave-size", fmt.Sprint(membership.Size),
			"--wave-shortfall", fmt.Sprint(membership.Shortfall),
		)
	}
	return args
}

func dispatchWaveLaneSerialWaveCount(candidates []dispatchWaveCandidate) int {
	if len(candidates) == 0 {
		return 0
	}
	return dispatchLaneSerialWaveCount(dispatchCandidateKeys(candidates,
		func(cand dispatchWaveCandidate) string {
			return dispatchLaneSerialKey(cand.Lane, cand.LeaseID, cand.ID)
		}))
}

func dispatchWaveRouteStepBudget(route dispatchtick.IssueRoute) int {
	if route.ExpectedSteps > 0 {
		return route.ExpectedSteps
	}
	return 1
}

func dispatchWaveLaneStepBudget(grp dispatchtick.RouterLaneGroup) int {
	if grp.StepBudget > 0 {
		return grp.StepBudget
	}
	if grp.Count > 0 {
		return grp.Count
	}
	return len(grp.Issues)
}

func dispatchWaveStepBudget(candidates []dispatchWaveCandidate) int {
	total := 0
	for _, cand := range candidates {
		if cand.StepBudget > 0 {
			total += cand.StepBudget
		} else {
			total++
		}
	}
	return total
}

func dispatchWavePct(n, d int) int {
	if d <= 0 {
		return 0
	}
	return int(float64(n)*100/float64(d) + 0.5)
}

func waveCandidateID(lane string, issue int) string {
	if issue > 0 {
		return fmt.Sprintf("%s#%d", lane, issue)
	}
	return lane
}

func dispatchLaneLeaseID(lane string) string {
	return "resolve-" + cleanDispatchLeaseToken(lane)
}

func dispatchIssueLeaseID(lane string, issue int) string {
	return fmt.Sprintf("resolve-%s-%d", cleanDispatchLeaseToken(lane), issue)
}

func cleanDispatchLeaseToken(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "unknown"
	}
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-.")
}

func firstLaunchableIssue(nums []int, live, cooled, excluded map[int]bool) (int, bool) {
	for _, n := range nums {
		if !live[n] && !cooled[n] && !excluded[n] {
			return n, true
		}
	}
	return 0, false
}

func writeDispatchWaveResult(stdout, stderr io.Writer, rec map[string]any, asJSON bool) int {
	dispatchWaveAnnotateOutcome(rec)
	if asJSON {
		if err := writeIndentedJSON(stdout, rec); err != nil {
			fmt.Fprintf(stderr, "fak dispatch wave: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderDispatchWave(rec))
	}
	if dispatchMapBool(rec, "ok") {
		return 0
	}
	if dispatchWaveExitBenign(rec) {
		return 0
	}
	return 1
}

func dispatchWaveAnnotateOutcome(rec map[string]any) {
	if rec == nil {
		return
	}
	if gate, ok := rec["prelaunch_gate"].(dispatchWavePrelaunchGate); ok {
		rec["approval_ready"] = gate.OK
		rec["approval_action"] = gate.Action
		rec["approval_verdict"] = dispatchWavePrelaunchVerdict(gate)
	}
	verdict, action := dispatchWaveOutcome(rec)
	rec["verdict"] = verdict
	rec["action"] = action
	if _, ok := rec["approval_ready"]; !ok {
		rec["approval_ready"] = verdict == "WOULD_WAVE" || verdict == "WAVED"
	}
	if _, ok := rec["approval_action"]; !ok {
		rec["approval_action"] = action
	}
	if _, ok := rec["approval_verdict"]; !ok {
		rec["approval_verdict"] = verdict
	}
}

func dispatchWaveOutcome(rec map[string]any) (string, string) {
	if dispatchMapBool(rec, "live") && dispatchMapInt(rec, "spawned") > 0 {
		return "WAVED", "waved"
	}
	if gate, ok := rec["prelaunch_gate"].(dispatchWavePrelaunchGate); ok {
		if gate.OK {
			return "WOULD_WAVE", "would_wave"
		}
		return dispatchWavePrelaunchVerdict(gate), "hold"
	}
	stop := dispatchMapString(rec, "stop_reason")
	switch {
	case dispatchMapString(rec, "failure_class") == "timeout":
		return "WAVE_DEPENDENCY_TIMEOUT", "retryable_error"
	case dispatchMapString(rec, "failure_class") == "upstream":
		return "WAVE_DEPENDENCY_ERROR", "error"
	case dispatchMapString(rec, "failure_class") == "internal":
		return "WAVE_INTERNAL_ERROR", "error"
	case stop == "preflight headroom exhausted before account allocation":
		if pre, ok := rec["preflight"].(map[string]any); ok {
			if verdict := dispatchMapString(pre, "verdict"); verdict != "" {
				return verdict, "refused"
			}
		}
		return "REFUSE_AT_CAP", "refused"
	case dispatchMapInt(rec, "allocation_requested") > 0 && dispatchMapInt(rec, "granted") == 0:
		return "WAVE_NO_SEATS", "no_seats"
	case stop == "priced fan-out found no launchable lane":
		return "WAVE_NO_LANE", "no_lane"
	case strings.HasPrefix(stop, "price fan-out:"):
		return "WAVE_PRICE_ERROR", "error"
	case !dispatchMapBool(rec, "live") && dispatchMapInt(rec, "granted") > 0:
		return "WOULD_WAVE", "would_wave"
	default:
		return "WAVE_EMPTY", "refused"
	}
}

func dispatchWavePrelaunchVerdict(gate dispatchWavePrelaunchGate) string {
	if gate.OK {
		return "WOULD_WAVE"
	}
	for _, row := range gate.Refused {
		if strings.TrimSpace(row.Verdict) != "" {
			return strings.TrimSpace(row.Verdict)
		}
		if strings.TrimSpace(row.Action) != "" {
			return strings.ToUpper(strings.TrimSpace(row.Action))
		}
		if row.Error != "" {
			return "WAVE_AUDIT_ERROR"
		}
	}
	if strings.TrimSpace(gate.Action) != "" {
		return strings.TrimSpace(gate.Action)
	}
	return "WAVE_HELD"
}

func dispatchWaveExitBenign(rec map[string]any) bool {
	if rec == nil {
		return false
	}
	if strings.HasPrefix(dispatchMapString(rec, "stop_reason"), "price fan-out:") {
		return false
	}
	gate, ok := rec["prelaunch_gate"].(dispatchWavePrelaunchGate)
	if ok {
		if gate.ErrorCount > 0 || gate.Action == "HOLD_ERROR" {
			return false
		}
		if gate.Action == "HOLD" || gate.Action == "HOLD_EMPTY" || gate.Action == "LAUNCH_READY" {
			for _, refusal := range gate.Refused {
				if refusal.Error != "" {
					return false
				}
				if refusal.Action != "" && !dispatchTickBenignActions[refusal.Action] {
					return false
				}
			}
			return true
		}
	}
	verdict, action := dispatchWaveOutcome(rec)
	if verdict == "WAVE_NO_SEATS" || action == "no_seats" {
		return true
	}
	switch dispatchMapString(rec, "stop_reason") {
	case "priced fan-out found no launchable lane", "filled requested wave", "preflight headroom exhausted before account allocation":
		return true
	}
	return false
}

func renderDispatchWave(rec map[string]any) string {
	var b strings.Builder
	mode := "dry-run"
	if dispatchMapBool(rec, "live") {
		mode = "live"
	}
	fmt.Fprintf(&b, "issue-dispatch-wave: %s  verdict=%s action=%s requested=%d granted=%d spawned=%d backend=%s\n",
		mode, dispatchMapString(rec, "verdict"), dispatchMapString(rec, "action"),
		dispatchMapInt(rec, "requested"), dispatchMapInt(rec, "granted"),
		dispatchMapInt(rec, "spawned"), dispatchMapString(rec, "backend"))
	if id := dispatchMapString(rec, "wave_id"); id != "" {
		fmt.Fprintf(&b, "  wave_id: %s\n", id)
	}
	if id := dispatchMapString(rec, "execution_plan_id"); id != "" {
		fmt.Fprintf(&b, "  execution_plan_id: %s\n", id)
	}
	if _, ok := rec["approval_ready"]; ok {
		fmt.Fprintf(&b, "  approval: ready=%t verdict=%s action=%s\n",
			dispatchMapBool(rec, "approval_ready"),
			dispatchMapString(rec, "approval_verdict"),
			dispatchMapString(rec, "approval_action"))
	}
	if reason := dispatchMapString(rec, "stop_reason"); reason != "" {
		fmt.Fprintf(&b, "  stop: %s\n", reason)
	}
	if gate, ok := rec["prelaunch_gate"].(dispatchWavePrelaunchGate); ok {
		fmt.Fprintf(&b, "  prelaunch_gate: action=%s ready=%d refused=%d errors=%d target_count=%d\n",
			gate.Action, gate.ReadyCount, gate.RefusedCount, gate.ErrorCount, gate.TargetCount)
		if gate.Reason != "" {
			fmt.Fprintf(&b, "    reason=%s\n", gate.Reason)
		}
	}
	if plan, ok := rec["execution_plan"].([]dispatchWaveExecutionPlan); ok && len(plan) > 0 {
		fmt.Fprintln(&b, "  execution_plan:")
		renderDispatchWavePlanRows(&b, plan)
	}
	if ready, ok := rec["ready_execution_plan"].([]dispatchWaveExecutionPlan); ok && len(ready) > 0 {
		if full, _ := rec["execution_plan"].([]dispatchWaveExecutionPlan); len(ready) != len(full) {
			fmt.Fprintln(&b, "  ready_execution_plan:")
			renderDispatchWavePlanRows(&b, ready)
		}
	}
	if price, ok := rec["price"].(dispatchWavePrice); ok {
		fmt.Fprintf(&b, "  priced fan-out: action=%s run=%s effective_cap=%d fresh_starts=%d/%d run_steps=%d candidate_steps=%d collisions_avoided=%d lanes_utilized=%d serialization_wasted=%d safe_concurrency=%d (%d%%) scope=%d%% same_lane_parallelism=%d repartition=%d\n",
			price.Action,
			strings.Join(price.RunLanes, ","), price.EffectiveCap, price.FreshStarts, price.FreshStartCap, price.RunStepBudget, price.CandidateStepBudget, price.CollisionsAvoided, price.LanesUtilized,
			price.SerializationWasted, price.SafeConcurrency, price.SafeConcurrencyPct,
			price.ScopeCoveragePct, price.SameLaneParallelism, len(price.Repartition))
		if len(price.RunTargets) > 0 {
			fmt.Fprintln(&b, "  selected_targets:")
			for _, target := range price.RunTargets {
				fmt.Fprintf(&b, "    rank=%d issue=%s lane=%s lease=%s scope=%s steps=%d reason=%s\n",
					target.Rank, dispatchWaveIssueLabel(target), target.Lane, target.LeaseID,
					dispatchWaveScopeLabel(target), target.StepBudget, target.Reason)
			}
		}
		if skipped := dispatchWaveSkippedCandidates(price.Candidates); len(skipped) > 0 {
			fmt.Fprintln(&b, "  skipped_candidates:")
			for _, cand := range skipped {
				fmt.Fprintf(&b, "    rank=%d issue=%s lane=%s disposition=%s reason=%s collides=%s\n",
					cand.Rank, dispatchWaveIssueLabel(cand), cand.Lane, cand.Disposition, cand.Reason,
					dispatchWaveCollisionLabel(cand.CollidesWith))
			}
		}
	}
	if !dispatchMapBool(rec, "live") {
		if _, ok := rec["approval_ready"]; ok && !dispatchMapBool(rec, "approval_ready") {
			fmt.Fprintln(&b, "  (dry-run held - resolve the refusal before using --live)")
		} else {
			fmt.Fprintln(&b, "  (dry-run - re-run with --live to spawn the wave)")
		}
	}
	return b.String()
}
