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
	"strconv"
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
	RequestedIssues      []int                             `json:"requested_issues,omitempty"`
	SelectedIssues       []int                             `json:"selected_issues,omitempty"`
	RefusedIssues        []dispatchWaveIssueRefusal        `json:"refused_issues,omitempty"`
}

const (
	dispatchWaveReasonWaveCap     = "wave-cap"
	dispatchWaveReasonFreshWIPCap = "fresh-wip-cap"
	dispatchWaveDefaultFreshCap   = 1

	dispatchWaveIssueRefusalRouting     = "routing"
	dispatchWaveIssueRefusalEligibility = "eligibility"
	dispatchWaveIssueRefusalCapacity    = "capacity"
	dispatchWaveIssueRefusalIntent      = "intent"

	dispatchWaveReasonIssueUnroutable = "ISSUE_UNROUTABLE"
	dispatchWaveReasonCapacity        = "WAVE_CAPACITY"
	dispatchWaveReasonIssueInFlight   = "ISSUE_IN_FLIGHT"
	dispatchWaveReasonIssueCooldown   = "ISSUE_COOLDOWN"
	dispatchWaveReasonIssueIneligible = "ISSUE_INELIGIBLE"
	dispatchWaveReasonLaneMismatch    = "ISSUE_LANE_MISMATCH"
	dispatchWaveReasonLaneUnavailable = "ISSUE_LANE_UNAVAILABLE"
)

type dispatchWaveIssueRefusal struct {
	Issue  int    `json:"issue"`
	Class  string `json:"class"`
	Reason string `json:"reason"`
	Detail string `json:"detail,omitempty"`
}

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

type dispatchWaveTrancheReceipt struct {
	RequestedCount        int      `json:"requested_count"`
	AttemptedTrancheSizes []int    `json:"attempted_tranche_sizes"`
	HeldCandidates        []string `json:"held_candidates,omitempty"`
	TimeoutSource         string   `json:"timeout_source,omitempty"`
	FinalAuditedCount     int      `json:"final_audited_count"`
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
	finishFirstOverride := fs.Bool("finish-first-override", false, "explicitly admit the configured fresh-start cap despite a finish-first progress hold")
	backend := fs.String("backend", "codex", "worker backend (claude|opencode|codex); default codex")
	workKind := fs.String("work-kind", "", "switcher work kind (default follows --backend)")
	goal := fs.String("goal", "", "durable dispatch loop goal id (for example throughput or high-priority); forwarded to each tick")
	goalProfile := fs.String("goal-profile", "", "dispatch picker profile: throughput|high-priority (default follows --goal, else throughput)")
	lane := fs.String("lane", "", "pin every tick to this repo lane (default: largest step-budget lane pick)")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the step-budget pick")
	var issueFlags repeatedString
	fs.Var(&issueFlags, "issue", "bind the wave to an issue number; repeatable and comma-separated (never substitutes general backlog work)")
	settleS := fs.Float64("settle-s", 2.0, "seconds to wait after each live spawn")
	noLedger := fs.Bool("no-loop-ledger", false, "disable loop-ledger append for spawned ticks")
	codexLoopGate := fs.String("codex-loop-gate", dispatchCodexLoopGateDefaultThreshold(), "for live Codex workers, opt in to a pre-spawn audit of recent Codex sessions and refuse at threshold loop|action, or use off (default: $FLEET_CODEX_LOOP_GATE, else off)")
	codexLoopGateSinceHours := fs.Float64("codex-loop-gate-since-hours", dispatchCodexLoopGateDefaultSinceHoursValue(), "with --codex-loop-gate, only scan Codex sessions modified within N hours (0 = all)")
	codexLoopGateLimit := fs.Int("codex-loop-gate-limit", dispatchCodexLoopGateDefaultLimitValue(), "with --codex-loop-gate, maximum newest Codex sessions to scan")
	live := fs.Bool("live", false, "actually spawn workers")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	requestedIssues, issueErr := parseDispatchWaveIssueNumbers(issueFlags)
	if issueErr != nil {
		fmt.Fprintf(stderr, "fak dispatch wave: --issue: %v\n", issueErr)
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

	preflightResult, preflightErr := dispatchWaveDependencyRetry(dispatchWaveDependencyTimeout, "dispatch preflight", 2, func(error) bool { return true }, func(ctx context.Context) (dispatchWavePreflightResult, error) {
		product, allocationCount, shortfall, preflight, err := dispatchWavePreflightAlloc(ctx, root, stderr, *maxWorkers, wk, backendNorm, *count)
		return dispatchWavePreflightResult{Product: product, AllocationCount: allocationCount, Shortfall: shortfall, Payload: preflight}, err
	})
	if preflightErr != nil {
		rec := newDispatchWaveRecord(root, *live, backendNorm, wk, goalID, profile, *count, 0, nil)
		dispatchWaveSeedExplicitIssueReceipt(rec, requestedIssues)
		rec["granted"] = 0
		rec["shortfall"] = *count
		dispatchWaveRecordDependencyError(rec, preflightErr)
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}
	product, allocationCount, preflightShortfall, preflight := preflightResult.Product, preflightResult.AllocationCount, preflightResult.Shortfall, preflightResult.Payload
	finishFirstAdmission := loadDispatchFinishFirstAdmission(root, *freshStartCap, allocationCount, *finishFirstOverride)
	*freshStartCap = finishFirstAdmission.AllowedFreshStarts
	if preflight != nil {
		preflight["finish_first_admission"] = finishFirstAdmission
	}

	rec := newDispatchWaveRecord(root, *live, backendNorm, wk, goalID, profile, *count, allocationCount, preflight)
	rec["finish_first_admission"] = finishFirstAdmission
	dispatchWaveSeedExplicitIssueReceipt(rec, requestedIssues)
	if allocationCount <= 0 {
		rec["granted"] = 0
		rec["shortfall"] = *count
		rec["stop_reason"] = "preflight headroom exhausted before account allocation"
		dispatchWaveRefuseAllExplicitIssues(rec, requestedIssues, dispatchWaveIssueRefusalCapacity, dispatchWaveReasonCapacity, "preflight capacity admitted no worker seats")
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
		dispatchWaveRefuseAllExplicitIssues(rec, requestedIssues, dispatchWaveIssueRefusalCapacity, dispatchWaveReasonCapacity, rec["stop_reason"].(string))
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}

	excludedLanes := splitCommaList(*excludeLane)
	router, err := dispatchWaveRouteIssuesBounded(root, stderr, dispatchWaveDependencyTimeout)
	if err != nil {
		dispatchWaveRecordDependencyError(rec, err)
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}
	intentHolds, err := dispatchWaveReadIntentHolds(root, requestedIssues)
	if err != nil {
		dispatchWaveRecordDependencyError(rec, fmt.Errorf("read live issue intents: %w", err))
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}
	executionPlan, code, done := planDispatchWave(stdout, stderr, dispatchWavePlanRequest{
		root: root, backend: backendNorm, workKind: wk, goalID: goalID, profile: profile,
		waveID: waveID, shortfall: shortfall, router: router, requestedIssues: requestedIssues,
		count: count, lanes: lanes, lane: lane, excludedLanes: excludedLanes,
		freshStartCap: freshStartCap, maxWorkers: maxWorkers, intentHolds: intentHolds,
		noLedger: noLedger, codexLoopGate: codexLoopGate, gateSinceHours: codexLoopGateSinceHours,
		gateLimit: codexLoopGateLimit, live: live, asJSON: asJSON, record: rec,
	})
	if done {
		return code
	}
	return executeDispatchWavePlan(stdout, stderr, dispatchWaveExecutionRequest{
		root: root, plan: executionPlan, maxWorkers: maxWorkers, excludeLane: excludeLane,
		live: live, settleSeconds: settleS, codexLoopGate: codexLoopGate,
		gateSinceHours: codexLoopGateSinceHours, gateLimit: codexLoopGateLimit,
		asJSON: asJSON, record: rec,
	})
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
func dispatchWavePreflightAlloc(ctx context.Context, root string, stderr io.Writer, maxWorkers int, wk, backendNorm string, count int) (string, int, int, map[string]any, error) {
	product := dispatchtick.ProductForBackend(backendNorm)
	preflight, err := dispatchPreflightContext(ctx, root, stderr, maxWorkers, wk, product)
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
	return priceDispatchWavePayloadBound(root, router, requested, granted, explicitLane, excluded, cooldownMin, excludedIssues, freshStartCap, nil, nil, goalProfile...)
}

// priceDispatchWaveExplicitIssues is the explicit-target planner seam. Supplying a set
// binds candidate construction to those identities; every member either reaches
// SelectedIssues or gets one typed refusal, and no general-backlog issue can enter.
func priceDispatchWaveExplicitIssues(root string, router dispatchtick.RouterPayload, requestedIssues []int, requested, granted int, explicitLane string, excluded []string, cooldownMin int, excludedIssues map[int]bool, freshStartCap int, intentHolds map[int]string, goalProfile ...string) (dispatchWavePrice, error) {
	return priceDispatchWavePayloadBound(root, router, requested, granted, explicitLane, excluded, cooldownMin, excludedIssues, freshStartCap, requestedIssues, intentHolds, goalProfile...)
}

func priceDispatchWavePayloadBound(root string, router dispatchtick.RouterPayload, requested, granted int, explicitLane string, excluded []string, cooldownMin int, excludedIssues map[int]bool, freshStartCap int, requestedIssues []int, intentHolds map[int]string, goalProfile ...string) (dispatchWavePrice, error) {
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	profile := dispatchWaveGoalProfile(goalProfile)
	requestedSet := map[int]bool{}
	for _, issue := range requestedIssues {
		requestedSet[issue] = true
	}
	explicitIssues := len(requestedSet) > 0
	seenRoutes := map[int]bool{}
	refused := map[int]dispatchWaveIssueRefusal{}
	refuse := func(issue int, class, reason, detail string) {
		if issue <= 0 || !explicitIssues || refused[issue].Issue != 0 {
			return
		}
		refused[issue] = dispatchWaveIssueRefusal{Issue: issue, Class: class, Reason: reason, Detail: strings.TrimSpace(detail)}
	}
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

	collector := dispatchWaveCandidateCollector{
		root: root, router: router, explicitIssues: explicitIssues, requestedSet: requestedSet,
		seenRoutes: seenRoutes, refuse: refuse, exclude: exclude, intentHolds: intentHolds,
		liveIssues: liveIssues, cooled: cooled, skipIssues: skipIssues,
		newlyUnblocked: newlyUnblocked, readyState: readyState, snapshot: snap,
		profile: profile, lanes: lanes, issueByLane: map[string]int{},
		metadata: map[string]dispatchWaveCandidate{}, candidates: []dispatchorder.Candidate{},
		unscopedByLane: map[string][]int{}, scopedByLane: map[string]bool{},
		explicitLane: explicitLane,
	}
	collector.addScopedRoutes()
	collector.addLaneFallbacks()
	issueByLane, meta, cands := collector.issueByLane, collector.metadata, collector.candidates

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
	if explicitIssues {
		price.RequestedIssues = append([]int(nil), requestedIssues...)
		for _, target := range price.RunTargets {
			price.SelectedIssues = append(price.SelectedIssues, target.Issue)
		}
		selectedSet := dispatchWaveIntSet(price.SelectedIssues)
		pricedRowForIssue := map[int]dispatchWaveCandidate{}
		for _, cand := range price.Candidates {
			pricedRowForIssue[cand.Issue] = cand
		}
		for _, issue := range requestedIssues {
			if selectedSet[issue] || refused[issue].Issue != 0 {
				continue
			}
			if cand, ok := pricedRowForIssue[issue]; ok {
				class, reason := dispatchWaveIssueRefusalEligibility, firstString(cand.Reason, dispatchWaveReasonIssueIneligible)
				if cand.Reason == dispatchWaveReasonWaveCap || cand.Reason == dispatchWaveReasonFreshWIPCap {
					class, reason = dispatchWaveIssueRefusalCapacity, dispatchWaveReasonCapacity
				}
				refuse(issue, class, reason, cand.Reason)
				continue
			}
			if seenRoutes[issue] {
				refuse(issue, dispatchWaveIssueRefusalCapacity, dispatchWaveReasonCapacity, "the issue could not fit the bounded lane candidate set")
				continue
			}
			detail := "issue is absent from the routed candidate set"
			for _, row := range router.UnroutableBacklog {
				if row.Number == issue {
					detail = firstString(row.Reason, row.NextAction, detail)
					break
				}
			}
			refuse(issue, dispatchWaveIssueRefusalRouting, dispatchWaveReasonIssueUnroutable, detail)
		}
		price.RefusedIssues = dispatchWaveRefusalRows(requestedIssues, refused)
	}
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

// Wave dependencies and standalone tick preflight share this budget. The wave passes its
// context through preflight so the committed-tree build cannot outlive this deadline.
const dispatchWaveDependencyTimeout = dispatchPreflightTimeout

// A context-aware dependency gets one short, bounded interval to publish its fail-open
// result after cancellation. Without it the wrapper races the tree-build probe's unwind
// and can manufacture WAVE_DEPENDENCY_TIMEOUT even though the probe honored the deadline.
const dispatchWaveDependencyCancelGrace = 250 * time.Millisecond

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

func dispatchWaveDependency[T any](timeout time.Duration, name string, run func(context.Context) (T, error)) (T, error) {
	return dispatchWaveDependencyRetry(timeout, name, 1, nil, run)
}

func waitDispatchWaveSnapshot(ctx context.Context, ch <-chan *runsSnapshot) (*runsSnapshot, error) {
	select {
	case snap, ok := <-ch:
		if !ok {
			return nil, io.EOF
		}
		return snap, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
func dispatchWaveDependencyRetry[T any](timeout time.Duration, name string, maxAttempts int, retry func(error) bool, run func(context.Context) (T, error)) (T, error) {

	if maxAttempts < 1 {
		maxAttempts = 1
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if ctx.Err() != nil {
			var zero T
			return zero, &dispatchWaveDependencyError{Dependency: name, Kind: "timeout", Attempts: attempt, Retryable: true, Timeout: timeout, Err: ctx.Err()}
		}
		type result struct {
			value T
			err   error
		}
		done := make(chan result, 1)
		go func() {
			value, err := run(ctx)
			done <- result{value: value, err: err}
		}()
		select {
		case got := <-done:
			if got.err == nil {
				return got.value, nil
			}
			if ctx.Err() != nil && errors.Is(got.err, ctx.Err()) {
				var zero T
				return zero, &dispatchWaveDependencyError{Dependency: name, Kind: "timeout", Attempts: attempt, Retryable: true, Timeout: timeout, Err: ctx.Err()}
			}
			canRetry := retry != nil && retry(got.err)
			if canRetry && attempt < maxAttempts {
				continue
			}
			var zero T
			return zero, &dispatchWaveDependencyError{Dependency: name, Kind: "upstream", Attempts: attempt, Retryable: canRetry, Err: got.err}
		case <-ctx.Done():
			// A cancellation-aware probe may need a scheduler turn to publish its fail-open
			// result. Accept that result without letting a context-ignorant dependency escape
			// the hard bound plus this fixed unwind allowance.
			grace := time.NewTimer(dispatchWaveDependencyCancelGrace)
			select {
			case got := <-done:
				grace.Stop()
				if got.err == nil {
					return got.value, nil
				}
			case <-grace.C:
			}
			var zero T
			return zero, &dispatchWaveDependencyError{Dependency: name, Kind: "timeout", Attempts: attempt, Retryable: true, Timeout: timeout, Err: ctx.Err()}
		}
	}
	panic("unreachable")
}

func dispatchWaveRouteIssuesBounded(root string, stderr io.Writer, timeout time.Duration) (dispatchtick.RouterPayload, error) {
	routeIssues := dispatchRouteIssues
	return dispatchWaveDependencyRetry(timeout, "issue-contract discovery", 2, func(error) bool { return true }, func(context.Context) (dispatchtick.RouterPayload, error) {
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

func dispatchWaveExecutionPlanWithSize(plan []dispatchWaveExecutionPlan, size int) []dispatchWaveExecutionPlan {
	if size >= len(plan) {
		return append([]dispatchWaveExecutionPlan(nil), plan...)
	}
	if size < 0 {
		size = 0
	}
	out := append([]dispatchWaveExecutionPlan(nil), plan[:size]...)
	for i := range out {
		out[i].Rank = i
		out[i].WaveSize = len(out)
		out[i].DispatchTickArgs = dispatchWaveSetIntArg(out[i].DispatchTickArgs, "--wave-rank", i)
		out[i].DispatchTickArgs = dispatchWaveSetIntArg(out[i].DispatchTickArgs, "--wave-size", len(out))
		out[i].DispatchTickCommand = append([]string{"fak", "dispatch", "tick"}, out[i].DispatchTickArgs...)
	}
	return out
}

func dispatchWaveSetIntArg(args []string, name string, value int) []string {
	out := append([]string(nil), args...)
	for i := 0; i+1 < len(out); i++ {
		if out[i] == name {
			out[i+1] = strconv.Itoa(value)
			return out
		}
	}
	return append(out, name, strconv.Itoa(value))
}
func dispatchWaveFallbackSize(size int) int {
	switch {
	case size > 8:
		return 8
	case size > 4:
		return 4
	default:
		return 0
	}
}

func auditDispatchWaveExecutionPlanWithFallback(plan []dispatchWaveExecutionPlan, audit func([]dispatchWaveExecutionPlan) ([]dispatchWaveExecutionAudit, error)) ([]dispatchWaveExecutionAudit, []dispatchWaveExecutionPlan, dispatchWaveTrancheReceipt, error) {
	receipt := dispatchWaveTrancheReceipt{RequestedCount: len(plan)}
	current := append([]dispatchWaveExecutionPlan(nil), plan...)
	for {
		receipt.AttemptedTrancheSizes = append(receipt.AttemptedTrancheSizes, len(current))
		rows, err := audit(current)
		if err == nil {
			receipt.FinalAuditedCount = len(rows)
			return rows, current, receipt, nil
		}
		var dep *dispatchWaveDependencyError
		next := dispatchWaveFallbackSize(len(current))
		if len(receipt.AttemptedTrancheSizes) > 1 || !errors.As(err, &dep) || dep.Kind != "timeout" || dep.Dependency != "prelaunch contract audit" || next == 0 {
			return rows, current, receipt, err
		}
		receipt.TimeoutSource = dep.Dependency
		for _, row := range current[next:] {
			receipt.HeldCandidates = append(receipt.HeldCandidates, row.Target.ID)
		}
		current = dispatchWaveExecutionPlanWithSize(current, next)
	}
}
func auditDispatchWaveExecutionPlanBounded(root string, maxWorkers int, exclude []string, plan []dispatchWaveExecutionPlan, codexLoopGate string, codexLoopGateSinceHours float64, codexLoopGateLimit int, timeout time.Duration) ([]dispatchWaveExecutionAudit, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	out, err := auditDispatchWaveExecutionPlan(ctx, root, maxWorkers, exclude, plan, codexLoopGate, codexLoopGateSinceHours, codexLoopGateLimit)
	if err == nil {
		return out, nil
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return nil, &dispatchWaveDependencyError{Dependency: "prelaunch contract audit", Kind: "timeout", Attempts: 1, Retryable: true, Timeout: timeout, Err: context.DeadlineExceeded}
	}
	return nil, &dispatchWaveDependencyError{Dependency: "prelaunch contract audit", Kind: "upstream", Attempts: 1, Err: err}
}

func auditDispatchWaveExecutionPlan(ctx context.Context, root string, maxWorkers int, exclude []string, plan []dispatchWaveExecutionPlan, codexLoopGate string, codexLoopGateSinceHours float64, codexLoopGateLimit int) ([]dispatchWaveExecutionAudit, error) {
	if len(plan) == 0 {
		return nil, nil
	}
	// Every prelaunch decider observes the same discovery source. The keyed registry opens
	// one upstream watch for the wave and tears it down after the last decider drops.
	subs := subscribeDispatchWaveDiscovery(root, len(plan))
	defer closeDispatchDiscoverySubscriptions(subs)
	out := make([]dispatchWaveExecutionAudit, 0, len(plan))
	for i, row := range plan {
		snapshot, err := waitDispatchWaveSnapshot(ctx, subs[i].Snapshots)
		if err != nil {
			return nil, err
		}
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
	return out, nil
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
