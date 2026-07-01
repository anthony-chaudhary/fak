package main

import (
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

const dispatchWaveReasonWaveCap = "wave-cap"

type dispatchWaveCandidate struct {
	ID           string                    `json:"id"`
	Lane         string                    `json:"lane"`
	LeaseID      string                    `json:"lease_id"`
	Issue        int                       `json:"issue,omitempty"`
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
	count := fs.Int("count", 2, "number of distinct account pools to allocate")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "hard cap on live workers, enforced by each tick's preflight")
	backend := fs.String("backend", "claude", "worker backend (claude|opencode|codex)")
	workKind := fs.String("work-kind", "", "switcher work kind (default follows --backend)")
	lane := fs.String("lane", "", "pin every tick to this repo lane (default: largest step-budget lane pick)")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the step-budget pick")
	settleS := fs.Float64("settle-s", 2.0, "seconds to wait after each live spawn")
	noLedger := fs.Bool("no-loop-ledger", false, "disable loop-ledger append for spawned ticks")
	live := fs.Bool("live", false, "actually spawn workers")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
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

	rows, err := dispatchReadAccountRoster(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch wave: allocate accounts: %v\n", err)
		return 1
	}
	alloc := dispatchtick.AllocateWave(dispatchtick.AccountWaveInput{
		Rows:     rows,
		Count:    *count,
		WorkKind: wk,
		Product:  dispatchtick.ProductForBackend(backendNorm),
	})
	lanes := alloc.Lanes
	waveID := alloc.WaveID
	shortfall := alloc.Shortfall
	rec := map[string]any{
		"schema":      "fleet-issue-dispatch-wave/1",
		"workspace":   root,
		"live":        *live,
		"backend":     backendNorm,
		"work_kind":   wk,
		"requested":   *count,
		"granted":     len(lanes),
		"shortfall":   shortfall,
		"wave_id":     waveID,
		"allocation":  scrubDispatchSecrets(alloc.Map()),
		"ticks":       []any{},
		"spawned":     0,
		"stop_reason": "",
		"ok":          false,
	}
	if len(lanes) == 0 {
		rec["stop_reason"] = firstString(alloc.Reason, "no distinct account pools available")
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}

	price, err := priceDispatchWave(root, stderr, *count, len(lanes), *lane, dispatchSplitCSV(*excludeLane), dispatchtick.DefaultCooldownMinutes)
	if err != nil {
		rec["stop_reason"] = "price fan-out: " + err.Error()
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}
	rec["price"] = price
	rec["planned_lanes"] = append([]string(nil), price.RunLanes...)
	executionPlan := dispatchWaveExecutionPlans(root, backendNorm, wk, waveID, shortfall, price.RunTargets, lanes, !*noLedger)
	executionPlanID := dispatchWaveExecutionPlanID(executionPlan)
	rec["execution_plan_id"] = executionPlanID
	rec["execution_plan"] = executionPlan
	if len(price.RunLanes) == 0 {
		rec["stop_reason"] = "priced fan-out found no launchable lane"
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}
	executionAudit := auditDispatchWaveExecutionPlan(root, *maxWorkers, dispatchSplitCSV(*excludeLane), executionPlan)
	rec["execution_plan_audit"] = executionAudit
	prelaunchGate := dispatchWavePrelaunchGateFromAudit(executionPlanID, executionAudit)
	rec["prelaunch_gate"] = prelaunchGate
	if *live && !prelaunchGate.OK {
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
	for i := 0; i < limit; i++ {
		row := executionPlan[i]
		payload, err := evaluateDispatchTick(dispatchWaveExecutionTickOptions(root, *maxWorkers, dispatchSplitCSV(*excludeLane), row, *live, i == 0), stderr)
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
			rec["stop_reason"] = "dry-run: planned the first wave tick only; re-run with --live to spawn"
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

func priceDispatchWave(root string, stderr io.Writer, requested, granted int, explicitLane string, excluded []string, cooldownMin int) (dispatchWavePrice, error) {
	router, err := dispatchRouteIssues(root, stderr)
	if err != nil {
		return dispatchWavePrice{}, err
	}
	return priceDispatchWavePayload(root, router, requested, granted, explicitLane, excluded, cooldownMin)
}

func priceDispatchWavePayload(root string, router dispatchtick.RouterPayload, requested, granted int, explicitLane string, excluded []string, cooldownMin int) (dispatchWavePrice, error) {
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	held := liveResolutionLanes(runsDir)
	liveIssues := liveResolutionIssues(runsDir)
	cooled := recentlyAttemptedIssues(runsDir, cooldownMin)
	exclude := map[string]bool{}
	for _, lane := range excluded {
		exclude[lane] = true
	}
	for lane := range held {
		exclude[lane] = true
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
		if liveIssues[route.Number] || cooled[route.Number] {
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
		meta[id] = dispatchWaveCandidate{
			ID:         id,
			Lane:       lane,
			LeaseID:    leaseID,
			Issue:      route.Number,
			StepBudget: stepBudget,
			Tree:       paths,
			Scoped:     true,
		}
		cands = append(cands, dispatchorder.Candidate{
			ID:          id,
			Key:         id,
			Lane:        leaseID,
			Tree:        paths,
			Mode:        "exclusive",
			UpdatedUnix: int64(stepBudget),
			CreatedUnix: int64(route.Number),
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
		sort.Ints(nums)
		issue, ok := firstLaunchableIssue(nums, liveIssues, cooled)
		if !ok {
			continue
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
			StepBudget: stepBudget,
			Tree:       append([]string(nil), grp.Tree...),
		}
		cands = append(cands, dispatchorder.Candidate{
			ID:          id,
			Key:         id,
			Lane:        leaseID,
			Tree:        grp.Tree,
			Mode:        "exclusive",
			UpdatedUnix: int64(stepBudget),
			CreatedUnix: int64(grp.Count*len(lanes) + (len(lanes) - i)),
		})
	}

	res := dispatchorder.Plan(dispatchorder.Input{
		Candidates:      cands,
		NowUnix:         time.Now().Unix(),
		CooldownSeconds: -1,
	})
	limit := minInt(requested, granted)
	runLanes := append([]string(nil), res.Keep...)
	if limit < len(runLanes) {
		runLanes = runLanes[:limit]
	}
	selected := map[string]bool{}
	for _, id := range runLanes {
		selected[id] = true
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
	}, nil
}

func dispatchWaveCandidateReason(row dispatchorder.Ranked, selected bool) string {
	if row.Disposition == dispatchorder.DispKeep && !selected {
		return dispatchWaveReasonWaveCap
	}
	return row.Reason
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
	if len(candidates) == 0 {
		return nil
	}
	ids := make([]string, 0, len(candidates))
	for _, cand := range candidates {
		ids = append(ids, cand.ID)
	}
	return dispatchWavesForIDs(ids, collisions, safeNow)
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

func auditDispatchWaveExecutionPlan(root string, maxWorkers int, exclude []string, plan []dispatchWaveExecutionPlan) []dispatchWaveExecutionAudit {
	if len(plan) == 0 {
		return nil
	}
	out := make([]dispatchWaveExecutionAudit, 0, len(plan))
	for _, row := range plan {
		payload, err := evaluateDispatchTick(dispatchWaveExecutionTickOptions(root, maxWorkers, exclude, row, false, false), io.Discard)
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

func dispatchWaveExecutionTickOptions(root string, maxWorkers int, exclude []string, row dispatchWaveExecutionPlan, live bool, refresh bool) dispatchTickOptions {
	acct := dispatchWaveAccountFromPlan(row.Account)
	mem := dispatchtick.Membership{
		Rank:      row.Rank,
		WaveID:    row.WaveID,
		Size:      row.WaveSize,
		Shortfall: row.Shortfall,
	}
	return dispatchTickOptions{
		Workspace:      root,
		MaxWorkers:     maxWorkers,
		WorkKind:       row.WorkKind,
		Lane:           row.Target.Lane,
		TargetIssue:    row.Target.Issue,
		LeaseID:        row.Target.LeaseID,
		LeaseTree:      append([]string(nil), row.Target.Tree...),
		Backend:        row.Backend,
		ExcludeLanes:   append([]string(nil), exclude...),
		Live:           live,
		Refresh:        refresh,
		CooldownMin:    dispatchtick.DefaultCooldownMinutes,
		WorkerTimeoutS: dispatchtick.DefaultWorkerTimeoutS,
		SpawnProbeS:    dispatchtick.DefaultSpawnProbeS,
		RecordLoop:     live && row.RecordLoop,
		Account:        &acct,
		Membership:     &mem,
	}
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
	if !gate.OK {
		gate.Action = "HOLD"
		if len(gate.Refused) > 0 {
			first := gate.Refused[0]
			gate.Reason = strings.TrimSpace(first.Target + " " + firstString(first.Reason, first.Error, first.Verdict, first.Action, "not ok"))
		}
	}
	return gate
}

func dispatchWaveExecutionPlanID(plan []dispatchWaveExecutionPlan) string {
	if len(plan) == 0 {
		return ""
	}
	return dispatchStablePlanID(plan)
}

func dispatchWaveExecutionPlans(root, backend, workKind, waveID string, shortfall int, targets []dispatchWaveCandidate, lanes []dispatchtick.AccountWaveLane, recordLoop bool) []dispatchWaveExecutionPlan {
	limit := minInt(len(targets), len(lanes))
	if limit <= 0 {
		return nil
	}
	out := make([]dispatchWaveExecutionPlan, 0, limit)
	for i := 0; i < limit; i++ {
		target := targets[i]
		acct := accountFromWaveLane(lanes[i])
		mem := dispatchtick.Membership{Rank: i, WaveID: waveID, Size: limit, Shortfall: shortfall}
		args := dispatchWaveExecutionTickArgs(root, backend, workKind, target, acct, mem, recordLoop)
		out = append(out, dispatchWaveExecutionPlan{
			Rank:                i,
			WaveID:              waveID,
			WaveSize:            limit,
			Shortfall:           shortfall,
			Backend:             backend,
			WorkKind:            workKind,
			Target:              dispatchWaveLaunchTarget(target),
			Account:             dispatchtick.AccountSidecar(acct),
			RecordLoop:          recordLoop,
			DispatchTickArgs:    args,
			DispatchTickCommand: append([]string{"fak", "dispatch", "tick"}, args...),
		})
	}
	return out
}

func dispatchWaveExecutionTickArgs(root, backend, workKind string, target dispatchWaveCandidate, account dispatchtick.Account, membership dispatchtick.Membership, recordLoop bool) []string {
	args := []string{"--workspace", root, "--backend", backend}
	if strings.TrimSpace(workKind) != "" {
		args = append(args, "--work-kind", workKind)
	}
	args = append(args, dispatchTickArgsForLaunchTarget(target)...)
	if !recordLoop {
		args = append(args, "--no-loop-ledger")
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
	keys := make([]string, 0, len(candidates))
	for _, cand := range candidates {
		key := strings.TrimSpace(cand.Lane)
		if key == "" {
			key = strings.TrimSpace(cand.LeaseID)
		}
		if key == "" {
			key = cand.ID
		}
		keys = append(keys, key)
	}
	return dispatchLaneSerialWaveCount(keys)
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

func firstLaunchableIssue(nums []int, live, cooled map[int]bool) (int, bool) {
	for _, n := range nums {
		if !live[n] && !cooled[n] {
			return n, true
		}
	}
	return 0, false
}

func writeDispatchWaveResult(stdout, stderr io.Writer, rec map[string]any, asJSON bool) int {
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
	return 1
}

func renderDispatchWave(rec map[string]any) string {
	var b strings.Builder
	mode := "dry-run"
	if dispatchMapBool(rec, "live") {
		mode = "live"
	}
	fmt.Fprintf(&b, "issue-dispatch-wave: %s  requested=%d granted=%d spawned=%d backend=%s\n",
		mode, dispatchMapInt(rec, "requested"), dispatchMapInt(rec, "granted"),
		dispatchMapInt(rec, "spawned"), dispatchMapString(rec, "backend"))
	if id := dispatchMapString(rec, "wave_id"); id != "" {
		fmt.Fprintf(&b, "  wave_id: %s\n", id)
	}
	if reason := dispatchMapString(rec, "stop_reason"); reason != "" {
		fmt.Fprintf(&b, "  stop: %s\n", reason)
	}
	if price, ok := rec["price"].(dispatchWavePrice); ok {
		fmt.Fprintf(&b, "  priced fan-out: action=%s run=%s effective_cap=%d run_steps=%d candidate_steps=%d collisions_avoided=%d lanes_utilized=%d serialization_wasted=%d safe_concurrency=%d (%d%%) scope=%d%% same_lane_parallelism=%d repartition=%d\n",
			price.Action,
			strings.Join(price.RunLanes, ","), price.EffectiveCap, price.RunStepBudget, price.CandidateStepBudget, price.CollisionsAvoided, price.LanesUtilized,
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
		fmt.Fprintln(&b, "  (dry-run - re-run with --live to spawn the wave)")
	}
	return b.String()
}

func dispatchWaveSkippedCandidates(candidates []dispatchWaveCandidate) []dispatchWaveCandidate {
	out := make([]dispatchWaveCandidate, 0, len(candidates))
	for _, cand := range candidates {
		if !cand.Selected {
			out = append(out, cand)
		}
	}
	return out
}

func dispatchWaveIssueLabel(cand dispatchWaveCandidate) string {
	if cand.Issue <= 0 {
		return "-"
	}
	return fmt.Sprintf("#%d", cand.Issue)
}

func dispatchWaveScopeLabel(cand dispatchWaveCandidate) string {
	switch {
	case len(cand.Tree) == 0:
		return "unknown"
	case cand.Scoped:
		return "issue"
	default:
		return "lane"
	}
}

func dispatchWaveCollisionLabel(ids []string) string {
	if len(ids) == 0 {
		return "-"
	}
	return strings.Join(ids, ",")
}

func accountFromWaveLane(m dispatchtick.AccountWaveLane) dispatchtick.Account {
	return dispatchtick.Account{
		Tag:   firstString(m.Tag, m.Account),
		Tier:  m.SelectedTier,
		Model: m.Model,
		Dir:   m.ConfigDir,
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func scrubDispatchSecrets(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, val := range x {
			if dispatchSecretKey(k) {
				continue
			}
			out[k] = scrubDispatchSecrets(val)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = scrubDispatchSecrets(val)
		}
		return out
	default:
		return v
	}
}

func dispatchSecretKey(k string) bool {
	k = strings.ToLower(strings.TrimSpace(k))
	return strings.Contains(k, "token") || strings.Contains(k, "secret") || strings.Contains(k, "api_key") || strings.Contains(k, "apikey")
}
