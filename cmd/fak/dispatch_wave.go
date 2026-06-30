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
	Schema              string                            `json:"schema"`
	Action              string                            `json:"action"`
	ActionReason        string                            `json:"action_reason"`
	Requested           int                               `json:"requested"`
	Granted             int                               `json:"granted"`
	CandidateCount      int                               `json:"candidate_count"`
	ScopedCount         int                               `json:"scoped_count"`
	UnscopedCount       int                               `json:"unscoped_count"`
	ScopeCoveragePct    int                               `json:"scope_coverage_pct"`
	RunLanes            []string                          `json:"run_lanes"`
	RunTargets          []dispatchWaveCandidate           `json:"run_targets"`
	HeldLanes           []string                          `json:"held_lanes,omitempty"`
	ExcludedLanes       []string                          `json:"excluded_lanes,omitempty"`
	Candidates          []dispatchWaveCandidate           `json:"candidates"`
	Collisions          []dispatchorder.Collision         `json:"collisions,omitempty"`
	Repartition         []dispatchorder.RepartitionAdvice `json:"repartition,omitempty"`
	CollisionsAvoided   int                               `json:"collisions_avoided"`
	LanesUtilized       int                               `json:"lanes_utilized"`
	SerializationWasted int                               `json:"serialization_wasted"`
	SafeConcurrency     int                               `json:"safe_concurrency"`
	SafeConcurrencyPct  int                               `json:"safe_concurrency_pct"`
	SameLaneParallelism int                               `json:"same_lane_parallelism"`
}

type dispatchWaveCandidate struct {
	ID           string                    `json:"id"`
	Lane         string                    `json:"lane"`
	LeaseID      string                    `json:"lease_id"`
	Issue        int                       `json:"issue,omitempty"`
	Tree         []string                  `json:"tree,omitempty"`
	Scoped       bool                      `json:"scoped"`
	Disposition  dispatchorder.Disposition `json:"disposition"`
	Reason       string                    `json:"reason"`
	CollidesWith []string                  `json:"collides_with,omitempty"`
	Rank         int                       `json:"rank"`
	Selected     bool                      `json:"selected"`
}

func runDispatchWave(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch wave", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	count := fs.Int("count", 2, "number of distinct account pools to allocate")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "hard cap on live workers, enforced by each tick's preflight")
	backend := fs.String("backend", "claude", "worker backend (claude|opencode|codex)")
	workKind := fs.String("work-kind", "", "switcher work kind (default follows --backend)")
	lane := fs.String("lane", "", "pin every tick to this repo lane (default: busiest-lane pick)")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the busiest-pick")
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
	if len(price.RunLanes) == 0 {
		rec["stop_reason"] = "priced fan-out found no launchable lane"
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}

	ticks := []any{}
	spawned := 0
	limit := len(price.RunLanes)
	if !*live {
		limit = 1
	}
	for i := 0; i < limit; i++ {
		target := price.RunTargets[i]
		acct := accountFromWaveLane(lanes[i])
		mem := dispatchtick.Membership{Rank: i, WaveID: waveID, Size: len(price.RunLanes), Shortfall: shortfall}
		payload, err := evaluateDispatchTick(dispatchTickOptions{
			Workspace:      root,
			MaxWorkers:     *maxWorkers,
			WorkKind:       wk,
			Lane:           target.Lane,
			TargetIssue:    target.Issue,
			LeaseID:        target.LeaseID,
			LeaseTree:      append([]string(nil), target.Tree...),
			Backend:        backendNorm,
			ExcludeLanes:   dispatchSplitCSV(*excludeLane),
			Live:           *live,
			Refresh:        i == 0,
			CooldownMin:    dispatchtick.DefaultCooldownMinutes,
			WorkerTimeoutS: dispatchtick.DefaultWorkerTimeoutS,
			SpawnProbeS:    dispatchtick.DefaultSpawnProbeS,
			RecordLoop:     !*noLedger,
			Account:        &acct,
			Membership:     &mem,
		}, stderr)
		if err != nil {
			ticks = append(ticks, map[string]any{"ok": false, "error": err.Error(), "rank": i})
			rec["stop_reason"] = err.Error()
			break
		}
		payload["wave_rank"] = i
		payload["wave_target"] = target
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
		meta[id] = dispatchWaveCandidate{
			ID:      id,
			Lane:    lane,
			LeaseID: leaseID,
			Issue:   route.Number,
			Tree:    paths,
			Scoped:  true,
		}
		cands = append(cands, dispatchorder.Candidate{
			ID:          id,
			Key:         id,
			Lane:        leaseID,
			Tree:        paths,
			Mode:        "exclusive",
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
		leaseID := dispatchLaneLeaseID(lane)
		issueByLane[lane] = issue
		meta[id] = dispatchWaveCandidate{
			ID:      id,
			Lane:    lane,
			LeaseID: leaseID,
			Issue:   issue,
			Tree:    append([]string(nil), grp.Tree...),
		}
		cands = append(cands, dispatchorder.Candidate{
			ID:          id,
			Key:         id,
			Lane:        leaseID,
			Tree:        grp.Tree,
			Mode:        "exclusive",
			UpdatedUnix: int64(grp.Count),
			CreatedUnix: int64(len(lanes) - i),
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
		cand.Reason = row.Reason
		cand.CollidesWith = append([]string(nil), row.CollidesWith...)
		cand.Rank = row.Rank
		cand.Selected = selected[row.ID]
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
	scopedCount := 0
	for _, cand := range rows {
		if cand.Scoped {
			scopedCount++
		}
	}
	unscopedCount := len(rows) - scopedCount
	action, actionReason := dispatchWaveAction(len(rows), len(runTargets), res.CollisionsAvoided, res.SerializationWasted)
	return dispatchWavePrice{
		Schema:              "fleet-issue-dispatch-wave-price/1",
		Action:              action,
		ActionReason:        actionReason,
		Requested:           requested,
		Granted:             granted,
		CandidateCount:      len(cands),
		ScopedCount:         scopedCount,
		UnscopedCount:       unscopedCount,
		ScopeCoveragePct:    dispatchWavePct(scopedCount, len(rows)),
		RunLanes:            runLaneNames,
		RunTargets:          runTargets,
		HeldLanes:           sortedStringSet(held),
		ExcludedLanes:       sortedStringSet(exclude),
		Candidates:          rows,
		Collisions:          res.Collisions,
		Repartition:         res.Repartition,
		CollisionsAvoided:   res.CollisionsAvoided,
		LanesUtilized:       len(runLanes),
		SerializationWasted: res.SerializationWasted,
		SafeConcurrency:     res.SafeConcurrency,
		SafeConcurrencyPct:  dispatchWavePct(len(runTargets), len(rows)),
		SameLaneParallelism: sameLaneParallelism(runTargets),
	}, nil
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
		fmt.Fprintf(&b, "  priced fan-out: action=%s run=%s collisions_avoided=%d lanes_utilized=%d serialization_wasted=%d safe_concurrency=%d (%d%%) scope=%d%% same_lane_parallelism=%d repartition=%d\n",
			price.Action,
			strings.Join(price.RunLanes, ","), price.CollisionsAvoided, price.LanesUtilized,
			price.SerializationWasted, price.SafeConcurrency, price.SafeConcurrencyPct,
			price.ScopeCoveragePct, price.SameLaneParallelism, len(price.Repartition))
	}
	if !dispatchMapBool(rec, "live") {
		fmt.Fprintln(&b, "  (dry-run - re-run with --live to spawn the wave)")
	}
	return b.String()
}

func waveAccountLanes(doc map[string]any) []map[string]any {
	raw, _ := doc["lanes"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func accountFromWaveLane(m dispatchtick.AccountWaveLane) dispatchtick.Account {
	return dispatchtick.Account{
		Tag:   firstString(m.Tag, m.Account),
		Tier:  m.SelectedTier,
		Model: m.Model,
		Dir:   m.ConfigDir,
	}
}

func firstAny(vals ...any) any {
	for _, v := range vals {
		if v != nil {
			return v
		}
	}
	return nil
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
