package main

// dispatch_auto.go — `fak dispatch auto`, the auto-sized, self-refilling front door to the
// multi-account wave. The operator (or a scheduled tick) types NO count: the verb folds the
// live ceilings — the preflight's effective cap (configured/lease/host/seat), the switcher's
// fresh account session slots, the router's ready work, an optional throughput target — into
// a steady-state Target, computes the Refill (Target minus live workers), and drives the
// existing priced `dispatch wave` path with that number. Run it on a cadence and the worker
// population converges to Target and tops itself back up as workers exit — load-balancing
// across accounts becomes the default, not an operator request (#1333).
//
//	fak dispatch auto                    # plan only: target, refill, and the binding ceiling
//	fak dispatch auto --live             # spawn the refill through the priced wave
//	fak dispatch auto --context-tokens 300000   # slice a fleet context budget per worker
//
// The DECISION is pure (internal/dispatchauto.PlanAuto): same ceilings in, same plan out.
// This shell does only the wire: probe the ceilings with the same folds tick/wave use, call
// PlanAuto, render, and (with --live) delegate the spawn to runDispatchWave — the wave still
// owns pricing, collision serialization, per-tick preflight, and account pinning.

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchauto"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func runDispatchAuto(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch auto", flag.ContinueOnError)
	fs.SetOutput(stderr)
	defaults := defaultDispatchAutoBudgets()
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "hard cap on live workers, enforced by each tick's preflight")
	backend := fs.String("backend", "codex", "worker backend (claude|opencode|codex); default codex")
	workKind := fs.String("work-kind", "", "switcher work kind (default follows --backend)")
	goal := fs.String("goal", "", "durable dispatch loop goal id (for example throughput or high-priority); forwarded to the refill wave")
	goalProfile := fs.String("goal-profile", "", "dispatch picker profile: throughput|high-priority (default follows --goal, else throughput)")
	lane := fs.String("lane", "", "pin the refill wave to this repo lane")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the refill wave")
	requiredWorkers := fs.Int("required-workers", 0, "optional throughput target (e.g. fleetcap required workers); 0 = unset")
	contextTokens := fs.Int("context-tokens", 0, "optional fleet context-token budget, sliced evenly across the wave; 0 = unset")
	live := fs.Bool("live", false, "actually spawn the refill through the priced dispatch wave")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	buildTimeout := fs.Duration("build-timeout", defaults.Build, "committed-build phase budget")
	backlogTimeout := fs.Duration("backlog-timeout", defaults.BacklogFetch, "GitHub backlog-fetch phase budget")
	rankingTimeout := fs.Duration("ranking-timeout", defaults.Ranking, "backlog ranking phase budget")
	pricingTimeout := fs.Duration("pricing-timeout", defaults.Pricing, "capacity pricing phase budget")
	outputTimeout := fs.Duration("render-timeout", defaults.Render, "result render phase budget")
	totalTimeout := fs.Duration("timeout", defaults.Total, "total dry-run admission budget")
	if !parseFlags(fs, argv) {
		return 2
	}
	budgets := dispatchAutoBudgets{
		Build: *buildTimeout, BacklogFetch: *backlogTimeout, Ranking: *rankingTimeout,
		Pricing: *pricingTimeout, Render: *outputTimeout, Total: *totalTimeout,
	}
	if err := budgets.validate(); err != nil {
		fmt.Fprintf(stderr, "fak dispatch auto: %v\n", err)
		return 2
	}
	root := *workspace
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch auto: getwd: %v\n", err)
			return 1
		}
		root = wd
	}
	backendNorm, err := dispatchtick.NormalizeBackend(*backend)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch auto: %v\n", err)
		return 2
	}
	wk := strings.TrimSpace(*workKind)
	if wk == "" {
		wk = dispatchtick.DefaultWorkKind(backendNorm)
	}
	goalID, profile, goalErr := normalizeDispatchGoal(*goal, *goalProfile)
	if goalErr != nil {
		fmt.Fprintf(stderr, "fak dispatch auto: %v\n", goalErr)
		return 2
	}
	excluded := splitCommaList(*excludeLane)
	started := time.Now()
	total, cancel := context.WithTimeout(context.Background(), budgets.Total)
	defer cancel()
	var (
		build     dispatchAutoBuildResult
		backlog   dispatchAutoBacklogResult
		router    dispatchtick.RouterPayload
		pricing   dispatchAutoPricingResult
		plan      dispatchauto.Plan
		timings   []dispatchAutoPhaseTiming
		phaseErrs []dispatchAutoError
		blocked   bool
	)
	rec := map[string]any{
		"schema":         "fleet-issue-dispatch-auto/1",
		"workspace":      root,
		"live":           *live,
		"backend":        backendNorm,
		"work_kind":      wk,
		"goal":           goalID,
		"goal_profile":   profile,
		"lane":           strings.TrimSpace(*lane),
		"excluded_lanes": excluded,
		"budget":         budgets.receipt(),
		"partial":        true,
		"ok":             false,
	}

	build, timing, phaseErr := runDispatchAutoPhase(total, dispatchAutoPhaseBuild, budgets.Build, func(ctx context.Context) (dispatchAutoBuildResult, error) {
		return dispatchAutoBuildProbe(ctx, root)
	})
	timings = append(timings, timing)
	rec["build"] = build.Evidence
	if phaseErr != nil {
		phaseErrs = append(phaseErrs, *phaseErr)
		blocked = true
	}

	if !blocked {
		backlog, timing, phaseErr = runDispatchAutoPhase(total, dispatchAutoPhaseBacklogFetch, budgets.BacklogFetch, func(ctx context.Context) (dispatchAutoBacklogResult, error) {
			return dispatchAutoBacklogProbe(ctx, root)
		})
		timings = append(timings, timing)
		rec["backlog"] = map[string]any{"issue_count": len(backlog.Fetched.Issues), "limit": backlog.Fetched.IssueLimit}
		if phaseErr != nil {
			phaseErrs = append(phaseErrs, *phaseErr)
			blocked = true
		}
	}

	if !blocked {
		router, timing, phaseErr = runDispatchAutoPhase(total, dispatchAutoPhaseRanking, budgets.Ranking, func(ctx context.Context) (dispatchtick.RouterPayload, error) {
			return dispatchAutoRankingProbe(ctx, root, backlog)
		})
		timings = append(timings, timing)
		rec["ranking"] = map[string]any{"routed_issues": len(router.Issues), "lanes": len(router.Lanes), "verdict": router.Verdict}
		if phaseErr != nil {
			phaseErrs = append(phaseErrs, *phaseErr)
			blocked = true
		}
	}

	if !blocked {
		pricing, timing, phaseErr = runDispatchAutoPhase(total, dispatchAutoPhasePricing, budgets.Pricing, func(ctx context.Context) (dispatchAutoPricingResult, error) {
			return dispatchAutoPricingProbe(ctx, root, stderr, *maxWorkers, wk, backendNorm, *lane, excluded, *requiredWorkers, *contextTokens, build.Tree, router)
		})
		timings = append(timings, timing)
		plan = pricing.Plan
		rec["input"] = pricing.Input
		rec["plan"] = plan
		rec["notes"] = pricing.Notes
		rec["preflight"] = pricing.Preflight
		if phaseErr != nil {
			phaseErrs = append(phaseErrs, *phaseErr)
			blocked = true
		}
	}

	rec["phase_timings"] = dispatchAutoCompleteTimings(timings, budgets)
	_, timing, phaseErr = runDispatchAutoPhase(total, dispatchAutoPhaseOutput, budgets.Render, func(ctx context.Context) (struct{}, error) {
		return struct{}{}, dispatchAutoRenderProbe(ctx, rec, plan, *asJSON)
	})
	timings = append(timings, timing)
	if phaseErr != nil {
		phaseErrs = append(phaseErrs, *phaseErr)
		blocked = true
	}
	rec["phase_timings"] = dispatchAutoCompleteTimings(timings, budgets)
	rec["admission_elapsed_ms"] = elapsedDispatchAutoMS(started)
	if len(phaseErrs) > 0 {
		legacy := make([]string, 0, len(phaseErrs))
		for _, item := range phaseErrs {
			legacy = append(legacy, item.Message)
		}
		rec["error"] = phaseErrs[0]
		rec["phase_errors"] = phaseErrs
		rec["errors"] = legacy
	} else {
		rec["partial"] = false
		rec["ok"] = true
	}
	cancel()

	if *live && !blocked && plan.Refill > 0 {
		waveArgv := []string{
			"--workspace", root,
			"--count", fmt.Sprint(plan.Refill),
			"--max-workers", fmt.Sprint(*maxWorkers),
			"--backend", backendNorm,
			"--work-kind", wk,
			"--live", "--json",
		}
		if goalID != "" {
			waveArgv = append(waveArgv, "--goal", goalID)
		}
		if profile != "" && profile != dispatchGoalProfileThroughput {
			waveArgv = append(waveArgv, "--goal-profile", profile)
		}
		if strings.TrimSpace(*lane) != "" {
			waveArgv = append(waveArgv, "--lane", strings.TrimSpace(*lane))
		}
		if len(excluded) > 0 {
			waveArgv = append(waveArgv, "--exclude-lane", strings.Join(excluded, ","))
		}
		var waveOut bytes.Buffer
		code := runDispatchWave(&waveOut, stderr, waveArgv)
		var waveRec map[string]any
		if json.Unmarshal(waveOut.Bytes(), &waveRec) == nil {
			rec["wave"] = waveRec
		} else {
			rec["wave_raw"] = waveOut.String()
		}
		rec["ok"] = code == 0
	}

	return writeDispatchAutoResult(stdout, stderr, rec, plan, *asJSON)
}

var dispatchAutoRenderProbe = func(ctx context.Context, rec map[string]any, plan dispatchauto.Plan, asJSON bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if asJSON {
		_, err := json.Marshal(rec)
		return err
	}
	_ = renderDispatchAuto(rec, plan)
	return ctx.Err()
}

// probeDispatchAutoInput gathers the live ceilings with the SAME folds tick/wave use: the
// preflight's already-min-folded effective cap and live count, the switcher's free
// account-slot headroom, and the router's ready-work count. A probe that fails contributes
// a note and a conservative value (0), never a crash — an unknown ceiling reads as
// "no wave" for the hard facts and "unset" for the optional ones, matching the fold's
// zero-value contract.
func probeDispatchAutoInput(root string, stderr io.Writer, maxWorkers int, workKind, backend, lane string, excluded []string) (dispatchauto.Input, []string, []string) {
	in := dispatchauto.Input{}
	notes := []string{}
	probeErrors := []string{}

	product := dispatchtick.ProductForBackend(backend)
	var preflightSeatFree *int
	if pf, err := dispatchPreflight(root, stderr, maxWorkers, workKind, product); err == nil {
		if terms, ok := pf["cap_terms"].(map[string]any); ok {
			in.EffectiveCap = dispatchMapInt(terms, "effective_cap")
		}
		in.LiveWorkers = dispatchMapInt(pf, "live")
		if seat, ok := pf["seat"].(map[string]any); ok {
			preflightSeatFree = intPtrFromAny(seat["free"])
		}
		if verdict := dispatchMapString(pf, "verdict"); verdict != "" {
			notes = append(notes, "preflight: "+verdict)
		}
	} else {
		msg := "preflight probe failed: " + err.Error()
		notes = append(notes, msg)
		probeErrors = append(probeErrors, msg)
	}

	if rows, err := dispatchReadAccountRoster(root); err == nil {
		leases := dispatchLiveSeatLeases(filepath.Join(root, dispatchtick.RunsDirName))
		pool := dispatchtick.BuildSeatPool(rows, leases, product)
		// Price the same work-kind tier that dispatch wave will allocate. BuildSeatPool
		// counts every product seat, including tiers this wave cannot use.
		alloc := dispatchtick.AllocateWave(dispatchtick.AccountWaveInput{
			Rows: rows, Leases: leases, Count: pool.FreeSeats, WorkKind: workKind, Product: product,
		})
		freeSlots := alloc.Granted
		if preflightSeatFree != nil && *preflightSeatFree < freeSlots {
			freeSlots = *preflightSeatFree
		}
		in.DistinctPools = dispatchAutoAccountCapacity(in.LiveWorkers, freeSlots)
		if pool.TotalSeats > 0 {
			notes = append(notes, fmt.Sprintf("accounts: %d/%d compatible session slot(s) free for %s", freeSlots, pool.TotalSeats, workKind))
		}
		if freeSlots == 0 && strings.TrimSpace(alloc.Reason) != "" {
			notes = append(notes, "accounts: "+strings.TrimSpace(alloc.Reason))
		}
	} else {
		msg := "account roster probe failed: " + err.Error()
		notes = append(notes, msg)
		probeErrors = append(probeErrors, msg)
	}

	if router, err := dispatchRouteIssues(root, stderr); err == nil {
		if reason := dispatchAutoRouterProbeError(router); reason != "" {
			msg := "issue router probe failed: " + reason
			notes = append(notes, msg)
			probeErrors = append(probeErrors, msg)
		} else {
			in.ReadyWork = dispatchAutoReadyWork(router, lane, excluded)
		}
	} else {
		msg := "issue router probe failed: " + err.Error()
		notes = append(notes, msg)
		probeErrors = append(probeErrors, msg)
	}

	return in, notes, probeErrors
}

func dispatchAutoAccountCapacity(liveWorkers, freeSlots int) int {
	if liveWorkers < 0 {
		liveWorkers = 0
	}
	if freeSlots < 0 {
		freeSlots = 0
	}
	return liveWorkers + freeSlots
}

func dispatchAutoRouterProbeError(router dispatchtick.RouterPayload) string {
	if strings.EqualFold(strings.TrimSpace(router.Verdict), "FETCH_ERROR") ||
		strings.EqualFold(strings.TrimSpace(router.Finding), "fetch_error") {
		reason := strings.TrimSpace(router.Reason)
		if reason == "" {
			reason = "router fetch error"
		}
		return reason
	}
	return ""
}

// dispatchAutoReadyWork counts the dispatchable units the router sees: the routed issue list
// when present, else the lane groups' counts (the same fallback the wave pricer walks).
func dispatchAutoReadyWork(router dispatchtick.RouterPayload, lane string, excluded []string) int {
	wantLane := strings.TrimSpace(lane)
	exclude := map[string]bool{}
	for _, name := range excluded {
		name = strings.TrimSpace(name)
		if name != "" {
			exclude[name] = true
		}
	}
	if n := len(router.Issues); n > 0 {
		total := 0
		for _, issue := range router.Issues {
			issueLane := strings.TrimSpace(issue.Lane)
			if wantLane != "" && issueLane != wantLane {
				continue
			}
			if exclude[issueLane] {
				continue
			}
			total++
		}
		return total
	}
	total := 0
	for name, grp := range router.Lanes {
		name = strings.TrimSpace(name)
		if wantLane != "" && name != wantLane {
			continue
		}
		if exclude[name] {
			continue
		}
		if grp.Count > 0 {
			total += grp.Count
		} else {
			total += len(grp.Issues)
		}
	}
	return total
}

func writeDispatchAutoResult(stdout, stderr io.Writer, rec map[string]any, plan dispatchauto.Plan, asJSON bool) int {
	if asJSON {
		if err := writeIndentedJSON(stdout, rec); err != nil {
			fmt.Fprintf(stderr, "fak dispatch auto: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderDispatchAuto(rec, plan))
	}
	if dispatchMapBool(rec, "ok") {
		return 0
	}
	return 1
}

func renderDispatchAuto(rec map[string]any, plan dispatchauto.Plan) string {
	var b strings.Builder
	mode := "dry-run"
	if dispatchMapBool(rec, "live") {
		mode = "live"
	}
	fmt.Fprintf(&b, "issue-dispatch-auto: %s  %s  backend=%s\n", mode, plan, dispatchMapString(rec, "backend"))
	fmt.Fprintf(&b, "  %s\n", plan.Reason)
	if plan.PerWorkerContextTokens > 0 {
		fmt.Fprintf(&b, "  per-worker context: %d tokens\n", plan.PerWorkerContextTokens)
	}
	if notes, ok := rec["notes"].([]string); ok {
		for _, note := range notes {
			fmt.Fprintf(&b, "  note: %s\n", note)
		}
	}
	if errs, ok := rec["errors"].([]string); ok {
		for _, err := range errs {
			fmt.Fprintf(&b, "  error: %s\n", err)
		}
	}
	if wave, ok := rec["wave"].(map[string]any); ok {
		fmt.Fprintf(&b, "  wave: spawned=%d stop=%s wave_id=%s\n",
			dispatchMapInt(wave, "spawned"), dispatchMapString(wave, "stop_reason"), dispatchMapString(wave, "wave_id"))
	}
	if !dispatchMapBool(rec, "live") {
		fmt.Fprintln(&b, "  (dry-run - re-run with --live to spawn the refill through the priced wave)")
	} else if plan.Refill == 0 {
		fmt.Fprintln(&b, "  (nothing to spawn - the population already meets the target)")
	}
	return b.String()
}
