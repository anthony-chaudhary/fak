package main

// dispatch_auto.go — `fak dispatch auto`, the auto-sized, self-refilling front door to the
// multi-account wave. The operator (or a scheduled tick) types NO count: the verb folds the
// live ceilings — the preflight's effective cap (configured/lease/host/seat), the switcher's
// distinct fresh account pools, the router's ready work, an optional throughput target — into
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
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchauto"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func runDispatchAuto(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch auto", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "hard cap on live workers, enforced by each tick's preflight")
	backend := fs.String("backend", "claude", "worker backend (claude|opencode|codex)")
	workKind := fs.String("work-kind", "", "switcher work kind (default follows --backend)")
	requiredWorkers := fs.Int("required-workers", 0, "optional throughput target (e.g. fleetcap required workers); 0 = unset")
	contextTokens := fs.Int("context-tokens", 0, "optional fleet context-token budget, sliced evenly across the wave; 0 = unset")
	live := fs.Bool("live", false, "actually spawn the refill through the priced dispatch wave")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
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

	in, notes := probeDispatchAutoInput(root, stderr, *maxWorkers, wk, backendNorm)
	in.RequiredWorkers = *requiredWorkers
	in.SharedContextTokens = *contextTokens
	plan := dispatchauto.PlanAuto(in)

	rec := map[string]any{
		"schema":    "fleet-issue-dispatch-auto/1",
		"workspace": root,
		"live":      *live,
		"backend":   backendNorm,
		"work_kind": wk,
		"input":     in,
		"plan":      plan,
		"notes":     notes,
		"ok":        true,
	}

	if *live && plan.Refill > 0 {
		waveArgv := []string{
			"--workspace", root,
			"--count", fmt.Sprint(plan.Refill),
			"--max-workers", fmt.Sprint(*maxWorkers),
			"--backend", backendNorm,
			"--work-kind", wk,
			"--live", "--json",
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
		return writeDispatchAutoResult(stdout, stderr, rec, plan, *asJSON)
	}

	return writeDispatchAutoResult(stdout, stderr, rec, plan, *asJSON)
}

// probeDispatchAutoInput gathers the live ceilings with the SAME folds tick/wave use: the
// preflight's already-min-folded effective cap and live count, the switcher's distinct-pool
// allocation, and the router's ready-work count. A probe that fails contributes a note and a
// conservative value (0), never a crash — an unknown ceiling reads as "no wave" for the hard
// facts and "unset" for the optional ones, matching the fold's zero-value contract.
func probeDispatchAutoInput(root string, stderr io.Writer, maxWorkers int, workKind, backend string) (dispatchauto.Input, []string) {
	in := dispatchauto.Input{}
	notes := []string{}

	product := dispatchtick.ProductForBackend(backend)
	if pf, err := dispatchPreflight(root, stderr, maxWorkers, workKind, product); err == nil {
		if terms, ok := pf["cap_terms"].(map[string]any); ok {
			in.EffectiveCap = dispatchMapInt(terms, "effective_cap")
		}
		in.LiveWorkers = dispatchMapInt(pf, "live")
		if verdict := dispatchMapString(pf, "verdict"); verdict != "" {
			notes = append(notes, "preflight: "+verdict)
		}
	} else {
		notes = append(notes, "preflight probe failed: "+err.Error())
	}

	if rows, err := dispatchReadAccountRoster(root); err == nil {
		ask := in.EffectiveCap
		if ask <= 0 {
			ask = maxWorkers
		}
		alloc := dispatchtick.AllocateWave(dispatchtick.AccountWaveInput{
			Rows:     rows,
			Count:    ask,
			WorkKind: workKind,
			Product:  product,
		})
		in.DistinctPools = len(alloc.Lanes)
		if alloc.Shortfall > 0 {
			notes = append(notes, fmt.Sprintf("accounts: %d distinct pool(s) free, %d short of the ask", len(alloc.Lanes), alloc.Shortfall))
		}
	} else {
		notes = append(notes, "account roster probe failed: "+err.Error())
	}

	if router, err := dispatchRouteIssues(root, stderr); err == nil {
		in.ReadyWork = dispatchAutoReadyWork(router)
	} else {
		notes = append(notes, "issue router probe failed: "+err.Error())
	}

	return in, notes
}

// dispatchAutoReadyWork counts the dispatchable units the router sees: the routed issue list
// when present, else the lane groups' counts (the same fallback the wave pricer walks).
func dispatchAutoReadyWork(router dispatchtick.RouterPayload) int {
	if n := len(router.Issues); n > 0 {
		return n
	}
	total := 0
	for _, grp := range router.Lanes {
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
