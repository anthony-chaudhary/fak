package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/fleetcap"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

const fallbackCodexOAuthSessions = 10

const dispatchPreflightTimeout = 30 * time.Second

func dispatchRefreshRegistry(root string, stderr io.Writer) map[string]any {
	obj, err := dispatchRunJSON(root, stderr, 120*time.Second, filepath.Join("tools", "fleet_sessions.py"), "registry")
	if err != nil {
		return map[string]any{"ok": false, "error": err.Error()}
	}
	obj["ok"] = obj["_error"] == nil
	return obj
}

func dispatchPreflight(root string, stderr io.Writer, maxWorkers int, workKind, product string) (map[string]any, error) {
	out, _, err := dispatchPreflightTimed(root, stderr, maxWorkers, workKind, product)
	return out, err
}

func dispatchPreflightTimed(root string, stderr io.Writer, maxWorkers int, workKind, product string) (map[string]any, map[string]int64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), dispatchPreflightTimeout)
	defer cancel()
	return dispatchPreflightTimedContext(ctx, root, stderr, maxWorkers, workKind, product)
}

func dispatchPreflightContext(ctx context.Context, root string, stderr io.Writer, maxWorkers int, workKind, product string) (map[string]any, error) {
	out, _, err := dispatchPreflightTimedContext(ctx, root, stderr, maxWorkers, workKind, product)
	return out, err
}

func dispatchPreflightTimedContext(ctx context.Context, root string, stderr io.Writer, maxWorkers int, workKind, product string) (map[string]any, map[string]int64, error) {
	return dispatchPreflightTimedWithTreeContext(ctx, root, stderr, maxWorkers, workKind, product, nil)
}

func dispatchPreflightTimedWithTree(root string, stderr io.Writer, maxWorkers int, workKind, product string, treeOverride *dispatchtick.TreeCheck) (map[string]any, map[string]int64, error) {
	return dispatchPreflightTimedWithTreeContext(context.Background(), root, stderr, maxWorkers, workKind, product, treeOverride)
}

func dispatchPreflightTimedWithTreeContext(ctx context.Context, root string, stderr io.Writer, maxWorkers int, workKind, product string, treeOverride *dispatchtick.TreeCheck) (map[string]any, map[string]int64, error) {
	timings := map[string]int64{}
	stamp := func(name string, started time.Time) {
		ms := time.Since(started).Milliseconds()
		if ms == 0 {
			ms = 1
		}
		timings[name] = ms
	}
	// One host process snapshot feeds both the safety fold and aggregate thread/RAM
	// capacity. Before #4258 those consumers each spawned their own full Get-Process.
	t0 := time.Now()
	processes := dispatchProbeProcesses()
	stamp("process_snapshot", t0)
	t0 = time.Now()
	kernel := dispatchPreflightKernel(root)
	stamp("kernel_probe", t0)
	wip := dispatchPreflightWIP(root)
	tree := dispatchPreflightTree(ctx, root, treeOverride)
	// OSWorkerProcs is the PUBLISHED per-worker load (#3376), not the raw per-tick
	// probe: dispatchPublishWorkerLoad still samples every tick but only lets a CHANGED
	// value through, and only after it survives a reset-on-change coalescing window.
	in := dispatchtick.PreflightInput{
		Workspace:     root,
		MaxWorkers:    maxWorkers,
		Host:          dispatchPreflightHostFromProcesses(processes),
		Tree:          tree,
		Account:       dispatchPreflightAccount(root, stderr, workKind, product),
		Kernel:        kernel,
		Seat:          dispatchPreflightSeat(root, stderr, product),
		Resources:     dispatchBuildHostResources(processes),
		Budgets:       dispatchtick.DefaultHostBudgets(),
		OSWorkerProcs: dispatchPublishWorkerLoad(root, product),
		WIP:           wip,
	}
	// The operator concurrency setpoint (#4036, wired live by #4165): fold
	// FAK_DISPATCH_SETPOINT through the pure ReconcileSetpoint plan into the input
	// BEFORE evaluation, so an operator-written level actually moves admits. An
	// unset/blank/malformed setpoint yields the inactive plan and leaves the input
	// untouched -- byte-identical to before the knob existed.
	in, setpointPlan := dispatchFoldSetpoint(in, os.Getenv(dispatchtick.SetpointConcurrencyEnv))
	// The slow predictive loop (#3368, two-timescale scaling): compute a forecast-driven
	// worker FLOOR from a target issue-throughput rate via fleetcap's Little's-law forecast
	// and fold it into the input the fast reactive tick clamps UP to. This is the live
	// PRODUCER the #3368 seam was built for: EvaluatePreflight already clamps capacity up to
	// WorkerFloor (bounded by the HARD host/seat/config ceilings, never overbooking), but the
	// term stayed inert at 0 without a signal -- so cap_terms.limiting == "floor" was
	// unreachable in production from a forecast. FAK_FLEET_TARGET_IPH is the slow-cadence
	// demand an operator or a slow scheduler writes (it changes rarely; the kernel lease
	// target the reactive min() reads changes every tick -- the two timescales); Little's law
	// L = lambda*W turns it into the workers that must be pre-warmed for the ramp. It composes
	// via max with any operator-setpoint floor already in WorkerFloor (the higher wins). Unset
	// FAK_FLEET_TARGET_IPH -> a 0 forecast -> the input is untouched, byte-identical to before
	// this producer existed. The tick-lead / never-dip-below property is unit-witnessed in
	// dispatchtick.TestEvaluatePreflightForecastFloorRaisesReactiveTick.
	in, forecast := dispatchFoldForecastFloor(in, dispatchForecastTargetIPH(), dispatchForecastSessionMinutes())
	// The dual-cadence lazy pull (#3371): this call is the BASE tick; each registered
	// consumer of an EXPENSIVE observation (the gate, rate_budget, and fresh_seat
	// folds below) declares its needs plus an execution interval, and a costly probe
	// is gathered only on a tick where a currently-registered DUE consumer declares
	// it -- between due ticks the fold consumes the LAST pulled observation instead
	// of re-scanning. Unset FAK_DISPATCH_CADENCE leaves every consumer due every
	// tick, so every probe is still gathered every tick -- byte-identical to before
	// the due-filter existed.
	slow, lazySkipped := dispatchGatherSlowProbes(root, product, time.Now())
	// The fifth cap term (#2221, G3 of epic #2218): fold the MEASURED guard-hook
	// latency rollup UP into admission so a slow kernel earns spawn reluctance. The
	// four in-struct terms only flow caps DOWN; this composes gate health on top and
	// can only lower the effective cap, never raise it.
	res := dispatchtick.EvaluatePreflight(in)
	res = dispatchtick.ApplyGateBackpressure(res, slow.Gate)
	// The rate_budget cap term (docs/safe-to-raise-cap-checklist.md): fold the MEASURED,
	// backend-scoped burst of GENUINE concurrency rate-limit worker exits UP into
	// admission so a fleet storming a throttled seat backs off (and routes to another
	// provider) instead of re-storming it. Fake 429s -- weekly caps, model caps, login
	// walls -- are excluded by the reason=rate_limit taxonomy filter; it only lowers the
	// effective cap, so a zero-signal fold is byte-identical to before.
	res = dispatchtick.ApplyRateLimitBackpressure(res, slow.RateLimit)
	// The host_churn cap term: fold the MEASURED whole-host process-spawn burst DOWN into
	// admission so a new dispatcher backs off when the box is ALREADY in a spawn storm --
	// typically several independent dispatchers co-launching waves in the same window, the
	// cross-dispatcher case the per-loop cadence floor cannot see. The signal is read from
	// the cheap stallscan self-monitor ledger (sampled by a background loop, so reading its
	// tail spawns nothing on this hot path); a missing/stale reading yields a zero-pressure
	// ChurnCheck (the fold abstains), so a box without the self-monitor is byte-identical to
	// before. It only lowers the effective cap, never raises it.
	churnCheck, churnArming := dispatchPreflightChurnState()
	res = dispatchtick.ApplyChurnBackpressure(res, churnCheck)
	// The fresh_seat cap term (#3579): fold fleetaccounts' AUTHORITATIVE fresh-account
	// ceiling (BuildCapacityPreflight's TrueConcurrentCeiling -- distinct session slots
	// that can actually serve fresh) into the launch cap. The seat gate above counts
	// session-LEASE slots; when most accounts are walled the lease pool can still show
	// free slots, so admission would size a wave larger than the accounts that can
	// serve and birth clustered REFUSE_NO_ACCOUNT non-starters (the "seats not slots"
	// trap). min-of-limits via fleetcap.AvailableFrom: the term only LOWERS the
	// effective cap; a healthy pool (ceiling >= cap) or an absent roster (ceiling 0,
	// the fold abstains) leaves the preflight byte-identical to before.
	res = dispatchApplyFreshSeatCeiling(res, slow.FreshSeatCeiling)
	out := decorateDispatchPreflight(dispatchPreflightReadout{
		root: root, stderr: stderr, workKind: workKind, product: product,
		result: res, churnCheck: churnCheck, churnArming: churnArming,
		setpointPlan: setpointPlan, forecast: forecast, lazySkipped: lazySkipped,
	})
	return out, timings, nil
}

func dispatchPreflightTree(ctx context.Context, root string, treeOverride *dispatchtick.TreeCheck) dispatchtick.TreeCheck {
	if treeOverride != nil {
		return *treeOverride
	}
	tree, _ := dispatchProbeTreeBuildContext(ctx, root)
	// The lower-level probe preserves infrastructure detail in TreeCheck.Error. A caller
	// deadline is different: it is the dispatch budget doing its job, not tree evidence.
	// Clear only the matching cancellation detail; a compiler poison verdict and every
	// non-cancellation infrastructure error keep their existing diagnostics.
	if !tree.Poisoned && ctx.Err() != nil {
		detail := strings.TrimSpace(tree.Error)
		canceled := ctx.Err().Error()
		if detail == canceled || strings.HasSuffix(detail, ": "+canceled) {
			return dispatchtick.TreeCheck{}
		}
	}
	return tree
}

// dispatchSetpointLive mirrors EvaluatePreflight's live fold EXACTLY -- the max of
// the kernel's alive count (counted only while a positive lease target is armed) and
// the OS worker census -- so the setpoint reconcile classifies grow/steady/drain
// against the SAME live number the preflight itself will judge and report. Drift here
// would let a setpoint misread a drain as a grow (or vice versa).
func dispatchSetpointLive(in dispatchtick.PreflightInput) int {
	live := in.OSWorkerProcs
	if in.Kernel.Target != nil && *in.Kernel.Target > 0 && in.Kernel.Alive != nil && *in.Kernel.Alive > live {
		live = *in.Kernel.Alive
	}
	return maxInt(live, 0)
}

// dispatchFoldSetpoint folds the operator-written concurrency setpoint (#4036,
// FAK_DISPATCH_SETPOINT) into the live preflight input -- the wiring that makes the
// pure ReconcileSetpoint plan actually move admitted workers (#4165).
//
// ParseConcurrencySetpoint yields 0 for a blank/malformed/non-positive value and
// ReconcileSetpoint returns the inactive plan for 0, so an unset knob is a total
// no-op. A DRAIN (setpoint below live) feeds the plan's ContractionTarget into the
// #4038 pending-contraction term, capping admits at the post-drain target so no new
// worker lands on capacity being reclaimed -- surplus workers drain as they finish,
// never killed. A GROW (setpoint above live) realizes DesiredCap through the
// ceiling-bounded #3368 worker floor, raising the effective cap toward the setpoint
// over a soft reactive dip WITHOUT ever overbooking the hard config/host/seat
// ceilings (the floor is min()'d against them downstream). Steady changes nothing.
// Existing input terms are only ever tightened (min for a contraction, max for a
// floor), never loosened.
func dispatchFoldSetpoint(in dispatchtick.PreflightInput, raw string) (dispatchtick.PreflightInput, dispatchtick.SetpointPlan) {
	plan := dispatchtick.ReconcileSetpoint(dispatchSetpointLive(in), dispatchtick.ParseConcurrencySetpoint(raw))
	if !plan.Active {
		return in, plan
	}
	if plan.ContractionTarget > 0 && (in.ContractionTarget <= 0 || plan.ContractionTarget < in.ContractionTarget) {
		in.ContractionTarget = plan.ContractionTarget
	}
	if plan.Mode == "grow" && plan.DesiredCap > in.WorkerFloor {
		in.WorkerFloor = plan.DesiredCap
	}
	return in, plan
}

// dispatchTurnResidencyPricing calculates the residency-discounted admission token cost
// using cacheprice.AdmissionTokens (#3893, vLLM M2 study). A warmer prefix is cheaper to
// schedule — residency directly discounts the admission cost.
func dispatchTurnResidencyPricing(promptTokens, residentPrefixTokens int) map[string]any {
	billable := cacheprice.AdmissionTokens(promptTokens, residentPrefixTokens)
	discount := promptTokens - billable
	discountRate := 0.0
	if promptTokens > 0 {
		discountRate = float64(discount) / float64(promptTokens)
	}
	return map[string]any{
		"prompt_tokens":   promptTokens,
		"resident_tokens": residentPrefixTokens,
		"billable_tokens": billable,
		"discount_tokens": discount,
		"discount_rate":   discountRate,
	}
}

func dispatchTurnResidencyDiscountEnv() map[string]any {
	rawPrompt := strings.TrimSpace(os.Getenv("FAK_DISPATCH_PROMPT_TOKENS"))
	if rawPrompt == "" {
		return nil
	}
	prompt, err := strconv.Atoi(rawPrompt)
	if err != nil || prompt <= 0 {
		return nil
	}
	resident := 0
	if rawRes := strings.TrimSpace(os.Getenv("FAK_DISPATCH_RESIDENT_TOKENS")); rawRes != "" {
		if r, err := strconv.Atoi(rawRes); err == nil {
			resident = r
		}
	}
	return dispatchTurnResidencyPricing(prompt, resident)
}

// forecastFloorPlan is the slow predictive loop's decision (#3368): the Little's-law forecast
// inputs and the worker floor they imply. Active is false (the zero value) when the forecast
// produced no floor -- a non-positive/unset target rate -- so the shell leaves the payload
// untouched and byte-identical to before the producer existed.
type forecastFloorPlan struct {
	Active               bool
	TargetRatePerHour    float64
	MedianSessionMinutes float64
	RequiredWorkers      int
}

// dispatchFoldForecastFloor computes the #3368 forecast-driven worker floor from a target
// issue-throughput rate via fleetcap's Little's-law forecast (L = lambda*W) and folds it into
// the preflight input the reactive tick clamps UP to. It only ever RAISES WorkerFloor (max
// with any operator-setpoint floor already present), never lowers it, and never bounds the
// floor itself -- EvaluatePreflight clamps the applied floor to the hard host/seat/config
// ceilings, so this producer can never overbook the box or the seat pool.
//
// fleetcap.RequiredWorkers returns 0 for a non-positive or non-finite rate/session, so an
// unset FAK_FLEET_TARGET_IPH yields the inactive plan and leaves the input untouched -- a
// total no-op, byte-identical to before the producer had a signal.
func dispatchFoldForecastFloor(in dispatchtick.PreflightInput, targetRatePerHour, medianSessionMinutes float64) (dispatchtick.PreflightInput, forecastFloorPlan) {
	floor := fleetcap.RequiredWorkers(targetRatePerHour, medianSessionMinutes)
	if floor <= 0 {
		return in, forecastFloorPlan{}
	}
	if floor > in.WorkerFloor {
		in.WorkerFloor = floor
	}
	return in, forecastFloorPlan{
		Active:               true,
		TargetRatePerHour:    targetRatePerHour,
		MedianSessionMinutes: medianSessionMinutes,
		RequiredWorkers:      floor,
	}
}

// fleetForecastDefaultSessionMinutes is W in Little's law when FAK_FLEET_SESSION_MIN is unset:
// a 10-minute median agent session, fleetcap's canonical example duration.
const fleetForecastDefaultSessionMinutes = 10.0

// dispatchForecastTargetIPH resolves the slow predictive loop's target issue-throughput rate
// (lambda, issues/hour) from FAK_FLEET_TARGET_IPH. Unset, non-positive, or unparseable yields
// 0 -- no forecast, so the floor producer is a no-op unless an operator/scheduler arms it.
func dispatchForecastTargetIPH() float64 {
	return dispatchEnvPosFloat("FAK_FLEET_TARGET_IPH", 0)
}

// dispatchForecastSessionMinutes resolves W (median session minutes) from FAK_FLEET_SESSION_MIN,
// falling back to fleetForecastDefaultSessionMinutes on empty/non-positive/unparseable input.
func dispatchForecastSessionMinutes() float64 {
	return dispatchEnvPosFloat("FAK_FLEET_SESSION_MIN", fleetForecastDefaultSessionMinutes)
}

// dispatchEnvPosFloat parses a strictly-positive float from env key, returning fallback when the
// value is empty, non-positive, or unparseable (a garbled write can never arm a wrong forecast).
func dispatchEnvPosFloat(key string, fallback float64) float64 {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	if v, err := strconv.ParseFloat(raw, 64); err == nil && v > 0 {
		return v
	}
	return fallback
}

// dispatchPreflightGate folds the workspace's MEASURED guard-hook latency rollup into
// the gate-health state the fifth preflight cap term consults. It reuses the same
// hook-observation streams `fak hooklat` discovers and folds; a missing/unreadable
// stream or a thin sample simply yields a zero-pressure GateCheck (the fold abstains),
// so preflight never grows an error path for an observability signal. The
// overhead-budget breach input stays false here until a standing-breach ledger exists
// to read -- the honest fence: the signal is the measured rollup, never a self-report.
func dispatchPreflightGate(root string) dispatchtick.GateCheck {
	var obs []turntaxmeter.HookObservation
	for _, p := range discoverHookObservationStreams(root) {
		f, err := os.Open(p)
		if err != nil {
			continue
		}
		rows, _, perr := turntaxmeter.ParseHookObservations(f)
		f.Close()
		if perr != nil {
			continue
		}
		obs = append(obs, rows...)
	}
	// Window the fold to LIVE kernel health before judging the tail. Backpressure is
	// a "hold until the tail recovers" signal, but a fold over the ALL-TIME stream can
	// never recover: a past slow period stays in the denominator forever, so the gate
	// would red permanently on a stream that has accumulated one. FAK_GATE_WINDOW (a
	// duration; default dispatchGateDefaultWindow) scopes it to recent rows the same
	// way `fak hooklat --since` does; "0"/"off" restores the whole-stream fold.
	if window := dispatchGateWindow(); window > 0 {
		obs = turntaxmeter.FilterHookObservationsSince(obs, time.Now().Add(-window))
	}
	rollup := turntaxmeter.FoldHookLatency(obs)
	return dispatchtick.GateCheck{
		Hook:         rollup.Total,
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
		MinWorkers:   dispatchGateMinWorkers(),
	}
}

// dispatchGateDefaultWindow scopes the gate's hook-latency fold to recent kernel
// health. Two hours is generous enough to hold a trustworthy tail (n well past
// turntaxmeter.MinHookAlarmSamples on any live fleet) while still letting a resolved
// regression age out so the gate can recover -- the property the all-time fold lacked.
const dispatchGateDefaultWindow = 2 * time.Hour

// dispatchGateWindow resolves the gate's observation lookback from FAK_GATE_WINDOW: a
// Go duration (e.g. "90m") windows the fold; "0" or "off" folds the whole stream; an
// empty or unparseable value falls back to dispatchGateDefaultWindow.
func dispatchGateWindow() time.Duration {
	raw := strings.TrimSpace(os.Getenv("FAK_GATE_WINDOW"))
	switch {
	case raw == "":
		return dispatchGateDefaultWindow
	case raw == "0" || strings.EqualFold(raw, "off"):
		return 0
	}
	if d, err := time.ParseDuration(raw); err == nil && d > 0 {
		return d
	}
	return dispatchGateDefaultWindow
}

// dispatchGateMinWorkers resolves the gate cold-start floor from FAK_GATE_MIN_WORKERS,
// falling back to dispatchtick.DefaultGateMinWorkers. A zero or negative override is
// clamped by GateCheck.floor() back to the default, so the deadlock-at-zero the floor
// forbids cannot be reintroduced through the env.
func dispatchGateMinWorkers() int {
	raw := strings.TrimSpace(os.Getenv("FAK_GATE_MIN_WORKERS"))
	if raw == "" {
		return dispatchtick.DefaultGateMinWorkers
	}
	if n, err := strconv.Atoi(raw); err == nil && n >= 0 {
		return n
	}
	return dispatchtick.DefaultGateMinWorkers
}

// dispatchRateLimitDefaultWindow scopes the rate_budget term's lookback to a RECENT
// burst. A concurrency 429 is transient (a shared seat is momentarily saturated), so the
// window must be short enough that an aged burst stops holding the backend once the
// storm clears -- 15 minutes holds enough recent worker exits to see a genuine cluster
// while letting it age out on its own, the recovery property a whole-stream fold lacks.
const dispatchRateLimitDefaultWindow = 15 * time.Minute

// dispatchRateLimitWindow resolves the rate_budget lookback from FAK_RATELIMIT_WINDOW: a
// Go duration (e.g. "20m") windows the count; "0" or "off" DISABLES the term (zero-value
// fold, a no-op); an empty or unparseable value falls back to the default.
func dispatchRateLimitWindow() time.Duration {
	raw := strings.TrimSpace(os.Getenv("FAK_RATELIMIT_WINDOW"))
	switch {
	case raw == "":
		return dispatchRateLimitDefaultWindow
	case raw == "0" || strings.EqualFold(raw, "off"):
		return 0
	}
	if d, err := time.ParseDuration(raw); err == nil && d > 0 {
		return d
	}
	return dispatchRateLimitDefaultWindow
}

// dispatchRateLimitThreshold resolves the burst arming threshold from
// FAK_RATELIMIT_MIN_429, falling back to dispatchtick.DefaultRateLimitMin429. A zero or
// negative override is ignored (kept at the default) so the term cannot be armed on a
// single stray 429.
func dispatchRateLimitThreshold() int {
	raw := strings.TrimSpace(os.Getenv("FAK_RATELIMIT_MIN_429"))
	if raw == "" {
		return dispatchtick.DefaultRateLimitMin429
	}
	if n, err := strconv.Atoi(raw); err == nil && n > 0 {
		return n
	}
	return dispatchtick.DefaultRateLimitMin429
}

// dispatchRateLimitMinWorkers resolves the cold-start floor from FAK_RATELIMIT_MIN_WORKERS,
// falling back to dispatchtick.DefaultRateLimitMinWorkers. A negative override is ignored;
// the pure fold's floor() re-clamps a zero back to the default, so the one-probe liveness
// carve-out cannot be removed through the env.
func dispatchRateLimitMinWorkers() int {
	raw := strings.TrimSpace(os.Getenv("FAK_RATELIMIT_MIN_WORKERS"))
	if raw == "" {
		return dispatchtick.DefaultRateLimitMinWorkers
	}
	if n, err := strconv.Atoi(raw); err == nil && n >= 0 {
		return n
	}
	return dispatchtick.DefaultRateLimitMinWorkers
}

// dispatchPreflightRateLimit folds the MEASURED, backend-scoped burst of GENUINE
// concurrency rate-limit worker exits into the rate_budget admission term
// (dispatchtick.ApplyRateLimitBackpressure). It counts the finished worker slots whose
// .witness sidecar graded CLAIM_NO_COMMIT with reason=rate_limit (a transient 429/529
// overload the classifier read from the worker log tail -- never a self-report) within a
// recent window on THIS product's backend.
//
// The DISAMBIGUATION is the taxonomy filter (the load-bearing correctness of this term):
// usage_cap (weekly/quota), model_unknown (model cap), and auth_wall (login) are DISTINCT
// classifier reasons and are never counted here, because backing off concurrency does not
// clear any of them -- they are owned by the seat gate, the Layer-2 downgrade ladder, and
// the auth flow. Only reason=rate_limit -- the residual transient-overload class the
// classifier's precedence leaves after skimming those off -- drives concurrency backoff.
//
// Fail-open and byte-identical when idle: a disabled window (FAK_RATELIMIT_WINDOW=0/off),
// a missing runs dir, or zero recent rate_limit exits yields the zero-value check, a no-op
// fold that leaves the preflight untouched.
func dispatchPreflightRateLimit(root, product string) dispatchtick.RateLimitCheck {
	window := dispatchRateLimitWindow()
	if window <= 0 {
		return dispatchtick.RateLimitCheck{}
	}
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if st, err := os.Stat(runsDir); err != nil || !st.IsDir() {
		return dispatchtick.RateLimitCheck{}
	}
	cutoff := time.Now().Add(-window)
	matches := []string{}
	for _, pattern := range []string{"resolve-*" + dispatchtick.WitnessSidecarSuffix, "repair-*" + dispatchtick.WitnessSidecarSuffix} {
		got, _ := filepath.Glob(filepath.Join(runsDir, pattern))
		matches = append(matches, got...)
	}
	count := 0
	for _, wf := range matches {
		info, err := os.Stat(wf)
		if err != nil || info.ModTime().Before(cutoff) {
			continue // aged out of the window -> not part of the current burst
		}
		stem := strings.TrimSuffix(wf, dispatchtick.WitnessSidecarSuffix)
		if product != "" {
			backend := ""
			if b, err := os.ReadFile(stem + ".backend"); err == nil {
				backend = strings.TrimSpace(string(b))
			}
			if !dispatchBackendInProduct(backend, product) {
				continue // a different backend's 429s must not throttle this one
			}
		}
		doc, err := dispatchReadJSONFile(wf)
		if err != nil {
			continue
		}
		if dispatchStringValue(doc["claim"]) != dispatchtick.ClaimNoCommit {
			continue
		}
		// The disambiguation: ONLY reason=rate_limit -- a weekly/usage cap, a model cap,
		// or a login wall is a different reason and is deliberately not counted.
		if dispatchStringValue(doc["reason"]) != dispatchtick.NoCommitRateLimit {
			continue
		}
		count++
	}
	return dispatchtick.RateLimitCheck{
		Recent:     count,
		Window:     window,
		Threshold:  dispatchRateLimitThreshold(),
		MinWorkers: dispatchRateLimitMinWorkers(),
	}
}

func dispatchPreflightHostFromProcesses(processes dispatchtick.ProcGuardInput) dispatchtick.HostCheck {
	res := dispatchtick.EvaluateProcGuard(processes)
	return dispatchtick.HostCheck{
		Safe:         res.OK,
		Error:        res.CollectError,
		Flagged:      res.ActionableFlaggedCount,
		FlaggedNames: res.ActionableNames(),
	}
}

// dispatchFallbackReadout gathers the live facts for the pure cross-provider failover
// decision (#3575) and returns its legible payload block, or (nil, false) when the feature
// is inert -- so the common preflight payload stays byte-identical to before the knob
// existed. It is inert unless FLEET_DISPATCH_FALLBACK_PRODUCT names a product distinct from
// the primary AND the primary preflight refused REFUSE_NO_ACCOUNT this tick (the seat wall
// the failover targets; any other verdict holds on the primary).
//
// The debounce count is the trailing run of consecutive REFUSE_NO_ACCOUNT tick verdicts for
// THIS product, folded from the durable loop ledger (the same store the #3523 no-seat park
// keys on) plus 1 for the current in-flight tick, which is not yet recorded. The fallback
// pool's own account/seat probe runs ONLY once that debounce is satisfied, so a below-
// threshold refused tick spends no extra probe. Fail-open: an unreadable ledger folds to a
// zero prior count (this tick alone), never an error path.
func dispatchFallbackReadout(root string, stderr io.Writer, workKind, primaryProduct, primaryVerdict string) (map[string]any, bool) {
	fallback := strings.TrimSpace(os.Getenv(dispatchtick.FallbackProductEnv))
	if fallback == "" || fallback == primaryProduct {
		return nil, false
	}
	if primaryVerdict != dispatchtick.PreflightRefuseNoAccount {
		return nil, false
	}
	// Trailing consecutive REFUSE_NO_ACCOUNT verdicts for this product (newest first).
	// Resolve the ledger exactly as recordDispatchTickLoop's writer does
	// (filepath.Join(root, defaultLoopLedger())) so the debounce reader and the tick
	// writer agree on one path.
	events, _ := loopmgr.Load(filepath.Join(root, defaultLoopLedger()))
	reasons := make([]string, 0, len(events))
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.Kind != loopmgr.EventAdmit || dispatchtick.ProductForBackend(ev.Principal) != primaryProduct {
			continue
		}
		reasons = append(reasons, ev.Reason)
	}
	threshold := dispatchtick.DefaultFallbackDebounceTicks
	consecutive := dispatchtick.CountTrailingNoAccountRefusals(reasons) + 1 // + this tick
	in := dispatchtick.FallbackProductInput{
		Enabled:             true,
		PrimaryProduct:      primaryProduct,
		FallbackProduct:     fallback,
		PrimaryVerdict:      primaryVerdict,
		ConsecutiveRefusals: consecutive,
		DebounceThreshold:   threshold,
	}
	// Probe the fallback pool only when the debounce is met (an idle below-threshold tick
	// must not spend an extra account/seat scan against a provider it will not use).
	if consecutive >= threshold {
		acct := dispatchPreflightAccount(root, stderr, workKind, fallback)
		seat := dispatchPreflightSeat(root, stderr, fallback)
		in.FallbackServable = acct.Available && !seat.Depleted
		in.FallbackReason = firstString(acct.Reason, acct.Error)
	}
	return dispatchtick.DecideFallbackProduct(in).Map(), true
}

// dispatchPreflightWIP measures started-and-unfinished work from the same durable
// ledger dispatch ticks write. The operator limit is opt-in; unreadable evidence
// abstains rather than inventing either headroom or pressure.
func dispatchPreflightWIP(root string) dispatchtick.WIPCensus {
	raw := strings.TrimSpace(os.Getenv(dispatchtick.WIPLimitEnv))
	limit, err := strconv.Atoi(raw)
	if raw == "" || err != nil || limit <= 0 {
		return dispatchtick.WIPCensus{}
	}
	ledger := filepath.Join(root, defaultLoopLedger())
	if _, err := os.Stat(ledger); err != nil {
		return dispatchtick.WIPCensus{Limit: limit}
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		return dispatchtick.WIPCensus{Limit: limit}
	}
	status := loopmgr.Summarize(events, time.Now())
	started, inventory := status.WIP()
	return dispatchtick.WIPCensus{Measured: true, Started: started, Inventory: inventory, Limit: limit}
}

func dispatchPreflightAccount(root string, _ io.Writer, workKind, product string) dispatchtick.AccountCheck {
	if product == dispatchtick.MicroBackend {
		return dispatchtick.AccountCheck{Available: true, Tag: "micro-local", Tier: 1, Reason: "offline in-process microagent host"}
	}
	if product == "codex" {
		return dispatchCodexAmbientAccount()
	}
	rows, err := dispatchReadAccountRoster(root)
	if err != nil {
		return dispatchtick.AccountCheck{Available: false, Error: err.Error()}
	}
	route := dispatchtick.RouteAccount(dispatchtick.AccountRouteInput{Rows: rows, Product: product, WorkKind: workKind})
	if !route.OK && route.Reason == "no worker accounts match product filter" {
		route.Reason = dispatchNoWorkerCensusReason(rows, product)
	}
	blocked := make([]string, 0, len(route.BlockedTargetAccounts))
	blockedAccounts := make([]dispatchtick.BlockedAccount, 0, len(route.BlockedTargetAccounts))
	for _, row := range route.BlockedTargetAccounts {
		if row.Tag != "" {
			blocked = append(blocked, row.Tag)
		}
		// Carry the per-account block REASON (throttled / needs-login / at session cap),
		// not just the tag, so a REFUSE_NO_ACCOUNT verdict can name why each seat was
		// refused -- the transparency task-#6 half of the storm/health work.
		blockedAccounts = append(blockedAccounts, dispatchtick.BlockedAccountFromRow(row))
	}
	return dispatchtick.AccountCheck{
		Available:       route.OK,
		Tag:             route.Account.Tag,
		Dir:             route.Account.Dir,
		Tier:            route.SelectedTier,
		Model:           route.Account.Model,
		Reason:          route.Reason,
		Blocked:         blocked,
		BlockedAccounts: blockedAccounts,
		LoginStatus:     route.Account.LoginStatus,
		CanServe:        route.Account.CanServe,
	}
}

func dispatchCodexAmbientAccount() dispatchtick.AccountCheck {
	dir := strings.TrimSpace(os.Getenv("CODEX_HOME"))
	reason := "ambient CODEX_HOME login"
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil || strings.TrimSpace(home) == "" {
			return dispatchtick.AccountCheck{Available: false, Reason: "could not resolve home directory for codex ambient login"}
		}
		dir = filepath.Join(home, ".codex")
		reason = "ambient ~/.codex login"
	}
	if _, err := os.Stat(filepath.Join(dir, "auth.json")); err == nil {
		return dispatchtick.AccountCheck{Available: true, Tag: "codex-ambient", Dir: dir, Tier: 1, Reason: reason}
	}
	return dispatchtick.AccountCheck{Available: false, Dir: dir, Reason: "no auth.json in Codex home - run `codex login`"}
}

func dispatchCodexOAuthSessionCap() int {
	raw := strings.TrimSpace(os.Getenv("FAK_CODEX_OAUTH_SESSIONS"))
	if raw == "" {
		return fallbackCodexOAuthSessions
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return fallbackCodexOAuthSessions
	}
	return n
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func dispatchPreflightSeat(root string, _ io.Writer, product string) dispatchtick.SeatCheck {
	if product == dispatchtick.MicroBackend {
		return dispatchtick.SeatCheck{Total: dispatchtick.IntPtr(1), Free: dispatchtick.IntPtr(1), Leased: dispatchtick.IntPtr(0), Depleted: false}
	}
	if product == "codex" {
		total := dispatchCodexOAuthSessionCap()
		// Capacity and seat admission must use the same attributable worker set.
		// Ambient interactive Codex UIs consume neither fleet WIP nor guarded seats.
		live := dispatchProductWorkerCount(root, product)
		leased := live
		if leased > total {
			leased = total
		}
		return dispatchtick.SeatCheck{
			Total:    dispatchtick.IntPtr(total),
			Free:     dispatchtick.IntPtr(maxInt(0, total-live)),
			Leased:   dispatchtick.IntPtr(leased),
			Depleted: live >= total,
		}
	}
	rows, err := dispatchReadAccountRoster(root)
	if err != nil {
		return dispatchtick.SeatCheck{Error: err.Error()}
	}
	pool := dispatchtick.BuildSeatPool(rows, dispatchLiveSeatLeases(filepath.Join(root, dispatchtick.RunsDirName)), product)
	return dispatchtick.SeatCheck{
		Total:    dispatchtick.IntPtr(pool.TotalSeats),
		Free:     dispatchtick.IntPtr(pool.FreeSeats),
		Leased:   dispatchtick.IntPtr(pool.LeasedSeats),
		Depleted: pool.Depleted,
	}
}

// dispatchFreshSeatLimiting is the cap_terms.limiting value recorded when the
// fleetaccounts fresh-seat ceiling (#3579) is the binding launch-cap term, so a
// downsized or refused wave names WHICH term held it (the acceptance's third leg).
const dispatchFreshSeatLimiting = "fresh_seat"

// dispatchLiveFreshSeatCeiling resolves the AUTHORITATIVE fresh-seat ceiling for
// the product: fleetaccounts.BuildCapacityPreflight(...).TrueConcurrentCeiling, the
// count of fresh distinct session slots a dispatcher may safely size a wave against.
// Codex carries no fleetaccounts roster, so it abstains (0); a missing/empty roster
// also yields 0, which the fold treats as "no signal" (fail-open), so a box without
// the roster stays byte-identical to before this term existed.
func dispatchLiveFreshSeatCeiling(root, product string) int {
	if product == "codex" {
		return 0
	}
	paths := fleetaccounts.ResolvePaths(filepath.Join(root, "tools"))
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	return dispatchFreshSeatCeilingFromRoster(fleetaccounts.AnnotatedRoster(paths.Home, paths.ConfigHome, pol, reg), product)
}

// dispatchFreshSeatCeilingFromRoster is the pure roster->ceiling core (testable with a
// fake seat pool): BuildCapacityPreflight folds annotated rows into fresh/stale/blocked
// session slots and TrueConcurrentCeiling is the fresh count. Required is passed as 0
// because the WAVE cap fold below does the sizing; only the ceiling is consumed here.
func dispatchFreshSeatCeilingFromRoster(rows []fleetaccounts.Account, product string) int {
	if len(rows) == 0 {
		return 0
	}
	return fleetaccounts.BuildCapacityPreflight(rows, product, 0).TrueConcurrentCeiling
}

// dispatchApplyFreshSeatCeiling folds the fresh-seat ceiling (#3579) into an
// already-evaluated preflight as an additional lowering cap term, making the effective
// launch cap min(host cap, session slots, fresh-seat ceiling) via fleetcap.AvailableFrom
// (the MIN-of-positive-limits helper built for exactly this). Monotonic like every other
// preflight term: it can only LOWER the cap, never raise it.
//
// Abstains (byte-identical result) when the preflight already refused for a
// higher-precedence reason, when the ceiling carries no signal (<= 0: codex, an absent
// roster), or when the ceiling is not binding (>= the existing cap -- a healthy pool).
// A binding ceiling downsizes the wave and records itself as cap_terms.limiting; when it
// leaves no headroom above the live count the verdict flips to REFUSE_NO_SEAT with a
// reason naming the ceiling, so the fleet stops bursting into walled seats instead of
// birthing clustered REFUSE_NO_ACCOUNT non-starters.
func dispatchApplyFreshSeatCeiling(res dispatchtick.PreflightResult, ceiling int) dispatchtick.PreflightResult {
	if !res.OK || ceiling <= 0 {
		return res
	}
	hold := fleetcap.AvailableFrom(res.Cap, ceiling)
	if hold >= res.Cap {
		return res // healthy pool: the ceiling is not the binding term
	}
	res.Cap = hold
	res.Headroom = hold - res.Live
	res.CapTerms.EffectiveCap = hold
	res.CapTerms.Limiting = dispatchFreshSeatLimiting
	if res.Headroom > 0 {
		return res // downsized wave, still SPAWN_OK; cap_terms records the binding term
	}
	res.OK = false
	res.Verdict = dispatchtick.PreflightRefuseNoSeat
	res.Reason = fmt.Sprintf("fresh-seat ceiling %d <= %d live worker(s): the lease pool may still show free session slots, but only %d fresh distinct seat(s) can actually serve -- refusing rather than launch onto walled seats (a seat frees when an account's wall resets)",
		ceiling, res.Live, ceiling)
	return res
}

// ---- Dual-cadence lazy pull (#3371) ----
//
// One BASE tick (each dispatchPreflightTimed call) drives every consumer, but each
// consumer of an EXPENSIVE observation carries its own execution interval; a costly
// probe (a hook-observation file scan, a witness-sidecar sweep, a fleetaccounts
// roster annotation) is gathered only on a tick where a due consumer declares a need
// for it. Between due ticks the consumer's fold runs on the LAST pulled observation
// -- the term never silently vanishes, only the gather is elided. Clean-room Go of
// the dual-cadence planner pattern: due = never-run or interval elapsed; needed =
// any due consumer's declared need equals the probe path or nests under it.

// Probe paths the slow consumers declare. Dotted paths so a need can address a
// nested observation under a probe (the exact-or-prefix match in
// lazyPullProbeNeeded).
const (
	dispatchProbePathHooklat   = "gate.hooklat"
	dispatchProbePathRateExits = "rate_limit.worker_exits"
	dispatchProbePathFreshSeat = "seat.fresh_ceiling"
)

// dispatchLazyCadenceEnv arms the dual cadence: a Go duration (e.g. "60s") is
// the execution interval of every registered slow consumer; unset, "0", or "off"
// leaves the due-filter disabled -- every consumer due every base tick, so every
// probe is gathered every tick exactly as before #3371.
const dispatchLazyCadenceEnv = "FAK_DISPATCH_CADENCE"

// dispatchLazyCadence resolves the per-consumer execution interval from
// dispatchLazyCadenceEnv. Empty/"0"/"off"/unparseable/non-positive all yield 0
// (disabled), so a garbled write can never starve a fold of a fresh observation.
func dispatchLazyCadence() time.Duration {
	raw := strings.TrimSpace(os.Getenv(dispatchLazyCadenceEnv))
	if raw == "" || raw == "0" || strings.EqualFold(raw, "off") {
		return 0
	}
	if d, err := time.ParseDuration(raw); err == nil && d > 0 {
		return d
	}
	return 0
}

// lazyPullConsumer is one registered consumer of the preflight's expensive
// observations: the probe paths it Needs and the Interval at which it executes. A
// non-positive Interval means due every base tick.
type lazyPullConsumer struct {
	Name     string
	Needs    []string
	Interval time.Duration
}

// dispatchLazyPullConsumers is the currently-registered consumer set: the three
// slow folds wired through dispatchGatherSlowProbes, each declaring exactly the
// expensive observation it folds. All share the one operator cadence.
func dispatchLazyPullConsumers() []lazyPullConsumer {
	interval := dispatchLazyCadence()
	return []lazyPullConsumer{
		{Name: "hooklat_backpressure", Needs: []string{dispatchProbePathHooklat}, Interval: interval},
		{Name: "rate_budget", Needs: []string{dispatchProbePathRateExits}, Interval: interval},
		{Name: "fresh_seat", Needs: []string{dispatchProbePathFreshSeat}, Interval: interval},
	}
}

// lazyPullConsumerDue reports whether a consumer is due at 'at': never run (zero
// lastRun), a non-positive interval (every tick), or its interval elapsed.
func lazyPullConsumerDue(c lazyPullConsumer, lastRun, at time.Time) bool {
	if lastRun.IsZero() || c.Interval <= 0 {
		return true
	}
	return at.Sub(lastRun) >= c.Interval
}

// dueLazyPullConsumers filters the registered consumers to those due at 'at' given
// each consumer's last execution instant.
func dueLazyPullConsumers(consumers []lazyPullConsumer, lastRun map[string]time.Time, at time.Time) []lazyPullConsumer {
	due := make([]lazyPullConsumer, 0, len(consumers))
	for _, c := range consumers {
		if lazyPullConsumerDue(c, lastRun[c.Name], at) {
			due = append(due, c)
		}
	}
	return due
}

// lazyPullProbeNeeded reports whether any DUE consumer declares the probe: a need
// matches by exact path or by nesting under it (need "gate.hooklat.p99" pulls probe
// "gate.hooklat"; a sibling sharing a string prefix does not match). No due consumer
// declaring it means the expensive gather is skipped this tick.
func lazyPullProbeNeeded(due []lazyPullConsumer, probe string) bool {
	for _, c := range due {
		for _, n := range c.Needs {
			if n == probe || strings.HasPrefix(n, probe+".") {
				return true
			}
		}
	}
	return false
}

// dispatchSlowProbes is the typed bundle of the expensive slow-cadence
// observations the preflight folds consume. Every zero value is the corresponding
// fold's fail-open abstain, so a probe that has never been gathered can only leave
// the preflight untouched, never wrongly gate it.
type dispatchSlowProbes struct {
	Gate             dispatchtick.GateCheck
	RateLimit        dispatchtick.RateLimitCheck
	FreshSeatCeiling int
}

// dispatchLazyPullState is the process-lifetime dual-cadence state: which (root,
// product) scope the cache belongs to, each consumer's last execution instant, and
// the last pulled observations. A scope change drops everything -- another
// workspace's or backend's observations must never be served.
var dispatchLazyPullState = struct {
	sync.Mutex
	root    string
	product string
	lastRun map[string]time.Time
	probes  dispatchSlowProbes
}{}

// dispatchGatherSlowProbes is the lazy pull itself: compute the due consumer set for
// this base tick, gather ONLY the expensive probes a due consumer declares (serving
// the last pulled observation for the rest), and mark the due consumers executed.
// It returns the observation bundle plus the probe paths skipped this tick (empty
// whenever the cadence is unarmed -- every consumer due -- so the caller's payload
// stays byte-identical in the common case).
func dispatchGatherSlowProbes(root, product string, at time.Time) (dispatchSlowProbes, []string) {
	s := &dispatchLazyPullState
	s.Lock()
	defer s.Unlock()
	if s.root != root || s.product != product {
		s.root, s.product = root, product
		s.lastRun = map[string]time.Time{}
		s.probes = dispatchSlowProbes{}
	}
	if s.lastRun == nil {
		s.lastRun = map[string]time.Time{}
	}
	due := dueLazyPullConsumers(dispatchLazyPullConsumers(), s.lastRun, at)
	var skipped []string
	pull := func(probe string, gather func()) {
		if lazyPullProbeNeeded(due, probe) {
			gather()
			return
		}
		skipped = append(skipped, probe)
	}
	pull(dispatchProbePathHooklat, func() { s.probes.Gate = dispatchProbeHooklat(root) })
	pull(dispatchProbePathRateExits, func() { s.probes.RateLimit = dispatchProbeRateLimit(root, product) })
	pull(dispatchProbePathFreshSeat, func() { s.probes.FreshSeatCeiling = dispatchProbeFreshSeat(root, product) })
	for _, c := range due {
		s.lastRun[c.Name] = at
	}
	return s.probes, skipped
}

var dispatchKernelCache = struct {
	sync.Mutex
	at    time.Time
	root  string
	check dispatchtick.KernelCheck
}{}

const dispatchKernelCacheTTL = 30 * time.Second

func dispatchPreflightKernel(root string) dispatchtick.KernelCheck {
	now := time.Now()
	dispatchKernelCache.Lock()
	defer dispatchKernelCache.Unlock()
	if dispatchKernelCache.root == root && !dispatchKernelCache.at.IsZero() && now.Sub(dispatchKernelCache.at) < dispatchKernelCacheTTL {
		return dispatchKernelCache.check
	}
	doc, err := dispatchRunExternalJSON(root, 60*time.Second, "dos", "loop", "--workspace", root, "--json")
	check := dispatchtick.KernelCheck{}
	if err != nil {
		check.Error = err.Error()
	} else {
		check = dispatchtick.KernelCheck{Alive: intPtrFromAny(doc["alive"]), Target: intPtrFromAny(doc["target"]), Verdict: dispatchMapString(doc, "verdict")}
	}
	dispatchKernelCache.root, dispatchKernelCache.at, dispatchKernelCache.check = root, now, check
	return check
}

// The change-gated + debounced worker-load publish this reads (dispatchPublishWorkerLoad)
// lives in dispatch_tick_load_debounce.go — one concern, split out to keep this file under
// the god-file ceiling.

var dispatchRunExternalJSON = dispatchRunExternalJSONImpl
var dispatchProbeHostResources = dispatchPreflightHostResources
var dispatchProbeWorkerCount = dispatchProductWorkerCount
var dispatchProbeProcesses = dispatchProbeProcessesNative
var dispatchProbeCodexProcessRows = dispatchScanCodexProcessRowsNative
var dispatchProbeWorkerProcessRows = dispatchScanWorkerProcessRowsNative
var dispatchReadAccountRoster = dispatchReadAccountRosterNative

// The expensive slow-cadence probe seams the #3371 due-filter gathers through, so a
// test can witness a gather happening (or being skipped) without a live host.
var dispatchProbeHooklat = dispatchPreflightGate
var dispatchProbeRateLimit = dispatchPreflightRateLimit
var dispatchProbeFreshSeat = dispatchLiveFreshSeatCeiling

// dispatchReadAccountRosterNative builds the dispatch account roster and then drops

func dispatchStringPtrValue(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

func dispatchIntPtrValue(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func dispatchBoolPtrValue(p *bool) bool {
	return p != nil && *p
}

func dispatchReadJSONFile(path string) (map[string]any, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.UseNumber()
	var doc map[string]any
	if err := dec.Decode(&doc); err != nil {
		return nil, err
	}
	if doc == nil {
		return nil, errors.New("json document is not an object")
	}
	return doc, nil
}

func dispatchLiveSeatLeases(runsDir string) []dispatchtick.SeatLease {
	st, err := os.Stat(runsDir)
	if err != nil || !st.IsDir() {
		return nil
	}
	matches := dispatchWorkerPIDFiles(runsDir)
	sort.Strings(matches)
	leases := make([]dispatchtick.SeatLease, 0, len(matches))
	for _, pidFile := range matches {
		if !dispatchResolvePIDRE.MatchString(filepath.Base(pidFile)) {
			continue
		}
		pid, ok := readPID(pidFile)
		if !ok || !dispatchPIDAlive(pid) {
			continue
		}
		stem := strings.TrimSuffix(pidFile, filepath.Ext(pidFile))
		lease := dispatchtick.SeatLease{Worker: filepath.Base(stem), PID: pid}
		if b, err := os.ReadFile(stem + dispatchtick.AccountSidecarSuffix); err == nil {
			var rec map[string]any
			if json.Unmarshal(b, &rec) == nil {
				lease.Tag = dispatchStringValue(rec["tag"])
				lease.Dir = dispatchStringValue(rec["dir"])
			}
		}
		leases = append(leases, lease)
	}
	return leases
}
