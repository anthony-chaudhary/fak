package main

import (
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

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/fleetcap"
	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

const fallbackCodexOAuthSessions = 10

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
	in := dispatchtick.PreflightInput{
		Workspace:     root,
		MaxWorkers:    maxWorkers,
		Host:          dispatchPreflightHostFromProcesses(processes),
		Tree:          dispatchProbeTreeBuild(root),
		Account:       dispatchPreflightAccount(root, stderr, workKind, product),
		Kernel:        kernel,
		Seat:          dispatchPreflightSeat(root, stderr, product),
		Resources:     dispatchBuildHostResources(processes),
		Budgets:       dispatchtick.DefaultHostBudgets(),
		OSWorkerProcs: dispatchProbeWorkerCount(root, product),
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
	// The fifth cap term (#2221, G3 of epic #2218): fold the MEASURED guard-hook
	// latency rollup UP into admission so a slow kernel earns spawn reluctance. The
	// four in-struct terms only flow caps DOWN; this composes gate health on top and
	// can only lower the effective cap, never raise it.
	res := dispatchtick.EvaluatePreflight(in)
	res = dispatchtick.ApplyGateBackpressure(res, dispatchPreflightGate(root))
	// The rate_budget cap term (docs/safe-to-raise-cap-checklist.md): fold the MEASURED,
	// backend-scoped burst of GENUINE concurrency rate-limit worker exits UP into
	// admission so a fleet storming a throttled seat backs off (and routes to another
	// provider) instead of re-storming it. Fake 429s -- weekly caps, model caps, login
	// walls -- are excluded by the reason=rate_limit taxonomy filter; it only lowers the
	// effective cap, so a zero-signal fold is byte-identical to before.
	res = dispatchtick.ApplyRateLimitBackpressure(res, dispatchPreflightRateLimit(root, product))
	// The host_churn cap term: fold the MEASURED whole-host process-spawn burst DOWN into
	// admission so a new dispatcher backs off when the box is ALREADY in a spawn storm --
	// typically several independent dispatchers co-launching waves in the same window, the
	// cross-dispatcher case the per-loop cadence floor cannot see. The signal is read from
	// the cheap stallscan self-monitor ledger (sampled by a background loop, so reading its
	// tail spawns nothing on this hot path); a missing/stale reading yields a zero-pressure
	// ChurnCheck (the fold abstains), so a box without the self-monitor is byte-identical to
	// before. It only lowers the effective cap, never raises it.
	res = dispatchtick.ApplyChurnBackpressure(res, dispatchPreflightChurn())
	out := res.Map()
	// #3109 self-heal: preflight is otherwise refuse-only on unattributed_live -- it
	// counts orphaned worker PIDs (a botched teardown's `claude` descendant still
	// carrying the dispatch marker but holding NO seat lease) as pool depletion and
	// wedges dispatch until a separately-scheduled janitor clears them. When the pool
	// shows unattributed_live > 0, surface those exact PIDs as a janitor worklist
	// (mirroring how `fak garden tick` surfaces orphan-run worklists) so the tick can
	// tree-reap them via procguard.KillPID and the pool recovers on its own next tick.
	// The predicate is the SAME one preflight already uses to COUNT them -- dispatch
	// marker AND no live lease -- so a leased or unrelated process can never be swept.
	// Observation only here (no side effect); the reap is next-tick / live-gated in the
	// dispatch tick, never in this hot admission path (mis-attributed-kill TOCTOU).
	if res.Seat.UnattributedLive > 0 {
		if worklist := dispatchUnattributedWorklist(dispatchProductWorkerPIDs(root, product), dispatchLeasedWorkerPIDs(root)); len(worklist) > 0 {
			out["janitor_worklist"] = worklist
		}
	}
	// The usage_cap ADVISORY term (advisory-only, folds nothing): when a majority of the
	// fleet's accounts sit under an active usage-limit cooldown, attach a note so an
	// operator/loop sees the pool is quota-exhausted -- and that a fresh spawn walling here
	// is a usage cap the witness classifier will likely mislabel reason=rate_limit, which
	// concurrency backoff cannot clear. It never touches the verdict or cap; the field is
	// added ONLY when armed, so the common preflight payload stays byte-identical.
	if adv := dispatchPreflightUsageCap(root, product, res.Seat).Note(); adv != nil {
		out["usage_cap_advisory"] = adv
	}
	// Surface the ACTIVE setpoint plan on the payload so an operator (and the tick
	// log) can see WHY the cap moved -- which branch (grow/steady/drain), the level
	// converged toward, and how many surplus workers are draining. Attached only when
	// a setpoint is active, so the common payload stays byte-identical.
	if setpointPlan.Active {
		out["setpoint"] = map[string]any{
			"mode":               setpointPlan.Mode,
			"desired_cap":        setpointPlan.DesiredCap,
			"contraction_target": setpointPlan.ContractionTarget,
			"draining":           setpointPlan.Draining,
		}
	}
	// Surface the ACTIVE forecast floor on the payload so an operator (and the tick log) can
	// see the slow predictive loop pre-warmed capacity, and the RAW forecast (required_workers)
	// vs what the hard ceilings allowed (cap_terms.worker_floor is the ceiling-bounded value).
	// Attached only when the forecast produced a floor, so the common payload stays byte-identical.
	if forecast.Active {
		out["forecast_floor"] = map[string]any{
			"target_iph":       forecast.TargetRatePerHour,
			"session_min":      forecast.MedianSessionMinutes,
			"required_workers": forecast.RequiredWorkers,
		}
	}
	return out, timings, nil
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

// dispatchChurnDefaultFreshness bounds how old the stallscan self-monitor reading may be
// and still gate admission. The burst signal is a point-in-time host census; a reading
// older than a couple sample intervals no longer describes the CURRENT scheduler load, and
// gating on a stale storm would wrongly freeze a fleet whose burst has long since drained.
// A reading older than this yields a zero-value ChurnCheck (the fold abstains). The impure
// shell overlays FAK_CHURN_FRESHNESS.
const dispatchChurnDefaultFreshness = 90 * time.Second

// dispatchChurnFreshness resolves the max age of a usable stallscan reading from
// FAK_CHURN_FRESHNESS (a Go duration; "0"/"off" disables the freshness gate and trusts the
// last reading regardless of age), falling back to the default on empty/unparseable input.
func dispatchChurnFreshness() time.Duration {
	raw := strings.TrimSpace(os.Getenv("FAK_CHURN_FRESHNESS"))
	switch {
	case raw == "":
		return dispatchChurnDefaultFreshness
	case raw == "0" || strings.EqualFold(raw, "off"):
		return 0
	}
	if d, err := time.ParseDuration(raw); err == nil && d > 0 {
		return d
	}
	return dispatchChurnDefaultFreshness
}

// dispatchChurnThreshold resolves the spawn-burst arming threshold from
// FAK_CHURN_BURST_THRESHOLD, falling back to dispatchtick.DefaultChurnBurstThreshold. A zero
// or negative override is ignored so the term cannot be armed on ordinary process turnover.
// A value of "off" disables the term entirely (returns 0 -> the shell yields a zero-value
// check that never gates).
func dispatchChurnThreshold() int {
	raw := strings.TrimSpace(os.Getenv("FAK_CHURN_BURST_THRESHOLD"))
	if raw == "" {
		return dispatchtick.DefaultChurnBurstThreshold
	}
	if strings.EqualFold(raw, "off") {
		return 0
	}
	if n, err := strconv.Atoi(raw); err == nil && n > 0 {
		return n
	}
	return dispatchtick.DefaultChurnBurstThreshold
}

// dispatchChurnMinWorkers resolves the cold-start floor from FAK_CHURN_MIN_WORKERS, falling
// back to dispatchtick.DefaultChurnMinWorkers. A negative override is ignored; the pure
// fold's floor() re-clamps a zero back to the default, so the one-probe liveness carve-out
// cannot be removed through the env.
func dispatchChurnMinWorkers() int {
	raw := strings.TrimSpace(os.Getenv("FAK_CHURN_MIN_WORKERS"))
	if raw == "" {
		return dispatchtick.DefaultChurnMinWorkers
	}
	if n, err := strconv.Atoi(raw); err == nil && n >= 0 {
		return n
	}
	return dispatchtick.DefaultChurnMinWorkers
}

// dispatchStallLedgerPath resolves the stallscan self-monitor ledger the churn fold reads.
// It mirrors cmd/fak/stallscan.go's defaultStallLogPath EXACTLY (FAK_STALL_DIR, else the
// Windows LOCALAPPDATA\Fleet location, else ~/.fak) so the reader and the writer agree on
// one path without importing the writer.
func dispatchStallLedgerPath() string {
	if d := os.Getenv("FAK_STALL_DIR"); d != "" {
		return filepath.Join(d, "stallscan.jsonl")
	}
	if la := os.Getenv("LOCALAPPDATA"); la != "" {
		return filepath.Join(la, "Fleet", "stallscan.jsonl")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".fak", "stallscan.jsonl")
}

// dispatchPreflightChurn folds the MEASURED whole-host spawn burst into the host_churn
// admission term (dispatchtick.ApplyChurnBackpressure). It reads the LAST line of the
// stallscan self-monitor ledger -- a background loop samples the host and appends one
// fak.stallscan.v1 record per interval, so reading its tail costs one small file read and
// spawns NOTHING on this hot admission path (the discipline that keeps the anti-churn term
// from adding to the churn it measures).
//
// Fail-open and byte-identical when idle: no ledger (the self-monitor is not running), an
// unreadable/garbled tail, a reading older than the freshness bound (a drained burst must
// not keep gating), or a disabled threshold (FAK_CHURN_BURST_THRESHOLD=off) each yields the
// zero-value check -- a no-op fold that leaves the preflight untouched. A box that never
// runs `fak stallscan --watch` therefore behaves exactly as before this term existed.
func dispatchPreflightChurn() dispatchtick.ChurnCheck {
	threshold := dispatchChurnThreshold()
	if threshold <= 0 {
		return dispatchtick.ChurnCheck{} // term disabled via env
	}
	line := dispatchLastLine(dispatchStallLedgerPath())
	if line == "" {
		return dispatchtick.ChurnCheck{} // no self-monitor ledger -> abstain
	}
	var rec struct {
		TS     string `json:"ts"`
		Sample struct {
			SpawnBurst int `json:"spawn_burst"`
		} `json:"sample"`
	}
	if err := json.Unmarshal([]byte(line), &rec); err != nil {
		return dispatchtick.ChurnCheck{} // garbled tail -> abstain
	}
	if fresh := dispatchChurnFreshness(); fresh > 0 {
		ts, err := time.Parse(time.RFC3339Nano, rec.TS)
		if err != nil || time.Since(ts) > fresh {
			return dispatchtick.ChurnCheck{} // stale reading -> the burst it saw may have drained
		}
	}
	return dispatchtick.ChurnCheck{
		Recent:     rec.Sample.SpawnBurst,
		Threshold:  threshold,
		MinWorkers: dispatchChurnMinWorkers(),
	}
}

// dispatchLastLine returns the last non-empty line of a file, or "" if the file is missing,
// empty, or unreadable. It reads only a bounded tail window so a large rolling ledger costs
// a small fixed read rather than an O(file) slurp on the admission path.
func dispatchLastLine(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || info.Size() == 0 {
		return ""
	}
	const tailCap = 64 << 10 // 64 KiB: far more than one fak.stallscan.v1 record
	start := int64(0)
	if info.Size() > tailCap {
		start = info.Size() - tailCap
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return ""
	}
	buf, err := io.ReadAll(f)
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(buf), "\r\n"), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if s := strings.TrimSpace(lines[i]); s != "" {
			return s
		}
	}
	return ""
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

func dispatchPreflightAccount(root string, _ io.Writer, workKind, product string) dispatchtick.AccountCheck {
	if product == "codex" {
		return dispatchCodexAmbientAccount()
	}
	rows, err := dispatchReadAccountRoster(root)
	if err != nil {
		return dispatchtick.AccountCheck{Available: false, Error: err.Error()}
	}
	route := dispatchtick.RouteAccount(dispatchtick.AccountRouteInput{Rows: rows, Product: product, WorkKind: workKind})
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
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return dispatchtick.AccountCheck{Available: false, Reason: "could not resolve home directory for codex ambient login"}
	}
	dir := filepath.Join(home, ".codex")
	if _, err := os.Stat(filepath.Join(dir, "auth.json")); err == nil {
		return dispatchtick.AccountCheck{Available: true, Tag: "codex-ambient", Dir: dir, Tier: 1, Reason: "ambient ~/.codex login"}
	}
	return dispatchtick.AccountCheck{Available: false, Reason: "no ~/.codex/auth.json - run `codex login`"}
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
	if product == "codex" {
		total := dispatchCodexOAuthSessionCap()
		live := dispatchAmbientCodexProcessCount()
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

var dispatchRunExternalJSON = dispatchRunExternalJSONImpl
var dispatchProbeHostResources = dispatchPreflightHostResources
var dispatchProbeWorkerCount = dispatchProductWorkerCount
var dispatchProbeProcesses = dispatchProbeProcessesNative
var dispatchProbeCodexProcessRows = dispatchScanCodexProcessRowsNative
var dispatchProbeWorkerProcessRows = dispatchScanWorkerProcessRowsNative
var dispatchReadAccountRoster = dispatchReadAccountRosterNative

// dispatchReadAccountRosterNative builds the dispatch account roster and then drops
// seats with an ACTIVE account cooldown from the servable pool. Both dispatch pickers
// (dispatchtick.RouteAccount for a bare tick and dispatchtick.AllocateWave for a wave)
// admit only rows whose Available flag survives the login gate; without this overlay
// they route onto a seat the guard already cooled after a usage cap / rehome, which
// immediately 429s and burns the slot. This mirrors the cooldown overlay that
// Registry.LoginReportAt already applies for `fak accounts` and guard rotation
// (internal/accounts/login.go), keyed on the same uuid: bucket the guard writes.
func dispatchReadAccountRosterNative(root string) ([]dispatchtick.AccountRow, error) {
	rows, err := dispatchBuildAccountRoster(root)
	if err != nil {
		return nil, err
	}
	return dispatchApplyAccountCooldown(rows, loadCooldownStoreFailOpen(), time.Now()), nil
}

// dispatchApplyAccountCooldown marks every roster row whose upstream account holds an
// active cooldown at now as unservable (Available=false, CanServe=false) with a block
// reason, so RouteAccount/AllocateWave skip it. A nil store leaves the roster untouched.
func dispatchApplyAccountCooldown(rows []dispatchtick.AccountRow, store *accounts.CooldownStore, now time.Time) []dispatchtick.AccountRow {
	if store == nil {
		return rows
	}
	for i := range rows {
		uuid := strings.TrimSpace(rows[i].AccountUUID)
		if uuid == "" {
			continue
		}
		entry, cooled := store.CooledDown(accounts.UUIDBucketKey(uuid), now)
		if !cooled {
			continue
		}
		rows[i].Available = false
		canServe := false
		rows[i].CanServe = &canServe
		if strings.TrimSpace(rows[i].BlockReason) == "" {
			rows[i].BlockReason = dispatchCooldownBlockReason(entry, now)
		}
	}
	return rows
}

// dispatchCooldownBlockReason renders a concise, audit-friendly block reason for a
// cooled seat, naming the cooldown kind and the reset instant.
func dispatchCooldownBlockReason(e accounts.CooldownEntry, now time.Time) string {
	remaining := e.ResetAt.Sub(now)
	if remaining < 0 {
		remaining = 0
	}
	return fmt.Sprintf("account in cooldown (%s) until %s (%s remaining)",
		e.Kind, e.ResetAt.UTC().Format(time.RFC3339), remaining.Round(time.Minute))
}

// dispatchUsageCapAdvisoryThreshold resolves the usage-cap advisory's arming floor from
// FAK_USAGECAP_ADVISORY_MIN (a positive integer count of usage-limit-cooled accounts),
// falling back to dispatchtick.DefaultUsageCapAdvisoryMin on empty/zero/unparseable input.
func dispatchUsageCapAdvisoryThreshold() int {
	raw := strings.TrimSpace(os.Getenv("FAK_USAGECAP_ADVISORY_MIN"))
	if raw == "" {
		return dispatchtick.DefaultUsageCapAdvisoryMin
	}
	if n, err := strconv.Atoi(raw); err == nil && n > 0 {
		return n
	}
	return dispatchtick.DefaultUsageCapAdvisoryMin
}

// dispatchPreflightUsageCap builds the ADVISORY-ONLY usage-cap census the preflight
// surfaces (dispatchtick.UsageCapAdvisory). Unlike the rate_budget term it folds nothing
// into the cap -- it reads the AUTHORITATIVE account-cooldown store (the one signal that
// distinguishes a usage cap from the transient 429 the witness classifier often mislabels
// it as) and reports how many of the backend's routable accounts sit under an active
// usage-limit-kind cooldown, plus the soonest reset. FreeSeats is carried from the seat
// gate's already-computed pool as context only. Fail-open: a codex backend, an unreadable
// roster, or a nil cooldown store yields a zero census (not armed), so the advisory never
// grows an error path and a fleet without the store is byte-identical to before.
func dispatchPreflightUsageCap(root, product string, seat dispatchtick.SeatCheck) dispatchtick.UsageCapAdvisory {
	// Seat-cooldown usage caps are a Claude-seat concept; codex carries no usage-limit
	// cooldown store, so the census abstains rather than counting an unrelated store.
	if product == "codex" {
		return dispatchtick.UsageCapAdvisory{}
	}
	store := loadCooldownStoreFailOpen()
	if store == nil {
		return dispatchtick.UsageCapAdvisory{}
	}
	rows, err := dispatchBuildAccountRoster(root)
	if err != nil {
		return dispatchtick.UsageCapAdvisory{}
	}
	free := 0
	if seat.Free != nil {
		free = *seat.Free
	}
	return dispatchUsageCapCensus(rows, store, product, time.Now(), dispatchUsageCapAdvisoryThreshold(), free)
}

// dispatchUsageCapCensus is the pure-ish counting core of the usage-cap advisory: over the
// backend's routable roster it counts unique accounts (deduped by uuid) and, among them,
// how many sit under an ACTIVE usage-limit-kind cooldown at now, tracking the soonest
// reset. Kept separate from the store/roster loading so it is testable in-memory with a
// seeded store (mirroring dispatchApplyAccountCooldown). A nil store yields a zero-capped
// census (nothing cooled), so the caller stays fail-open.
func dispatchUsageCapCensus(rows []dispatchtick.AccountRow, store *accounts.CooldownStore, product string, now time.Time, threshold, freeSeats int) dispatchtick.UsageCapAdvisory {
	total, capped := 0, 0
	var earliest time.Time
	seen := map[string]bool{}
	for _, raw := range rows {
		row := dispatchtick.NormalizeAccountRow(raw)
		// Scope the census to this backend's product, mirroring BuildSeatPool's filter, so
		// another product's usage caps never colour this backend's advisory.
		if product != "" && product != "all" && row.Product != product {
			continue
		}
		uuid := strings.TrimSpace(row.AccountUUID)
		if uuid == "" || seen[uuid] {
			continue // dedup by account: one usage cap removes the account, counted once
		}
		seen[uuid] = true
		total++
		if store == nil {
			continue
		}
		entry, cooled := store.CooledDown(accounts.UUIDBucketKey(uuid), now)
		if !cooled || entry.Kind != accounts.CooldownUsageLimit {
			continue
		}
		capped++
		if earliest.IsZero() || entry.ResetAt.Before(earliest) {
			earliest = entry.ResetAt
		}
	}
	return dispatchtick.UsageCapAdvisory{
		Capped:        capped,
		Accounts:      total,
		FreeSeats:     freeSeats,
		EarliestReset: earliest,
		Threshold:     threshold,
		Now:           now,
	}
}

func dispatchBuildAccountRoster(root string) ([]dispatchtick.AccountRow, error) {
	if rows := dispatchAuthoritativeAccountRows(root); len(rows) > 0 {
		return rows, nil
	}
	registryPath := dispatchAccountRegistryPath(root)
	doc, err := dispatchReadJSONFile(registryPath)
	if err != nil {
		return nil, fmt.Errorf("read account registry %s: %w", registryPath, err)
	}
	rawAccounts, _ := doc["accounts"].([]any)
	if len(rawAccounts) == 0 {
		return nil, fmt.Errorf("account registry %s has no accounts array", registryPath)
	}
	weights := dispatchLoadAccountRouteWeights(root)
	rows := make([]dispatchtick.AccountRow, 0, len(rawAccounts))
	for _, item := range rawAccounts {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		row := dispatchtick.AccountRow{
			Account:        dispatchStringValue(m["account"]),
			Tag:            dispatchStringValue(m["tag"]),
			Product:        dispatchStringValue(m["product"]),
			Dir:            firstString(dispatchStringValue(m["config_dir"]), dispatchStringValue(m["dir"])),
			Model:          dispatchStringValue(m["model"]),
			ModelTier:      dispatchIntValue(m["model_tier"]),
			Available:      dispatchBoolValue(m["available"]),
			BlockReason:    firstString(dispatchStringValue(m["block_reason"]), dispatchStringValue(m["reason"])),
			ActiveSessions: dispatchIntValue(m["active_sessions"]),
			LiveSessions:   dispatchIntValue(m["live_sessions"]),
			RouteWeight:    dispatchIntValue(m["route_weight"]),
			IdentityRole:   dispatchStringValue(m["identity_role"]),
			AccountUUID:    dispatchStringValue(m["account_uuid"]),
			LoginStatus:    dispatchStringValue(m["login_status"]),
		}
		if rawCanServe, ok := m["can_serve"]; ok {
			canServe := dispatchBoolValue(rawCanServe)
			row.CanServe = &canServe
		}
		if row.Account == "" && row.Dir != "" {
			row.Account = dispatchAnyOSBase(row.Dir)
		}
		if row.BlockReason == "" && dispatchBoolValue(m["blocked"]) {
			row.BlockReason = "blocked"
		}
		row = dispatchtick.NormalizeAccountRow(row)
		if row.RouteWeight == 0 {
			row.RouteWeight = dispatchAccountRouteWeight(row, weights)
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("account registry %s has no readable account rows", registryPath)
	}
	return rows, nil
}

func dispatchAuthoritativeAccountRows(root string) []dispatchtick.AccountRow {
	toolsDir := filepath.Join(root, "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	accounts := fleetaccounts.AnnotatedRoster(paths.Home, paths.ConfigHome, pol, reg)
	weights := dispatchLoadAccountRouteWeights(root)
	rows := make([]dispatchtick.AccountRow, 0, len(accounts))
	for _, acct := range accounts {
		row := dispatchAccountRowFromFleetAccount(acct)
		if row.Account == "" && row.Dir != "" {
			row.Account = dispatchAnyOSBase(row.Dir)
		}
		row = dispatchtick.NormalizeAccountRow(row)
		if row.RouteWeight == 0 {
			row.RouteWeight = dispatchAccountRouteWeight(row, weights)
		}
		rows = append(rows, row)
	}
	return rows
}

func dispatchAccountRowFromFleetAccount(acct fleetaccounts.Account) dispatchtick.AccountRow {
	row := dispatchtick.AccountRow{
		Account:      acct.Account,
		Tag:          acct.Tag,
		Product:      acct.Product,
		Dir:          acct.Dir,
		Kind:         string(acct.Kind),
		Available:    dispatchBoolPtrValue(acct.Available),
		BlockReason:  firstString(dispatchStringPtrValue(acct.BlockReason), acct.Reason),
		RouteWeight:  dispatchIntPtrValue(acct.RouteWeight),
		IdentityRole: dispatchStringPtrValue(acct.IdentityRole),
		AccountUUID:  dispatchStringPtrValue(acct.AccountUUID),
		LoginStatus:  dispatchStringPtrValue(acct.LoginStatus),
	}
	if acct.Model != nil {
		row.Model = *acct.Model
	}
	if acct.ModelTier != nil {
		row.ModelTier = *acct.ModelTier
	}
	if acct.ActiveSessions != nil {
		row.ActiveSessions = *acct.ActiveSessions
	}
	if acct.LiveSessions != nil {
		row.LiveSessions = *acct.LiveSessions
	}
	if acct.CanServe != nil {
		canServe := *acct.CanServe
		row.CanServe = &canServe
	}
	return row
}

func dispatchAccountRegistryPath(root string) string {
	if dir := strings.TrimSpace(os.Getenv("FLEET_REG_DIR")); dir != "" {
		return filepath.Join(dir, "sessions.json")
	}
	return filepath.Join(root, "tools", "_registry", "sessions.json")
}

func dispatchAccountPolicyPath(root string) string {
	if path := strings.TrimSpace(os.Getenv("FLEET_POLICY_PATH")); path != "" {
		return path
	}
	if dir := strings.TrimSpace(os.Getenv("FLEET_POLICY_DIR")); dir != "" {
		return filepath.Join(dir, "accounts_policy.json")
	}
	return filepath.Join(root, "tools", "_registry", "accounts_policy.json")
}

func dispatchLoadAccountRouteWeights(root string) map[string]int {
	doc, err := dispatchReadJSONFile(dispatchAccountPolicyPath(root))
	if err != nil {
		return nil
	}
	raw, _ := doc["route_weights"].(map[string]any)
	if len(raw) == 0 {
		return nil
	}
	weights := make(map[string]int, len(raw))
	for key, val := range raw {
		weights[key] = dispatchIntValue(val)
	}
	return weights
}

func dispatchAccountRouteWeight(row dispatchtick.AccountRow, weights map[string]int) int {
	if len(weights) == 0 {
		return 0
	}
	product := row.Product
	if product == "" {
		product = dispatchtick.ProductFromAccount(row.Account)
	}
	tag := row.Tag
	if tag == "" {
		tag = dispatchtick.TagFromAccount(row.Account)
	}
	for _, key := range []string{row.Account, product + ":" + row.Account, product + ":" + tag, tag, product} {
		if key == "" {
			continue
		}
		if weight, ok := weights[key]; ok {
			return weight
		}
	}
	return 0
}

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
