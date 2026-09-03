package main

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

type dispatchPreflightReadout struct {
	root, workKind, product string
	stderr                  io.Writer
	result                  dispatchtick.PreflightResult
	churnCheck              dispatchtick.ChurnCheck
	churnArming             stallscan.Arming
	setpointPlan            dispatchtick.SetpointPlan
	forecast                forecastFloorPlan
	lazySkipped             []string
}

func decorateDispatchPreflight(readout dispatchPreflightReadout) map[string]any {
	root, stderr := readout.root, readout.stderr
	workKind, product := readout.workKind, readout.product
	res, churnCheck, churnArming := readout.result, readout.churnCheck, readout.churnArming
	setpointPlan, forecast, lazySkipped := readout.setpointPlan, readout.forecast, readout.lazySkipped
	out := res.Map()
	// Churn ARMING readout. The host_churn term fails open -- an unmeasured host must
	// never block dispatch -- but failing open SILENTLY is how it sat inert for weeks
	// while the box kept freezing: a tick with no ledger and a tick on a genuinely calm
	// host produced byte-identical payloads, so every reader concluded "no churn" when
	// the truth was "no measurement". This block is always present and names which of
	// the two it is, so an absent signal reads as absent rather than as healthy.
	out["host_churn"] = map[string]any{
		"armed":  churnArming.Armed(),
		"state":  string(churnArming.Status()),
		"detail": churnArming.Reason(),
		"age_seconds": func() any {
			if s := churnArming.Status(); s == stallscan.ArmStateMissing || s == stallscan.ArmStateDisabled {
				return nil // nothing to age
			}
			return churnArming.AgeSeconds
		}(),
		"spawn_burst": churnArming.SpawnBurst,
		// Publish the window and the derived rate next to the raw count. A bare count
		// is not comparable across callers sampling at different intervals, so a
		// payload carrying only the count invites a reader to compare two numbers that
		// were never the same quantity. Both are null when the reading predates the
		// window being recorded — which is the honest answer, rather than a rate
		// divided by an assumed interval.
		"spawn_window_seconds": func() any {
			if churnArming.SpawnWindowSeconds <= 0 {
				return nil
			}
			return churnArming.SpawnWindowSeconds
		}(),
		"spawn_rate_per_sec": func() any {
			if churnArming.SpawnWindowSeconds <= 0 {
				return nil
			}
			return float64(churnArming.SpawnBurst) / churnArming.SpawnWindowSeconds
		}(),
		"threshold":      churnCheck.Threshold,
		"rate_threshold": dispatchtick.DefaultChurnBurstRate,
	}
	// Debounce transparency (#3376): when the RAW worker-count probe disagrees with the
	// load that was actually PUBLISHED to admission -- a change still waiting out its
	// coalescing window -- surface the raw sample alongside it, so an operator reading a
	// tick payload can tell a debounced lag from a stale probe. The key is absent
	// whenever the two agree, which is every settled tick, so the steady-state payload is
	// byte-identical to before the debounce existed.
	if raw, lagging := dispatchWorkerLoad.pending(); lagging {
		out["os_worker_procs_sampled"] = raw
	}
	// Cross-provider seat failover readout (#3575, gen/next). When the primary product's
	// seats are walled (REFUSE_NO_ACCOUNT) for a debounced run of ticks and
	// FLEET_DISPATCH_FALLBACK_PRODUCT names a servable alternative pool (e.g. an ambient
	// codex login while the Claude roster is capped), surface the failover decision --
	// which pool refused, which product would take the work -- so the tick can route the
	// launch elsewhere instead of parking on a wall only a peer finishing can move
	// (CONCEPT-IDEAL 1.2). Gated: with the knob unset the helper returns (nil,false) and
	// nothing is attached, so the common preflight payload stays byte-identical. The
	// launch-target SWITCH itself (lease/witness on the fallback backend) is the promotion
	// step the tick consumes this readout for; this shell only decides and reports it.
	if fb, ok := dispatchFallbackReadout(root, stderr, workKind, product, res.Verdict); ok {
		out["fallback"] = fb
	}
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
	// Surface the due-filter's work (#3371) so an operator (and the tick log) can see
	// WHICH expensive probes this base tick served from the last pulled observation
	// instead of re-gathering. Attached only when the armed cadence actually skipped a
	// gather, so the common (unarmed / all-due) payload stays byte-identical.
	if len(lazySkipped) > 0 {
		out["lazy_pull"] = map[string]any{
			"cadence_ms": dispatchLazyCadence().Milliseconds(),
			"skipped":    lazySkipped,
		}
	}
	// Residency discount pricing (#3893, vLLM M2 study): price a turn's admission cost
	// discounted by resident-prefix coverage. Attached when prompt tokens are configured.
	if pricing := dispatchTurnResidencyDiscountEnv(); pricing != nil {
		out["residency_pricing"] = pricing
	}

	return out
}
