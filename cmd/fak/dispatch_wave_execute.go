package main

import (
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

type dispatchWavePlanRequest struct {
	root, backend, workKind, goalID, profile, waveID string
	shortfall                                        int
	router                                           dispatchtick.RouterPayload
	requestedIssues                                  []int
	count, freshStartCap, maxWorkers                 *int
	lanes                                            []dispatchtick.AccountWaveLane
	lane                                             *string
	excludedLanes                                    []string
	intentHolds                                      map[int]string
	noLedger, live, asJSON                           *bool
	codexLoopGate                                    *string
	gateSinceHours                                   *float64
	gateLimit                                        *int
	record                                           map[string]any
}

func planDispatchWave(stdout, stderr io.Writer, req dispatchWavePlanRequest) ([]dispatchWaveExecutionPlan, int, bool) {
	root, backendNorm, wk, goalID, profile := req.root, req.backend, req.workKind, req.goalID, req.profile
	waveID, shortfall, router, requestedIssues := req.waveID, req.shortfall, req.router, req.requestedIssues
	count, lanes, lane, excludedLanes := req.count, req.lanes, req.lane, req.excludedLanes
	freshStartCap, maxWorkers, intentHolds := req.freshStartCap, req.maxWorkers, req.intentHolds
	noLedger, codexLoopGate := req.noLedger, req.codexLoopGate
	codexLoopGateSinceHours, codexLoopGateLimit := req.gateSinceHours, req.gateLimit
	live, asJSON, rec := req.live, req.asJSON, req.record
	var err error
	heldIssues := map[int]bool{}
	var explicitRefusals []dispatchWaveIssueRefusal
	var executionPlan []dispatchWaveExecutionPlan
	const maxPrelaunchReprice = 8
	for attempt := 0; ; attempt++ {
		var price dispatchWavePrice
		if len(requestedIssues) > 0 {
			price, err = priceDispatchWaveExplicitIssues(root, router, requestedIssues, *count, len(lanes), *lane, excludedLanes, dispatchtick.DefaultCooldownMinutes, heldIssues, *freshStartCap, intentHolds, profile)
		} else {
			price, err = priceDispatchWavePayloadFilteredWithFreshCap(root, router, *count, len(lanes), *lane, excludedLanes, dispatchtick.DefaultCooldownMinutes, heldIssues, *freshStartCap, profile)
		}
		if err != nil {
			rec["stop_reason"] = "price fan-out: " + err.Error()
			return nil, writeDispatchWaveResult(stdout, stderr, rec, *asJSON), true
		}
		price = dispatchWaveMergeIssueRefusals(price, explicitRefusals)
		rec["price"] = price
		dispatchWaveAttachExplicitIssueReceipt(rec, price)
		rec["planned_lanes"] = append([]string(nil), price.RunLanes...)
		executionPlan = dispatchWaveExecutionPlans(root, backendNorm, wk, goalID, profile, waveID, shortfall, price.RunTargets, lanes, !*noLedger, *codexLoopGate, maxFloat64(0, *codexLoopGateSinceHours), *codexLoopGateLimit)
		executionPlanID := dispatchWaveExecutionPlanID(executionPlan)
		rec["execution_plan_id"] = executionPlanID
		rec["execution_plan"] = executionPlan
		if len(price.RunLanes) == 0 {
			rec["stop_reason"] = "priced fan-out found no launchable lane"
			return nil, writeDispatchWaveResult(stdout, stderr, rec, *asJSON), true
		}
		requestedExecutionPlan := executionPlan
		executionAudit, auditedPlan, trancheReceipt, auditErr := auditDispatchWaveExecutionPlanWithFallback(executionPlan, func(plan []dispatchWaveExecutionPlan) ([]dispatchWaveExecutionAudit, error) {
			return auditDispatchWaveExecutionPlanBounded(root, *maxWorkers, excludedLanes, plan, *codexLoopGate, maxFloat64(0, *codexLoopGateSinceHours), *codexLoopGateLimit, dispatchWaveDependencyTimeout)
		})
		if len(trancheReceipt.AttemptedTrancheSizes) > 1 {
			rec["requested_execution_plan"] = requestedExecutionPlan
			rec["tranche_fallback"] = trancheReceipt
			executionPlan = auditedPlan
			executionPlanID = dispatchWaveExecutionPlanID(executionPlan)
			rec["execution_plan_id"] = executionPlanID
			rec["execution_plan"] = executionPlan
		}
		if auditErr != nil {
			rec["execution_plan_audit"] = executionAudit
			if len(trancheReceipt.AttemptedTrancheSizes) > 0 {
				rec["tranche_fallback"] = trancheReceipt
			}
			dispatchWaveRecordDependencyError(rec, auditErr)
			return nil, writeDispatchWaveResult(stdout, stderr, rec, *asJSON), true
		}
		rec["execution_plan_audit"] = executionAudit
		prelaunchGate := dispatchWavePrelaunchGateFromAudit(executionPlanID, executionAudit)
		rec["prelaunch_gate"] = prelaunchGate
		readyPlan := dispatchWaveReadyExecutionPlan(executionPlan, executionAudit)
		rec["ready_execution_plan"] = readyPlan
		if len(requestedIssues) > 0 {
			price = dispatchWaveApplyAuditIssueOutcomes(price, executionAudit)
			explicitRefusals = dispatchWaveMergeRefusalRows(requestedIssues, explicitRefusals, price.RefusedIssues)
			price = dispatchWaveMergeIssueRefusals(price, explicitRefusals)
			rec["price"] = price
			dispatchWaveAttachExplicitIssueReceipt(rec, price)
		}
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
		return nil, writeDispatchWaveResult(stdout, stderr, rec, *asJSON), true
	}

	return executionPlan, 0, false
}

type dispatchWaveExecutionRequest struct {
	root           string
	plan           []dispatchWaveExecutionPlan
	maxWorkers     *int
	excludeLane    *string
	live           *bool
	settleSeconds  *float64
	codexLoopGate  *string
	gateSinceHours *float64
	gateLimit      *int
	asJSON         *bool
	record         map[string]any
}

func executeDispatchWavePlan(stdout, stderr io.Writer, req dispatchWaveExecutionRequest) int {
	root, executionPlan := req.root, req.plan
	maxWorkers, excludeLane, live := req.maxWorkers, req.excludeLane, req.live
	settleS, codexLoopGate := req.settleSeconds, req.codexLoopGate
	codexLoopGateSinceHours, codexLoopGateLimit := req.gateSinceHours, req.gateLimit
	asJSON, rec := req.asJSON, req.record
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
		action := dispatchMapString(payload, "action")
		if action == "spawned" || action == "enrolled" {
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
