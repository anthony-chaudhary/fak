package main

import (
	"fmt"
	"io"
	"strings"
)

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
	if admission, ok := rec["finish_first_admission"].(dispatchFinishFirstAdmission); ok {
		fmt.Fprintf(&b, "  finish_first: state=%s fresh_allowed=%d fresh_denied=%d finishers_allowed=%d override=%t recovery=%d/%d\n",
			admission.State, admission.AllowedFreshStarts, admission.DeniedFreshStarts,
			admission.AllowedFinishers, admission.Override,
			admission.Recovery.ObservedConvergingWindows, admission.Recovery.RequiredConvergingWindows)
		if admission.Reason != "" {
			fmt.Fprintf(&b, "    reason=%s\n", admission.Reason)
		}
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
