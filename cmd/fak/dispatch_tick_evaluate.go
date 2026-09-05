package main

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func evaluateDispatchTick(opts dispatchTickOptions, stderr io.Writer) (map[string]any, error) {
	// Keep the host-probe shell-reuse arm explicit at the orchestration seam; tests and
	// operators can verify the optimization remains reachable after this function split.
	dispatchArmHostProbeShellReuse(opts.HostProbeShellReuse)
	closeHostProbeShells := dispatchArmHostProbeShellReuse(opts.HostProbeShellReuse)
	defer closeHostProbeShells()
	prep, err := prepareDispatchTickEvaluation(&opts, stderr)
	if err != nil {
		return nil, err
	}
	if prep.terminal != nil {
		return prep.terminal, nil
	}
	root, runsDir, t0, timings := prep.root, prep.runsDir, prep.started, prep.timings
	spawnStart, reg, witnessedSlots := prep.spawnStart, prep.registry, prep.witnessedSlots
	witnessRecords, freshWitnessRecords := prep.witnessRecords, prep.freshWitnessRecords
	heldNoCommit, recoverableNoCommit := prep.heldNoCommit, prep.recoverableNoCommit
	pre, account := prep.preflight, prep.account

	tPick := time.Now()
	pickRes, err := resolveDispatchTickPick(root, stderr, opts, runsDir, heldNoCommit, recoverableNoCommit)
	if err != nil {
		return nil, err
	}
	dispatchStampMs(timings, "lane_pick", tPick)
	pick := pickRes.pick
	held := pickRes.held
	liveIssueDetails := pickRes.liveIssueDetails
	liveScopes := pickRes.liveScopes
	target := pickRes.target
	hasTarget := pickRes.hasTarget
	var leaseReroute map[string]any
	// Automatic dry and live ticks share one launch plan. If its first choice is
	// already leased, spend one bounded re-pick with that lane excluded instead of
	// returning or executing a plan already known to lose. Explicit --lane remains
	// exact; the live path still acquires and launches only the final single tree.
	if dispatchShouldRerouteLeasedLane(opts, pick) {
		firstLease := inspectDispatchLaneLease(root, pick.Lane, pick.Tree, opts.Goal)
		if refused, _ := firstLease["refused"].(bool); refused {
			rerouteOpts := opts
			rerouteOpts.ExcludeLanes = append(append([]string(nil), opts.ExcludeLanes...), pick.Lane)
			rerouteStart := time.Now()
			if alternate, rerouteErr := resolveDispatchTickPick(root, stderr, rerouteOpts, runsDir, heldNoCommit, recoverableNoCommit); rerouteErr == nil && alternate.pick.Lane != "" {
				leaseReroute = map[string]any{"from_lane": pick.Lane, "lease": firstLease, "to_lane": alternate.pick.Lane}
				pickRes = alternate
				pick, held, liveIssueDetails, liveScopes = alternate.pick, alternate.held, alternate.liveIssueDetails, alternate.liveScopes
				target, hasTarget = alternate.target, alternate.hasTarget
			}
			dispatchStampMs(timings, "lease_reroute", rerouteStart)
		}
	}

	tSeed := time.Now()
	payload := seedDispatchTickPayload(root, opts, reg, pre, account, pickRes)
	if leaseReroute != nil {
		payload["lease_reroute"] = leaseReroute
	}
	if selection, ok := dispatchTickSeatSelection(root, opts.WorkKind, dispatchtick.ProductForBackend(opts.Backend), account.Tag); ok {
		payload["seat_selection"] = selection
	}
	dispatchStampMs(timings, "startup_bundle", tSeed)
	if hasTarget {
		payload["target_issue"] = target
	}
	if next, ok := dispatchtick.ModelDowngradeReDispatch(witnessRecords, workerDowngradeChain(opts.Backend))[target]; ok {
		source := "durable"
		for _, rec := range freshWitnessRecords {
			if rec.Issue == target {
				source = "fresh"
				break
			}
		}
		payload["model_recovery_evidence"] = map[string]any{"source": source, "next_model": next}
	}
	// Surface the slot witness only on a live tick where the sweep ran, and the
	// structurally-held issues only when something is actually held, so the common
	// dry-run / nothing-held payloads stay byte-identical to before (#1396).
	if opts.Live {
		payload["witnessed_slots"] = witnessedSlots
		// #2523: the spine-first fan-out default, wired at the ONE seam that knows a spine
		// just shipped. Plan-only and fail-open (see dispatch_fanout.go); a turn that
		// shipped nothing adds no key, so an idle payload is unchanged.
		if fan := dispatchIssueFanout(root, runsDir, witnessRecords, time.Now()); fan != nil {
			payload["issue_fanout"] = fan
		}
	}
	if len(heldNoCommit) > 0 {
		payload["held_no_commit"] = sortedSet(heldNoCommit)
	}

	finish := func(p map[string]any) map[string]any {
		// The spawn phase (dispatchTickLiveSpawn / dispatchTickHostEnroll) calls finish
		// internally, so its duration is stamped here from the start captured just before
		// that tail call -- only on the live-spawn path where spawnStart was set.
		if !spawnStart.IsZero() {
			dispatchStampMs(timings, "spawn", spawnStart)
		}
		timings["total"] = time.Since(t0).Milliseconds()
		p["timings_ms"] = timings
		// #3405: report the reuse dividend for the tick that earned it, read here (inside the
		// funnel) while the rack is still alive -- the deferred teardown drops it on the way
		// out. tasks_run is the no-reuse cost this tick would have paid (one process per
		// probe, the before), cold_spawns is what it actually paid (the after), and
		// spawns_avoided is the difference: the ConPTY/conhost creations that did not happen.
		// A tick that ran no warm task at all -- the spine off, or any non-Windows GOOS, whose
		// probes never touch PowerShell -- adds no key, so those payloads stay byte-identical.
		if st := dispatchHostProbeRackStats(); st.TasksRun > 0 {
			p["host_probe_shells"] = map[string]any{
				"cold_spawns":       st.ColdSpawns,
				"warm_reuses":       st.WarmReuses,
				"tasks_run":         st.TasksRun,
				"unhealthy_retired": st.UnhealthyRetired,
				"spawns_avoided":    st.SpawnsAvoided(),
			}
		}
		if opts.RecordLoop {
			p["loop_ledger"] = recordDispatchTickLoop(root, opts.LoopLedger, p)
		}
		return p
	}

	if pick.Lane == "" {
		// All-self-source edge case (#1397): the auto-pick found candidate lanes but
		// every one was held as the trust-critical witness machinery (the adjudicator,
		// policy, kernel, shipgate/architest gates -- the referee's own trees) under
		// guard, so `chosen` stayed "". This is NOT an empty/error router -- the backlog
		// is real, it is just all the one narrow set a self-guarded RSI worker must never
		// SHIP an edit to (rewriting its own referee). Say so honestly with the
		// SELF_MODIFY_HOLD vocabulary (over the whole held set) instead of the misleading
		// "router empty/error" NO_LANE, so the operator routes the work to an unguarded/
		// operator or worktree-isolated path (#1334). (Merely-self-source lanes -- gateway,
		// agent, cmd, ... -- are NOT held: the worker guard permits shipping those.)
		if len(pick.SelfSourceHeld) > 0 {
			payload["ok"] = false
			payload["action"] = "self_modify_hold"
			payload["verdict"] = "SELF_MODIFY_HOLD"
			payload["self_modify_held_lanes"] = append([]string(nil), pick.SelfSourceHeld...)
			payload["reason"] = fmt.Sprintf("every candidate lane (%s) is rooted in fak's trust-critical witness machinery (the adjudicator/policy/kernel/shipgate the referee binds to): a guarded %s worker can investigate but must never SHIP an edit that rewrites its own referee (reason=SELF_MODIFY), so this narrow set is operator-gated -- route it to an unguarded/operator or worktree-isolated path (#1334), not a self-guarded worker", strings.Join(pick.SelfSourceHeld, ", "), opts.Backend)
			return finish(payload), nil
		}
		payload["ok"] = false
		payload["action"] = "no_lane"
		payload["verdict"] = "NO_LANE"
		payload["reason"] = "no lane has open issues (router empty/error)"
		return finish(payload), nil
	}
	if live, ok := inFlightDuplicateForPick(opts, pick.Numbers, hasTarget, liveIssueDetails); ok {
		payload["ok"] = false
		payload["action"] = "in_flight_duplicate"
		payload["verdict"] = "IN_FLIGHT_DUPLICATE"
		payload["target_issue"] = live.Issue
		payload["in_flight_duplicate"] = dispatchLiveScopeMap(live)
		payload["reason"] = fmt.Sprintf("issue #%d already has live worker %s (pid %d, lease %q)", live.Issue, live.Worker, live.PID, live.LeaseID)
		return finish(payload), nil
	}
	if hasTarget {
		if live, ok := treeCollisionFromScopes(liveScopes, pick.Tree); ok {
			payload["ok"] = false
			payload["action"] = "collision_risk"
			payload["verdict"] = dispatchorder.ReasonCollisionRisk
			payload["live_collision"] = map[string]any{
				"issue": live.Issue,
				"lane":  live.Lane,
				"tree":  append([]string(nil), live.Tree...),
				"log":   live.Log,
			}
			payload["reason"] = fmt.Sprintf("candidate issue #%d tree %v overlaps live worker issue #%d lane %q tree %v", target, pick.Tree, live.Issue, live.Lane, live.Tree)
			return finish(payload), nil
		}
	}
	if opts.Lane != "" && held[pick.Lane] && opts.TargetIssue == 0 {
		payload["ok"] = false
		payload["action"] = "lane_busy"
		payload["verdict"] = "LANE_BUSY"
		payload["reason"] = fmt.Sprintf("lane %q already has a live resolution worker", pick.Lane)
		return finish(payload), nil
	}
	if !hasTarget {
		payload["ok"] = false
		payload["action"] = "no_issue"
		payload["verdict"] = "NO_ISSUE"
		reason := fmt.Sprintf("every open issue on lane %q is live or cooling", pick.Lane)
		if len(heldNoCommit) > 0 {
			reason = fmt.Sprintf("every open issue on lane %q is live, cooling, or held after a structural guard refusal (held: %v)", pick.Lane, sortedSet(heldNoCommit))
		}
		payload["reason"] = reason
		return finish(payload), nil
	}

	tPrompt := time.Now()
	return completeDispatchTickEvaluation(root, runsDir, opts, stderr, pick, pickRes, account, target, witnessRecords, payload, finish, timings, spawnStart, tPrompt)

}
func completeDispatchTickEvaluation(root, runsDir string, opts dispatchTickOptions, stderr io.Writer, pick dispatchLanePick, pickRes dispatchTickPick, account dispatchtick.Account, target int, witnessRecords []dispatchtick.WitnessRecord, payload map[string]any, finish func(map[string]any) map[string]any, timings map[string]int64, spawnStart, tPrompt time.Time) (map[string]any, error) {
	// #4167: hand dispatchPrompt the router-fetched row for the selected target so it
	// reuses the already-fetched body instead of a second `gh issue view`. A cache miss
	// (unrouted --target-issue) yields the zero value, and dispatchPrompt falls back.
	promptRec, err := dispatchPrompt(root, stderr, target, pick.Lane, pick.IssueByNumber[target])
	if err != nil {
		return nil, err
	}
	contract := dispatchtick.ParseObjectiveContract(dispatchMapString(promptRec, "body"))
	if contract.Refusal != "" {
		payload["ok"] = false
		payload["action"] = "objective_contract_refused"
		payload["verdict"] = contract.Refusal
		payload["reason"] = "issue objective has no attached Witness scorer"
		payload["objective_contract"] = contract
		return finish(payload), nil
	}
	dispatchStampMs(timings, "prompt", tPrompt)
	promptChars := dispatchMapInt(promptRec, "prompt_chars")
	labels := dispatchStringSlice(promptRec["labels"])
	payload["prompt_chars"] = promptChars
	if receipt, ok := promptRec["repo_pulse_receipt"].(map[string]any); ok {
		payload["repo_pulse_receipt"] = receipt
	}
	payload["issue_title"] = dispatchMapString(promptRec, "title")
	payload["development_branch"] = dispatchMapString(promptRec, "development_branch")
	payload["child_curve"] = dispatchtick.ChildCurve(root, target)
	if errText := dispatchMapString(promptRec, "branch_role_error"); errText != "" {
		payload["branch_role_error"] = errText
	}
	if warning := dispatchMapString(mapAt(payload, "stale_base"), "warning"); warning != "" {
		prompt := dispatchMapString(promptRec, "prompt") + "\n\nworker preflight warning:\n- " + warning + "\n"
		promptRec["prompt"] = prompt
		promptRec["prompt_chars"] = len(prompt)
		payload["worker_preflight_warning"] = warning
		promptChars = len(prompt)
		payload["prompt_chars"] = promptChars
	}
	// #2030 gen/second-next: the micro backend enrolls this routed issue into the
	// in-process microagent host (internal/microagent, M2) instead of exec-spawning a
	// detached guarded CLI. It runs AFTER the shared duplicate/collision/lane-busy/
	// no-issue gates (so tree-safety is decided identically) and BEFORE the CLI-only
	// model/guard/command machinery below (BuildWorkerCommand refuses micro). Opt-in
	// only — a default claude/opencode/codex tick never enters here.
	if dispatchtick.IsMicroBackend(opts.Backend) {
		spawnStart = time.Now()
		return dispatchTickHostEnroll(root, runsDir, opts, pick, pickRes.leaseID, account, target, payload, finish), nil
	}
	launch, launchPreview, guardedPreview, err := prepareDispatchWorkerCommand(root, opts, pick, account, target, promptChars, labels, witnessRecords, payload)
	if err != nil {
		return nil, err
	}

	if !applyDispatchModelAcceptance(opts.AcceptanceArtifact, launch.Model, labels, time.Now().UTC(), opts.AcceptanceOverride, payload) {
		return finish(payload), nil
	}

	// Self-modify pre-route (#1397): a GUARDED worker aimed at the trust-critical witness
	// machinery (the adjudicator/policy/kernel/shipgate -- the referee's own trees) can
	// investigate but must never SHIP -- rewriting its own referee is the RSI hazard, so
	// hold rather than let it burn turns. NOTE the correction to the original #1338 model:
	// the hold is NOT the whole cmd/**+internal/** module. `fak guard` (no --policy) runs
	// the embedded guard-default-policy.json, whose self_modify_globs are secrets/dotfiles
	// only -- it names no cmd/ or internal/ code tree, so a guarded worker DOES ship
	// gateway/agent/cmd/... normally. Only the trust-critical subset is held. The hold
	// fires on TWO signals: the lane tree is trust-critical (a correctly-routed
	// adjudicator/policy/... lane), OR the target issue's own text references that
	// machinery even though it routed to a SAFE lane -- the MIS-ROUTE the router's path
	// extractor hides (a `fix(policy):` issue whose real work is internal/policy aliases
	// to the tools lane carrying zero extracted paths). Hold BEFORE both the dry-run plan
	// and the live spawn so the loop honest-STOPs and the operator routes it to an
	// unguarded/operator or worktree-isolated path (#1334). An unguarded worker
	// (FLEET_DOGFOOD_GUARD=0, or a worktree-isolated path) never trips this.
	issueText := dispatchMapString(promptRec, "title") + "\n" + dispatchMapString(promptRec, "body")
	if held, tree := dispatchtick.SelfModifyHoldForPick(guardedPreview, pick.Tree, issueText); held {
		payload["ok"] = false
		payload["action"] = "self_modify_hold"
		payload["verdict"] = "SELF_MODIFY_HOLD"
		payload["self_modify_tree"] = tree
		payload["reason"] = fmt.Sprintf("issue #%d targets fak's trust-critical witness machinery (lane %q, tree %q -- the adjudicator/policy/kernel/shipgate the referee binds to): a guarded %s worker can investigate but must never SHIP an edit that rewrites its own referee (reason=SELF_MODIFY), so this work is operator-gated -- route it to an unguarded/operator or worktree-isolated path (#1334), not a self-guarded worker", target, pick.Lane, tree, opts.Backend)
		return finish(payload), nil
	}

	dryRunGrant := launchSpawnBroker(newLaunchBrokerAttempt("dispatch_tick", opts.Backend, launchPreview, nil, root))
	payload["spawn_broker"] = launchBrokerGrantMap(dryRunGrant)
	if !dryRunGrant.Allow {
		payload["ok"] = false
		payload["action"] = "broker_denied"
		payload["verdict"] = "SPAWN_BROKER_DENIED"
		payload["reason"] = "spawn broker denied dispatch worker launch: " + dryRunGrant.Reason
		return finish(payload), nil
	}

	gates := map[string]any{}
	var preflightReq dispatchWorkerPreflightRequest
	var workerPreflight *dispatchWorkerPreflightResult
	if opts.Backend == "codex" {
		preflightReq = newDispatchWorkerPreflightRequest(root, opts, account, launch, launchPreview, guardedPreview)
		ctx, cancel := context.WithTimeout(context.Background(), dispatchWorkerPreflightTimeout)
		result := dispatchWorkerPreflight(ctx, preflightReq, time.Now().UTC())
		cancel()
		workerPreflight = &result
		payload["worker_preflight"] = result.Map()
		gates["worker_identity"] = result.Map()
		if !result.AllowsStartup() {
			payload["ok"] = false
			payload["action"] = "worker_preflight_refused"
			payload["verdict"] = result.Verdict
			payload["reason"] = result.Reason
			payload["admitted_workers"] = 0
			payload["launch_checks"] = gates
			return finish(payload), nil
		}
	}

	providerCheck := dispatchProviderReachabilityCheck(launchPreview)
	gates["provider_reachability"] = providerCheck
	payload["launch_checks"] = gates
	if evaluated, _ := providerCheck["evaluated"].(bool); evaluated {
		if ok, _ := providerCheck["ok"].(bool); !ok && !(workerPreflight != nil && !workerPreflight.Ready && workerPreflight.AllowsStartup() && dispatchTransientProviderCheck(providerCheck)) {
			payload["ok"] = false
			payload["action"] = "provider_unreachable"
			payload["verdict"] = "PROVIDER_REACHABILITY_REFUSED"
			payload["reason"] = dispatchMapString(providerCheck, "reason")
			return finish(payload), nil
		}
	}
	if gate, refused, err := dispatchCodexLoopGateForTick(opts, account, guardedPreview); err != nil {
		return nil, err
	} else if gate != nil {
		payload["codex_loop_gate"] = gate
		gates["codex_loop"] = gate
		if refused {
			payload["ok"] = false
			payload["action"] = "codex_loop_gate_refused"
			payload["verdict"] = "CODEX_LOOP_GATE_REFUSED"
			payload["reason"] = fmt.Sprintf("Codex loop gate refused dispatch: fail-on=%s verdict=%s reason=%s",
				dispatchMapString(gate, "fail_on"), dispatchMapString(gate, "verdict"), dispatchMapString(gate, "reason"))
			return finish(payload), nil
		}
	}

	// Focus WIP-breadth backpressure (#3223): warn-first. When the fleet is at/over the
	// focusscore WIP cap AND this spawn OPENS a new objective, attach the FOCUS_WIP_SATURATED
	// advisory so `fak dispatch status` / the tick JSON surface it distinctly from the
	// rate-limit and collision holds. Default posture WARN (advise + still spawn, so the
	// live fleet is byte-identical below cap and on a continuation); the --focus-hold /
	// FLEET_DISPATCH_FOCUS_HOLD posture instead REFUSES opening the new objective. It never
	// blocks a continuation of an already-open objective and runs AFTER every higher-precedence
	// gate, so it is the last, narrowest throttle before a genuinely new concurrent objective.
	if focusAdm := dispatchEvaluateFocus(root, dispatchFocusHoldPosture(opts), target); focusAdm.Advise {
		payload["focus"] = focusAdm.Map()
		if focusAdm.Hold {
			payload["ok"] = false
			payload["action"] = "focus_hold"
			payload["verdict"] = dispatchtick.FocusWIPSaturated
			payload["reason"] = focusAdm.Reason
			return finish(payload), nil
		}
	}

	if !opts.Live {
		lease := inspectDispatchLaneLease(root, pick.Lane, pick.Tree, opts.Goal)
		if applyDispatchLaneLease(payload, lease, fmt.Sprintf("lane %q lease is held by a live peer", pick.Lane)) {
			return finish(payload), nil
		}
		payload["ok"] = true
		payload["action"] = "would_spawn"
		payload["verdict"] = "WOULD_SPAWN"
		payload["reason"] = fmt.Sprintf("safe to spawn 1 %s issue-resolution worker on #%d (lane %q) under account %q", opts.Backend, target, pick.Lane, account.Tag)
		payload = finish(payload)
		recordDispatchPayload(runsDir, opts.Backend, payload)
		return payload, nil
	}

	spawnStart = time.Now()
	return dispatchTickLiveSpawn(root, runsDir, opts, pick, pickRes.leaseID, account, launch, preflightReq, workerPreflight, target, promptRec, payload, finish)
}
