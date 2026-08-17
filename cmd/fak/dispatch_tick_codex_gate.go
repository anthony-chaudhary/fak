package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

var diagnoseRecentCodexLoopsForGate = diagnoseRecentCodexLoops

func dispatchCodexLoopGateForTick(opts dispatchTickOptions, account dispatchtick.Account, launchSafe ...bool) (map[string]any, bool, error) {
	if opts.Backend != "codex" {
		return nil, false, nil
	}
	threshold := strings.ToLower(strings.TrimSpace(opts.CodexLoopGate))
	if threshold == "" {
		threshold = dispatchCodexLoopGateDefaultThreshold()
	}
	if threshold == "" || threshold == "off" || threshold == "none" || threshold == "false" || threshold == "0" {
		return nil, false, nil
	}
	if _, ok := codexLoopFailOnRank(threshold); !ok {
		return nil, false, fmt.Errorf("invalid --codex-loop-gate %q (want loop, action, or off)", opts.CodexLoopGate)
	}
	childSafetyKnown := len(launchSafe) > 0
	safeLaunch := childSafetyKnown && launchSafe[0]
	var current *codexLoopDiagnosis
	if d, ok, err := diagnoseCurrentCodexLoop(account.Dir); ok {
		if err != nil {
			return nil, false, fmt.Errorf("Codex current-thread gate audit failed: %w", err)
		}
		current = &d
		if !childSafetyKnown && codexLoopDiagnosisUnguarded(d) {
			return dispatchCodexLoopCurrentThreadPayload(d, threshold), true, nil
		}
	}
	limit := opts.CodexLoopGateLimit
	if limit <= 0 {
		limit = dispatchCodexLoopGateDefaultLimitValue()
	}
	rep, err := diagnoseRecentCodexLoopsForGate(account.Dir, opts.CodexLoopGateSinceHours, limit)
	if err != nil {
		return nil, false, fmt.Errorf("Codex loop gate audit failed: %w", err)
	}
	lifecycle := classifyLoopStates(rep)
	payload := dispatchCodexLoopGatePayload(rep, threshold)
	payload["lifecycle"] = loopStatePayload(lifecycle)
	gateCode := 0
	if len(lifecycle.Live) > 0 {
		gateCode, _ = codexLoopFailOnExitCode("LOOP", threshold)
	}
	if len(lifecycle.Ambiguous) > 0 {
		gateCode = 1
		payload["verdict"] = "AMBIGUOUS"
		payload["reason"] = "loop session liveness is ambiguous"
		payload["next_action"] = "run fak sessions codex-loop --recent --json, then reconcile or clean terminal session registrations before retrying"
	}
	if rep.LoopCount > 0 && len(lifecycle.Terminal) == rep.LoopCount {
		payload["verdict"] = "OK"
		payload["reason"] = "historical terminal loops remain visible but do not block a fresh guarded child"
		payload["next_action"] = "spawn the prepared child through fak guard"
	}
	if childSafetyKnown {
		payload["launch"] = map[string]any{"guarded": safeLaunch}
		if current != nil {
			payload["parent"] = map[string]any{
				"source":          "current_thread",
				"session_id":      current.SessionID,
				"model_provider":  current.ModelProvider,
				"guard_witnessed": current.GuardWitnessed,
			}
		}
		if gateCode != 0 {
			payload["action"] = "refused"
			payload["next_action"] = firstString(rep.NextAction, "change approach before retrying the prepared child")
			return payload, true, nil
		}
		if !safeLaunch {
			payload["fail_on"] = "unguarded"
			payload["verdict"] = "OK"
			payload["reason"] = "codex_session_bypassed_fak_guard"
			payload["action"] = "refused"
			payload["next_action"] = "prepare the child through fak guard before spawning"
			if current != nil {
				payload["verdict"] = current.Verdict
			}
			return payload, true, nil
		}
		payload["action"] = "spawned"
		if current != nil && codexLoopDiagnosisUnguarded(*current) {
			payload["next_action"] = "spawn the prepared child through fak guard"
		} else {
			payload["next_action"] = "spawn the prepared child"
		}
	}
	return payload, false, nil
}

func dispatchCodexLoopGateDefaultThreshold() string {
	return firstString(strings.TrimSpace(os.Getenv("FLEET_CODEX_LOOP_GATE")), "loop")
}

func dispatchCodexLoopGateDefaultSinceHoursValue() float64 {
	raw := strings.TrimSpace(os.Getenv("FLEET_CODEX_LOOP_GATE_SINCE_HOURS"))
	if raw == "" {
		return dispatchCodexLoopGateDefaultSinceHours
	}
	if n, err := strconv.ParseFloat(raw, 64); err == nil && n >= 0 {
		return n
	}
	return dispatchCodexLoopGateDefaultSinceHours
}

func dispatchCodexLoopGateDefaultLimitValue() int {
	raw := strings.TrimSpace(os.Getenv("FLEET_CODEX_LOOP_GATE_LIMIT"))
	if raw == "" {
		return dispatchCodexLoopGateDefaultLimit
	}
	if n, err := strconv.Atoi(raw); err == nil && n > 0 {
		return n
	}
	return dispatchCodexLoopGateDefaultLimit
}

func dispatchCodexLoopGatePayload(rep codexLoopRecentReport, threshold string) map[string]any {
	return map[string]any{
		"id":           "codex_loop",
		"evaluated":    true,
		"schema":       rep.Schema,
		"source":       "recent",
		"codex_home":   rep.CodexHome,
		"fail_on":      codexLoopFailOnName(threshold),
		"verdict":      rep.Verdict,
		"reason":       rep.Reason,
		"since_hours":  rep.SinceHours,
		"limit":        rep.Limit,
		"scanned":      rep.Scanned,
		"loop_count":   rep.LoopCount,
		"action_count": rep.ActionCount,
		"ok_count":     rep.OKCount,
		"tool_calls":   rep.ToolCalls,
		"tool_outputs": rep.ToolOutputs,
		"top_repeated": rep.TopRepeated,
	}
}

func dispatchCodexLoopCurrentThreadPayload(d codexLoopDiagnosis, threshold string) map[string]any {
	return map[string]any{
		"id":               "codex_loop",
		"evaluated":        true,
		"schema":           d.Schema,
		"source":           "current_thread",
		"fail_on":          "unguarded",
		"threshold":        codexLoopFailOnName(threshold),
		"verdict":          d.Verdict,
		"reason":           codexLoopDiagnosisGateReason(d, "unguarded"),
		"path":             d.Path,
		"session_id":       d.SessionID,
		"model_provider":   d.ModelProvider,
		"tool_calls":       d.ToolCalls,
		"tool_outputs":     d.ToolOutputs,
		"repeated_outputs": d.RepeatedOutcomes,
	}
}
